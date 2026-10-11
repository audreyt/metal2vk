#!/usr/bin/env python3
"""Compile assembled MSL files and validate each SPIR-V module."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def entry_name(source: str) -> str:
    match = re.search(r'\[\[host_name\("([^"]+)"\)\]\]', source)
    if match:
        return match.group(1)
    match = re.search(r"\[\[kernel\]\]\s+void\s+([A-Za-z_][A-Za-z0-9_]*)", source)
    if not match:
        raise ValueError("no kernel entry or host_name instantiation")
    return match.group(1)


def comment_value(source: str, prefix: str) -> str:
    for line in source.splitlines():
        text = line.removeprefix("// ").strip()
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msl-dir", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--clspv", type=Path, required=True)
    parser.add_argument("--spirv-val", type=Path, default=None)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    validator = args.spirv_val or (Path(os.environ["SPIRV_VAL"]) if os.environ.get("SPIRV_VAL") else Path(shutil.which("spirv-val") or ""))
    if not validator or not validator.is_file():
        parser.error("set SPIRV_VAL or pass --spirv-val")
    env = os.environ.copy()
    env["CLSPV"] = str(args.clspv)
    env["M2V_CACHE_DIR"] = str(args.cache)
    rows = []
    # non-recursive: the family directories (vlmA/, vlmb/, others/, repro/) keep their own
    # tables and runners, so a per-family --msl-dir must not sweep the subdirectories
    for msl in sorted(args.msl_dir.glob("*.metal")):
        source = msl.read_text()
        name = entry_name(source)
        digest = hashlib.sha256(msl.read_bytes()).hexdigest()
        run = subprocess.run(
            [str(args.tool), "--msl", str(msl), "--out", str(args.out),
             "--name", name, "--json"],
            text=True, capture_output=True, env=env, check=False,
        )
        try:
            result = json.loads(first_line(run.stdout))
        except json.JSONDecodeError:
            result = {"status": "error", "detail": first_line(run.stderr or run.stdout)}
        status = result.get("status")
        valid = "no"
        error = ""
        giveup = "no"
        if status == "refused" or run.returncode == 3:
            giveup = result.get("refused", "M2V-GIVEUP")
            error = first_line(run.stdout or run.stderr)
        elif status == "ok" and run.returncode == 0:
            spv = Path(result["spv"])
            check = subprocess.run(
                [str(validator), "--target-env", "vulkan1.3", str(spv)],
                text=True, capture_output=True, check=False,
            )
            if check.returncode == 0:
                valid = "yes"
            else:
                error = first_line(check.stderr or check.stdout)
        else:
            error = first_line(run.stderr) or result.get("detail", "compile failed")
        rows.append((
            msl.stem,
            comment_value(source, "Origin:"),
            digest,
            valid,
            error,
            giveup,
            comment_value(source, "Representative call:") or comment_value(source, "Call shape:"),
        ))

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as output:
        output.write("kernel\torigin\tsource_sha256\tvalid\tfirst_error\tm2v_giveup\trepresentative_call\n")
        for row in sorted(rows):
            output.write("\t".join(row) + "\n")
    valid_count = sum(row[3] == "yes" for row in rows)
    print(f"assembled={len(rows)} valid={valid_count} failed={len(rows)-valid_count}")
    return 0 if valid_count == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
