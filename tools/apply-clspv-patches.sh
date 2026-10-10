#!/bin/bash
# Applies every script in clspv-patches/ to a clspv checkout, in the order the CI workflow uses.
# Usage: tools/apply-clspv-patches.sh /path/to/clspv   (the checkout is clspv f2b01dd6, see the README)
# The order lives in one place, .github/workflows/sweep.yml (the `for p in ...` line); this script reads it from there.
set -euo pipefail
clspv=${1:?usage: apply-clspv-patches.sh /path/to/clspv}
here=$(cd "$(dirname "$0")/.." && pwd)
workflow="$here/.github/workflows/sweep.yml"
list=$(sed -n 's/.*for p in \(.*\); do python3.*/\1/p' "$workflow")
[ -n "$list" ] || { echo "apply-clspv-patches.sh: no patch list found in $workflow" >&2; exit 1; }
for p in $list; do
  echo "== $p"
  python3 "$here/clspv-patches/$p" "$clspv"
done
