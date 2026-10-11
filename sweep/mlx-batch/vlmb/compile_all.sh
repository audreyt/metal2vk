#!/bin/sh
# Compile every assembled .metal in this directory through tools/m2v-compile.
# Run from sweep/mlx-batch: M2V_TOOL=.../tools/m2v-compile sh vlmb/compile_all.sh
set -u
: > vlmb/compile-results.txt
for m in vlmb/*.metal; do
  stem=$(basename "$m" .metal)
  out=$(ionice -c3 nice -n 19 "$M2V_TOOL" --msl "$m" --out vlmb/out --json 2>&1)
  jsonline=$(printf '%s\n' "$out" | grep -m1 '^{' || true)
  status=$(printf '%s' "$jsonline" | sed -n 's/.*"status": *"\([a-z][a-z -]*\)".*/\1/p')
  [ -n "$status" ] || status="no-json"
  err=""
  if [ "$status" != "ok" ]; then
    err=$(printf '%s' "$out" | tr ',' '\n' | grep -m1 detail | cut -c1-200)
  fi
  printf '%s\t%s\t%s\n' "$stem" "$status" "$err" >> vlmb/compile-results.txt
done
echo DONE >> vlmb/compile-results.txt
