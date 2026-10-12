# Tickets: native Metal windows and the Linux side

Goal: one translated-vs-native Metal table for ONE chip.
The tool is `compare.py` in this directory.
The native side runs on macOS. The translated side runs on the Linux GPU host.
Both sides must be the same chip. Both must be quiet.
Both must stay on one CPU governor for the whole session.
No table gets published from a mixed pair.

## macOS window A: 16-inch M1 Max

The lab pane boots this Mac into macOS on AC power.
Work over `ssh 16m1mbp-macos`.
After boot, touch nothing for 5 minutes.
A run started seconds after boot lands on load above 10.
Those numbers are unusable.

```sh
ssh 16m1mbp-macos
git clone https://github.com/joshuaswarren/metal2vk.git
cd metal2vk/bench/native-metal

# 1. system record, after the 5-minute settle
sw_vers > system.txt
system_profiler -detailLevel mini SPHardwareDataType >> system.txt
sysctl -n hw.memsize >> system.txt
uptime >> system.txt
pmset -g therm >> system.txt

# 2. three back-to-back runs. The JSON name is dated.
#    Every run overwrites it. Rename between runs, or you keep only run 3.
bash run.sh 2>&1 | tee native-run1.txt
cp results/results-native-metal-*.json results/native-run1.json
bash run.sh 2>&1 | tee native-run2.txt
cp results/results-native-metal-*.json results/native-run2.json
bash run.sh 2>&1 | tee native-run3.txt
cp results/results-native-metal-*.json results/native-run3.json

# 3. the commit the harness came from
git rev-parse HEAD > commit.txt
```

Each `run.sh` prints the chip and macOS version, then the load average, then all 12 rows.
Every check field must say `OK`. A failed check exits nonzero.
A run takes 15 to 60 seconds.
Keep the Mac quiet for the whole window.
No other GPU work. No builds. No model serving.

Send back:

- `native-run1.txt`, `native-run2.txt`, `native-run3.txt`
- `results/native-run1.json`, `native-run2.json`, `native-run3.json`
- `system.txt` and `commit.txt`

## macOS window B: 13-inch M1

Same steps as window A. The lab pane boots this Mac into macOS on AC power.
Work over `ssh jw-m1-lan`.
This Mac has 16 GB of memory; the harness run is small.
No big model, no swap pressure; the whole harness fits in memory.

```sh
ssh jw-m1-lan
git clone https://github.com/joshuaswarren/metal2vk.git
cd metal2vk/bench/native-metal

# 1. system record, after the 5-minute settle
sw_vers > system.txt
system_profiler -detailLevel mini SPHardwareDataType >> system.txt
sysctl -n hw.memsize >> system.txt
uptime >> system.txt
pmset -g therm >> system.txt

# 2. three back-to-back runs, renamed as in window A
bash run.sh 2>&1 | tee native-run1.txt
cp results/results-native-metal-*.json results/native-run1.json
bash run.sh 2>&1 | tee native-run2.txt
cp results/results-native-metal-*.json results/native-run2.json
bash run.sh 2>&1 | tee native-run3.txt
cp results/results-native-metal-*.json results/native-run3.json

# 3. the commit the harness came from
git rev-parse HEAD > commit.txt
```

Send back the same file set as window A.

## Linux side: 14-inch M2 Max

```sh
ssh jw14m2-linux
cd ~/src/metal2vk-ggmlsweep          # or clone the repo as in window A
git rev-parse HEAD > commit.txt

# 1. name the governor and driver BEFORE the runs
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor > governor.txt
cat governor.txt
mesa-git-sha > mesa-git.txt
cat mesa-git.txt
uname -r > kernel.txt
uptime > load-before.txt

# 2. one run. Needs UZU_ROOT, CLSPV, M2V_ICD. See bench/env-local.sh.template.
bash bench/run-all.sh /tmp/m2v-out --device g14c 2>&1 | tee linux-run1.txt
cp bench/results-g14c-*.json bench/linux-run1.json
```

The run takes a few minutes.
`run-all.sh` compiles the cases, then runs the bench, then the sweep.
The receipts land in `bench/` as `results-g14c-YYYY-MM-DD.json`.
The dated name overwrites. Rename it for each extra run.
Two or three back-to-back runs give the Linux side a median and a spread too.
Keep the machine quiet. Keep one governor for every run.
Never mix governors in one comparison.

Send back: the results JSONs, `governor.txt`, `mesa-git.txt`, `kernel.txt`, `load-before.txt`, `linux-run1.txt`, `commit.txt`.

## Pairing them

Work from `bench/native-metal/` in a checkout that holds both file sets:

```sh
python3 compare.py \
  --native results/native-run1.json results/native-run2.json results/native-run3.json \
  --linux ../linux-run1.json ../linux-run2.json \
  --meta kernel=$(cat kernel.txt) \
  --meta driver=$(cat mesa-git.txt) \
  --meta governor=$(cat governor.txt) \
  --meta ane=idle \
  --meta load=0.12
```

`compare.py` refuses on a chip mismatch. It refuses on a missing case.
It refuses on a failed check. It refuses on a native JSON without `load` and `device`.
It prints a loud warning when a recorded load average goes above 0.5.
Only a clean run produces a table worth publishing.
One chip. One governor. Driver named.
