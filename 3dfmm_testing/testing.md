# 3DFMM Test Runs and Comparisons

This folder contains matched motor pairs for comparing existing analytical/2D behavior against 3DFMM behavior.

## Included Motor Pairs

1. `analytical_bates.ric` vs `3dfmm_bates.ric`
2. `analytical_conical.ric` vs `3dfmm_conical.ric`
3. `analytical_finocyl.ric` vs `3dfmm_finocyl.ric`

## CLI Behavior (from app.py)

- Headless mode is enabled with `-h`.
- CSV output is written when `-o <output_file>` is provided.
- The input `.ric` file should be the last CLI argument.
- Entry point is `main.py` (it constructs `App(sys.argv)`).

## Run All 6 Motors and Export CSV

From repo root:

```bash
source .venv/bin/activate
mkdir -p 3dfmm_testing/out

python main.py -h -o 3dfmm_testing/out/analytical_bates.csv 3dfmm_testing/analytical_bates.ric
python main.py -h -o 3dfmm_testing/out/3dfmm_bates.csv      3dfmm_testing/3dfmm_bates.ric

python main.py -h -o 3dfmm_testing/out/analytical_conical.csv 3dfmm_testing/analytical_conical.ric
python main.py -h -o 3dfmm_testing/out/3dfmm_conical.csv      3dfmm_testing/3dfmm_conical.ric

python main.py -h -o 3dfmm_testing/out/analytical_finocyl.csv 3dfmm_testing/analytical_finocyl.ric
python main.py -h -o 3dfmm_testing/out/3dfmm_finocyl.csv      3dfmm_testing/3dfmm_finocyl.ric
```

## One-Command Runner (Recommended)

Use the helper script to run all cases, save per-case logs, and print deltas:

```bash
source .venv/bin/activate
python 3dfmm_testing/run_compare.py --keep-going
```

Artifacts are written to `3dfmm_testing/out`:

- `*.csv` for each motor run
- `logs/*.log` for full stdout/stderr per case
- `summary.txt` with analytical vs 3dfmm metric deltas

## Targeted 3DFMM Kernel Benchmark

Run a focused Cython-vs-fallback check for the 3dfmm hot paths:

```bash
source .venv/bin/activate
python 3dfmm_testing/benchmark_3dfmm.py --motor 3dfmm_testing/3dfmm_bates.ric --samples 24 --reps 5
```

This prints:

- max absolute differences between fallback and Cython for:
    - end position detection
    - port area
    - regressed volume
    - non-meshed mass flux
- timing/speedup for each of those method sweeps

## Quick Pairwise Comparison Script

Run this after CSVs are generated:

```bash
python - <<'PY'
import csv
from pathlib import Path

BASE = Path("3dfmm_testing/out")
PAIRS = [
    ("bates", "analytical_bates.csv", "3dfmm_bates.csv"),
    ("conical", "analytical_conical.csv", "3dfmm_conical.csv"),
    ("finocyl", "analytical_finocyl.csv", "3dfmm_finocyl.csv"),
]

def read_csv(path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    return rows

def col(rows, starts_with):
    keys = rows[0].keys()
    for k in keys:
        if k.startswith(starts_with):
            return [float(r[k]) for r in rows if r[k] != ""]
    raise KeyError(f"Missing column prefix: {starts_with}")

def summary(rows):
    t = col(rows, "Time")
    f = col(rows, "Force")
    p = col(rows, "Chamber Pressure")

    burn_time = t[-1]
    peak_force = max(f)
    peak_pressure = max(p)

    impulse = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        impulse += 0.5 * (f[i] + f[i - 1]) * dt

    return {
        "burn_time": burn_time,
        "peak_force": peak_force,
        "peak_pressure": peak_pressure,
        "total_impulse": impulse,
    }

def pct_delta(a, b):
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0

for label, a_name, b_name in PAIRS:
    a_rows = read_csv(BASE / a_name)
    b_rows = read_csv(BASE / b_name)

    a = summary(a_rows)
    b = summary(b_rows)

    print(f"\n=== {label.upper()} ===")
    for k in ["burn_time", "peak_force", "peak_pressure", "total_impulse"]:
        print(
            f"{k:14s}  analytical={a[k]:12.5f}  3dfmm={b[k]:12.5f}  "
            f"delta={pct_delta(a[k], b[k]):9.3f}%"
        )
PY
```

## What to Compare

For each pair, compare:

1. Burn time
2. Peak thrust
3. Peak chamber pressure
4. Total impulse

Then inspect thrust-time and pressure-time traces in the CSVs if deltas are larger than expected.

## Notes

- If a run prints alerts, capture them alongside the CSV when comparing.
- If headless run fails, run the same `.ric` in the GUI and inspect alerts first, then retry headless.