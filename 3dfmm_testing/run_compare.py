#!/usr/bin/env python3
"""Run analytical vs 3dfmm comparisons and print a debug-friendly summary."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAIRS = [
    ("bates", "analytical_bates.ric", "3dfmm_bates.ric"),
    ("conical", "analytical_conical.ric", "3dfmm_conical.ric"),
    ("finocyl", "analytical_finocyl.ric", "3dfmm_finocyl.ric"),
]


@dataclass
class CaseResult:
    name: str
    ok: bool
    return_code: int
    csv_path: Path
    log_path: Path
    error: str = ""
    error_detail: str = ""
    burn_time: float = math.nan
    peak_force: float = math.nan
    peak_pressure: float = math.nan
    total_impulse: float = math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare 3dfmm test motors")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to openMotor repository root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
        help="Directory where CSV and logs are written",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining runs even if one fails",
    )
    return parser.parse_args()


def _find_column(rows: list[dict[str, str]], prefixes: Iterable[str]) -> list[float]:
    if not rows:
        raise ValueError("CSV has no data rows")

    keys = list(rows[0].keys())
    lowered = {k.lower(): k for k in keys}

    for prefix in prefixes:
        p = prefix.lower()
        for lk, original in lowered.items():
            if lk.startswith(p):
                out = []
                for row in rows:
                    value = row.get(original, "")
                    if value != "":
                        out.append(float(value))
                if not out:
                    raise ValueError(f"Column '{original}' has no numeric rows")
                return out

    raise KeyError(f"Could not find any column matching prefixes: {list(prefixes)}")


def summarize_csv(csv_path: Path) -> tuple[float, float, float, float]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    t = _find_column(rows, ["time"])
    fvals = _find_column(rows, ["force", "thrust"])
    pvals = _find_column(rows, ["chamber pressure", "pressure"])

    burn_time = t[-1]
    peak_force = max(fvals)
    peak_pressure = max(pvals)

    impulse = 0.0
    limit = min(len(t), len(fvals))
    for i in range(1, limit):
        dt = t[i] - t[i - 1]
        impulse += 0.5 * (fvals[i] + fvals[i - 1]) * dt

    return burn_time, peak_force, peak_pressure, impulse


def run_case(repo_root: Path, testing_dir: Path, out_dir: Path, label: str, ric_name: str) -> CaseResult:
    safe_name = label.replace("/", "_").replace(" ", "_")
    csv_path = out_dir / f"{safe_name}.csv"
    log_path = out_dir / "logs" / f"{safe_name}.log"

    ric_path = testing_dir / ric_name
    cmd = [sys.executable, "main.py", "-h", "-o", str(csv_path), str(ric_path)]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(f"return_code={proc.returncode}\n\n")
        f.write("[stdout]\n")
        f.write(proc.stdout)
        f.write("\n[stderr]\n")
        f.write(proc.stderr)

    if proc.returncode != 0:
        detail = ""
        for line in reversed(combined_output.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            if "[ERR]" in stripped or "Traceback" in stripped or "Exception" in stripped:
                detail = stripped
                break
        if detail == "":
            for line in reversed(combined_output.splitlines()):
                stripped = line.strip()
                if stripped:
                    detail = stripped
                    break

        return CaseResult(
            name=label,
            ok=False,
            return_code=proc.returncode,
            csv_path=csv_path,
            log_path=log_path,
            error="non-zero exit",
            error_detail=detail,
        )

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return CaseResult(
            name=label,
            ok=False,
            return_code=proc.returncode,
            csv_path=csv_path,
            log_path=log_path,
            error="CSV missing or empty",
            error_detail="headless run completed but did not emit a CSV file",
        )

    try:
        burn_time, peak_force, peak_pressure, total_impulse = summarize_csv(csv_path)
    except Exception as exc:
        return CaseResult(
            name=label,
            ok=False,
            return_code=proc.returncode,
            csv_path=csv_path,
            log_path=log_path,
            error=f"summary parse error: {exc}",
            error_detail="CSV was written but parsing failed; inspect header/data formatting",
        )

    return CaseResult(
        name=label,
        ok=True,
        return_code=proc.returncode,
        csv_path=csv_path,
        log_path=log_path,
        burn_time=burn_time,
        peak_force=peak_force,
        peak_pressure=peak_pressure,
        total_impulse=total_impulse,
    )


def pct_delta(a: float, b: float) -> float:
    if a == 0:
        return math.nan
    return (b - a) / a * 100.0


def format_metric(name: str, a: float, b: float) -> str:
    return (
        f"  {name:14s} analytical={a:12.5f} 3dfmm={b:12.5f} "
        f"delta={pct_delta(a, b):9.3f}%"
    )


def write_summary(out_dir: Path, lines: list[str]) -> Path:
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    testing_dir = repo_root / "3dfmm_testing"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    all_lines.append(f"repo_root: {repo_root}")
    all_lines.append(f"out_dir: {out_dir}")
    all_lines.append("")

    had_failures = False

    for shape, analytical_ric, fmm_ric in PAIRS:
        all_lines.append(f"=== {shape.upper()} ===")

        analytical = run_case(repo_root, testing_dir, out_dir, f"analytical_{shape}", analytical_ric)
        fmm = run_case(repo_root, testing_dir, out_dir, f"3dfmm_{shape}", fmm_ric)

        for case in (analytical, fmm):
            status = "OK" if case.ok else "FAIL"
            all_lines.append(
                f"- {case.name:18s} {status:4s} rc={case.return_code} csv={case.csv_path.name} log={case.log_path.name}"
            )
            if not case.ok:
                had_failures = True
                all_lines.append(f"  error: {case.error}")
                if case.error_detail:
                    all_lines.append(f"  detail: {case.error_detail}")

        if analytical.ok and fmm.ok:
            all_lines.append(format_metric("burn_time", analytical.burn_time, fmm.burn_time))
            all_lines.append(format_metric("peak_force", analytical.peak_force, fmm.peak_force))
            all_lines.append(format_metric("peak_pressure", analytical.peak_pressure, fmm.peak_pressure))
            all_lines.append(format_metric("total_impulse", analytical.total_impulse, fmm.total_impulse))

        all_lines.append("")

        if had_failures and not args.keep_going:
            all_lines.append("Stopped early after failure. Re-run with --keep-going to continue.")
            break

    summary_path = write_summary(out_dir, all_lines)
    print("\n".join(all_lines))
    print(f"Summary written to: {summary_path}")

    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
