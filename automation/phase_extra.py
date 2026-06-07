from __future__ import annotations

import argparse
import csv as _csv
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from automation import config, runner
from automation.config import DEFAULT_BENCHMARKS, PHASE_C_CSV, PHASE_D_CSV
from automation.sweep import _append_row, _row_header

COUPLING_CAPPED = "CAPPED_MULTIPLICATIVE"

PHASE_C_BENCH_NAMES = {
    "Soundex", "Hex", "Fraction", "Precision", "NumberUtils",
    "IEEE754rUtils", "CharUtils", "Strings",
}

PHASE_D_BENCH_NAMES = {
    "ArithmeticUtils", "Complex", "BooleanUtils", "Fraction", "IntMath", "WordUtils",
}

PHASE_C_BENCHES = [b for b in DEFAULT_BENCHMARKS if b.short_name in PHASE_C_BENCH_NAMES]
PHASE_D_BENCHES = [b for b in DEFAULT_BENCHMARKS if b.short_name in PHASE_D_BENCH_NAMES]

EXTRA_C_K = (1.0, 1.5, 3.0, 4.0)
EXTRA_D_K = (2.0, 3.0)

EXTRA_C_SEEDS = (42, 1234, 2024, 7, 13, 99, 256, 777)
EXTRA_D_SEEDS = (42, 1234, 2024, 7, 13, 99)
EXTRA_CAPS = (1.0,)


def _build_runs_c():
    from itertools import product as iproduct
    runs = []
    for bench, seed, k, cap in iproduct(PHASE_C_BENCHES, EXTRA_C_SEEDS, EXTRA_C_K, EXTRA_CAPS):
        runs.append(config.RunSpec(
            benchmark=bench, seed=seed, coupling=COUPLING_CAPPED, k=k,
            diversity_cap=cap,
            search_budget_s=60, run_timeout_s=240,
            criterion="STRONGMUTATION", client_mem_mb=1500, population=5,
        ))
    return runs


def _build_runs_d():
    from itertools import product as iproduct
    runs = []
    for bench, seed, k, cap in iproduct(PHASE_D_BENCHES, EXTRA_D_SEEDS, EXTRA_D_K, EXTRA_CAPS):
        runs.append(config.RunSpec(
            benchmark=bench, seed=seed, coupling=COUPLING_CAPPED, k=k,
            diversity_cap=cap,
            search_budget_s=120, run_timeout_s=360,
            criterion="STRONGMUTATION", client_mem_mb=1500, population=5,
        ))
    return runs


def _load_done(csv_path: Path) -> set:
    done: set = set()
    if csv_path.exists():
        for row in _csv.DictReader(open(csv_path)):
            if row.get("status") == "ok":
                done.add(row["config_id"])
    return done


def _run_batch(runs, csv_path: Path, raw_dir: Path, parallelism: int, label: str):
    done = _load_done(csv_path)
    pending = [r for r in runs if r.config_id not in done]
    header = _row_header()
    print(f"[{label}] total={len(runs)}, done={len(done & {r.config_id for r in runs})}, "
          f"pending={len(pending)}")
    if not pending:
        print(f"[{label}] all done, skipping.")
        return
    est = math.ceil(len(pending) * 80 / parallelism / 60)
    print(f"[{label}] Estimated ~{est} min at parallelism={parallelism}")

    done_count = 0

    def _task(spec):
        raw_dir.mkdir(parents=True, exist_ok=True)
        return runner.run_one(spec, config.EVOSUITE_JAR, results_base=raw_dir)

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(_task, spec): spec for spec in pending}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = runner.RunResult(
                    config_id=spec.config_id,
                    target_class=spec.benchmark.target_class,
                    coupling=spec.coupling, k=spec.k, seed=spec.seed,
                    search_budget_s=spec.search_budget_s, status="error",
                    wall_time_s=0.0, error=str(exc),
                )
            _append_row(csv_path, res.merged_row(), header)
            done_count += 1
            flag = f"  <{res.status}>" if res.status != "ok" else ""
            cov = res.stats.get("es_Coverage", res.stats.get("Coverage", "-"))
            print(f"  [{done_count}/{len(pending)}] {res.config_id} cov={cov} {res.wall_time_s:.0f}s{flag}")

    print(f"[{label}] finished -> {csv_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fill CAPPED k-curve gaps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.set_defaults(build=True)
    args = parser.parse_args(argv)

    config.ensure_dirs()

    runs_c = _build_runs_c()
    runs_d = _build_runs_d()
    print(f"Phase C extra: {len(runs_c)} runs  |  Phase D extra: {len(runs_d)} runs")

    if args.dry_run:
        done_c = _load_done(PHASE_C_CSV)
        done_d = _load_done(PHASE_D_CSV)
        pend_c = [r for r in runs_c if r.config_id not in done_c]
        pend_d = [r for r in runs_d if r.config_id not in done_d]
        print(f"Phase C pending: {len(pend_c)}  |  Phase D pending: {len(pend_d)}")
        est_c = math.ceil(len(pend_c) * 70 / 8 / 60)
        est_d = math.ceil(len(pend_d) * 135 / 8 / 60)
        print(f"Estimated: Phase C ~{est_c} min, Phase D ~{est_d} min, total ~{est_c+est_d} min")
        return 0

    from automation import builder
    if args.build:
        res = builder.ensure_built()
        if not res.verified:
            print(f"Build failed: {res.message}"); return 1

    _run_batch(runs_c, PHASE_C_CSV, PHASE_C_CSV.parent / "raw_phc", 8, "Phase-C-extra")
    _run_batch(runs_d, PHASE_D_CSV, PHASE_D_CSV.parent / "raw_phd", 6, "Phase-D-extra")
    return 0


if __name__ == "__main__":
    sys.exit(main())
