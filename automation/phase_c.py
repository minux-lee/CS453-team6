from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from automation import config
from automation.config import (
    COUPLING_ADD, COUPLING_MULT, COUPLING_NONE, DEFAULT_BENCHMARKS,
    PHASE_C_CSV, SweepConfig, enumerate_runs,
)
from automation.sweep import run_sweep

COUPLING_CAPPED = "CAPPED_MULTIPLICATIVE"

DENSE_K_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)

CAPPED_K_GRID = (0.5, 2.0, 8.0)
CAPPED_CAP_VALUES = (0.5, 1.0)

PHASE_C_SEEDS = (42, 1234, 2024, 7, 13, 99, 256, 777)

PHASE_C_BENCHMARKS = {
    "Soundex", "Hex",
    "Fraction", "Precision", "NumberUtils", "IEEE754rUtils",
    "CharUtils", "Strings",
}


def build_phase_c_runs():
    from itertools import product as iproduct
    runs = []

    target_benches = [b for b in DEFAULT_BENCHMARKS if b.short_name in PHASE_C_BENCHMARKS]
    for bench, seed in iproduct(target_benches, PHASE_C_SEEDS):
        common = dict(
            benchmark=bench, seed=seed,
            search_budget_s=60, run_timeout_s=240,
            criterion="STRONGMUTATION", client_mem_mb=1500,
            population=5,
        )
        runs.append(config.RunSpec(coupling=COUPLING_NONE, k=0.0, diversity_cap=1.0, **common))
        for coupling in (COUPLING_MULT, COUPLING_ADD):
            for k in DENSE_K_GRID:
                runs.append(config.RunSpec(coupling=coupling, k=k, diversity_cap=1.0, **common))
        for k in CAPPED_K_GRID:
            for cap in CAPPED_CAP_VALUES:
                runs.append(config.RunSpec(coupling=COUPLING_CAPPED, k=k,
                                           diversity_cap=cap, **common))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase C: dense k + real evolution")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.set_defaults(build=True)
    args = parser.parse_args(argv)

    config.ensure_dirs()

    import csv as _csv
    done: set = set()
    if PHASE_C_CSV.exists():
        for row in _csv.DictReader(open(PHASE_C_CSV)):
            if row.get("status") == "ok":
                done.add(row["config_id"])

    all_runs = build_phase_c_runs()
    pending = [r for r in all_runs if r.config_id not in done]

    import math
    est = math.ceil(len(pending) * 65 / 8 / 60)
    print(f"[phase_c] total={len(all_runs)}, done={len(done)}, pending={len(pending)}")
    print(f"[phase_c] Estimated ~{est} min at parallelism=8, 65s/run")
    print(f"[phase_c] Output: {PHASE_C_CSV}")

    if args.dry_run:
        print("[phase_c] --dry-run: not executing.")
        return 0

    from automation import builder
    if args.build:
        result = builder.ensure_built()
        if not result.verified:
            print(f"[phase_c] Build failed: {result.message}")
            return 1

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from automation import runner
    from automation.sweep import _append_row, _row_header

    header = _row_header()
    done_count = 0
    failures = []

    def _task(spec):
        raw_base = PHASE_C_CSV.parent / "raw_phc"
        raw_base.mkdir(parents=True, exist_ok=True)
        return runner.run_one(spec, config.EVOSUITE_JAR, results_base=raw_base)

    with ThreadPoolExecutor(max_workers=8) as pool:
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
            _append_row(PHASE_C_CSV, res.merged_row(), header)
            done_count += 1
            gen = res.stats.get("es_Generations", res.stats.get("Generations", "-"))
            cov = res.stats.get("es_Coverage", res.stats.get("Coverage", "-"))
            flag = f"  <{res.status}>" if res.status != "ok" else ""
            if res.status != "ok":
                failures.append(f"{res.config_id}: {res.status}")
            print(f"[{done_count}/{len(pending)}] {res.config_id} "
                  f"cov={cov} gen={gen} {res.wall_time_s:.0f}s{flag}")

    print(f"\n[phase_c] finished -> {PHASE_C_CSV}")
    if failures:
        print(f"[phase_c] {len(failures)} non-ok:")
        for f in failures[:10]:
            print("  - " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
