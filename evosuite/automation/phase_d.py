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
from automation.config import (
    COUPLING_ADD, COUPLING_MULT, COUPLING_NONE, PHASE_D_CSV,
    DEFAULT_BENCHMARKS,
)
from automation.sweep import _append_row, _row_header

COUPLING_CAPPED = "CAPPED_MULTIPLICATIVE"

LARGE_CLASS_NAMES = {
    "ArithmeticUtils", "Complex", "BooleanUtils", "WordUtils", "Fraction",
    "IntMath",
}

LARGE_BENCHMARKS = [b for b in DEFAULT_BENCHMARKS if b.short_name in LARGE_CLASS_NAMES]

PHASE_D_K_GRID = (0.25, 0.5, 0.75, 1.0)
PHASE_D_SEEDS = (42, 1234, 2024, 7, 13, 99)
PHASE_D_CAPS = (0.5, 1.0)


def build_phase_d_runs():
    from itertools import product as iproduct
    runs = []
    for bench, seed in iproduct(LARGE_BENCHMARKS, PHASE_D_SEEDS):
        common = dict(
            benchmark=bench, seed=seed,
            search_budget_s=120, run_timeout_s=360,
            criterion="STRONGMUTATION", client_mem_mb=1500,
            population=5,
        )
        runs.append(config.RunSpec(coupling=COUPLING_NONE, k=0.0, diversity_cap=1.0, **common))
        for coupling in (COUPLING_MULT, COUPLING_ADD):
            for k in PHASE_D_K_GRID:
                runs.append(config.RunSpec(coupling=coupling, k=k, diversity_cap=1.0, **common))
        for k in (0.5, 1.0, 4.0):
            for cap in PHASE_D_CAPS:
                runs.append(config.RunSpec(coupling=COUPLING_CAPPED, k=k,
                                           diversity_cap=cap, **common))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase D: extended budget for large classes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.set_defaults(build=True)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    done: set = set()
    if PHASE_D_CSV.exists():
        for row in _csv.DictReader(open(PHASE_D_CSV)):
            if row.get("status") == "ok":
                done.add(row["config_id"])

    all_runs = build_phase_d_runs()
    pending = [r for r in all_runs if r.config_id not in done]

    est = math.ceil(len(pending) * 130 / 8 / 60)
    print(f"[phase_d] large classes: {[b.short_name for b in LARGE_BENCHMARKS]}")
    print(f"[phase_d] total={len(all_runs)}, done={len(done)}, pending={len(pending)}")
    print(f"[phase_d] Estimated ~{est} min at parallelism=8, 130s/run")
    print(f"[phase_d] Output: {PHASE_D_CSV}")

    if args.dry_run:
        print("[phase_d] --dry-run: not executing.")
        return 0

    from automation import builder
    if args.build:
        result = builder.ensure_built()
        if not result.verified:
            print(f"[phase_d] Build failed: {result.message}")
            return 1

    header = _row_header()
    done_count = 0
    failures = []

    def _task(spec):
        raw_base = PHASE_D_CSV.parent / "raw_phd"
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
            _append_row(PHASE_D_CSV, res.merged_row(), header)
            done_count += 1
            gen = res.stats.get("es_Generations", res.stats.get("Generations", "-"))
            cov = res.stats.get("es_Coverage", res.stats.get("Coverage", "-"))
            flag = f"  <{res.status}>" if res.status != "ok" else ""
            if res.status != "ok":
                failures.append(f"{res.config_id}: {res.status}")
            print(f"[{done_count}/{len(pending)}] {res.config_id} "
                  f"cov={cov} gen={gen} {res.wall_time_s:.0f}s{flag}")

    print(f"\n[phase_d] finished -> {PHASE_D_CSV}")
    if failures:
        print(f"[phase_d] {len(failures)} non-ok:")
        for f in failures[:10]:
            print("  - " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
