from __future__ import annotations

import argparse
import csv as _csv
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from automation import config, runner
from automation.config import COUPLING_NONE, DEFAULT_BENCHMARKS, PHASE_POP_CSV
from automation.sweep import _append_row, _row_header

POP_BENCH_NAMES = {"Soundex", "Fraction", "NumberUtils", "Strings"}
POP_BENCHES = [b for b in DEFAULT_BENCHMARKS if b.short_name in POP_BENCH_NAMES]

POP_GRID = (5, 10, 50)
SEEDS = (7, 13, 42, 99, 256)
BUDGET_S = 40
TIMEOUT_S = 180


def _build_runs():
    runs = []
    for bench, pop, seed in product(POP_BENCHES, POP_GRID, SEEDS):
        runs.append(
            config.RunSpec(
                benchmark=bench,
                coupling=COUPLING_NONE,
                k=0.0,
                seed=seed,
                search_budget_s=BUDGET_S,
                run_timeout_s=TIMEOUT_S,
                criterion="STRONGMUTATION",
                client_mem_mb=1500,
                population=pop,
            )
        )
    return runs


def _load_done(csv_path: Path) -> set:
    done: set = set()
    if csv_path.exists():
        for row in _csv.DictReader(open(csv_path)):
            if row.get("status") == "ok":
                done.add(row["config_id"])
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Population ablation (40s, NONE)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.set_defaults(build=True)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    runs = _build_runs()
    print(f"Population ablation: {len(runs)} runs "
          f"({len(POP_BENCHES)} benchmarks x pop{POP_GRID} x {len(SEEDS)} seeds)")

    if args.dry_run:
        done = _load_done(PHASE_POP_CSV)
        pending = [r for r in runs if r.config_id not in done]
        print(f"Done: {len(runs)-len(pending)}  Pending: {len(pending)}")
        est = math.ceil(len(pending) * 55 / 8 / 60)
        print(f"Estimated ~{est} min at parallelism=8")
        return 0

    from automation import builder

    if args.build:
        res = builder.ensure_built()
        if not res.verified:
            print(f"Build failed: {res.message}")
            return 1

    done = _load_done(PHASE_POP_CSV)
    pending = [r for r in runs if r.config_id not in done]
    header = _row_header()
    raw_dir = PHASE_POP_CSV.parent / "raw_pop_control"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not pending:
        print("All population-ablation runs already complete.")
        return 0

    print(f"Pending: {len(pending)} runs -> {PHASE_POP_CSV}")

    def _task(spec):
        return runner.run_one(spec, config.EVOSUITE_JAR, results_base=raw_dir)

    done_count = 0
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
                    coupling=spec.coupling,
                    k=spec.k,
                    seed=spec.seed,
                    search_budget_s=spec.search_budget_s,
                    status="error",
                    wall_time_s=0.0,
                    error=str(exc),
                )
            _append_row(PHASE_POP_CSV, res.merged_row(), header)
            done_count += 1
            cov = res.stats.get("es_Coverage", res.stats.get("Coverage", "-"))
            gen = res.stats.get("es_Generations", res.stats.get("Generations", "-"))
            print(f"  [{done_count}/{len(pending)}] {res.config_id} "
                  f"cov={cov} gen={gen} {res.wall_time_s:.0f}s")

    print(f"Finished -> {PHASE_POP_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
