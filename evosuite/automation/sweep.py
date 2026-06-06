from __future__ import annotations

import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import builder, config, env_manager, runner
from .config import RunSpec, SweepConfig

_CSV_LOCK = threading.Lock()


def _load_completed(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    done: Set[str] = set()
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add(row["config_id"])
    return done


def _append_row(csv_path: Path, row: Dict[str, object], header: List[str]) -> None:
    with _CSV_LOCK:
        exists = csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in header})


def _row_header() -> List[str]:
    base = [
        "config_id",
        "target_class",
        "coupling",
        "k",
        "seed",
        "search_budget_s",
        "status",
        "wall_time_s",
        "error",
    ]
    es = [f"es_{v}" for v in config.OUTPUT_VARIABLES]
    return base + es


def run_sweep(cfg: SweepConfig, do_build: bool = True, resume: bool = True,
              output_csv: Optional[Path] = None) -> Path:
    config.ensure_dirs()
    csv_path = output_csv or config.AGGREGATED_CSV
    raw_base = csv_path.parent / ("raw_" + csv_path.stem)
    raw_base.mkdir(parents=True, exist_ok=True)

    print(env_manager.summary(cfg))
    env_manager.verify(cfg)

    if do_build:
        print("\n[build] ensuring diversity-aware EvoSuite jar ...")
        result = builder.ensure_built()
        print(f"[build] {result.message}")
        if not result.verified:
            raise SystemExit("Aborting: EvoSuite jar could not be verified.")
    elif not builder.jar_contains_diversity(config.EVOSUITE_JAR):
        raise SystemExit(
            "EvoSuite jar missing/unverified and --no-build was given. Build first."
        )

    runs = config.enumerate_runs(cfg)
    completed = _load_completed(csv_path) if resume else set()
    pending = [r for r in runs if r.config_id not in completed]

    header = _row_header()
    print(
        f"\n[sweep] total={len(runs)} completed={len(completed)} "
        f"pending={len(pending)} parallelism={cfg.parallelism} "
        f"budget={cfg.search_budget_s}s"
    )

    done_count = 0
    failures: List[str] = []

    def _task(spec: RunSpec):
        return runner.run_one(spec, config.EVOSUITE_JAR, results_base=raw_base)

    with ThreadPoolExecutor(max_workers=cfg.parallelism) as pool:
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
                    error=f"runner exception: {exc}",
                )
            _append_row(csv_path, res.merged_row(), header)
            done_count += 1
            cov = res.stats.get("Coverage", "-")
            ent = res.stats.get("MutantTypeEntropyNorm", "-")
            flag = "" if res.status == "ok" else f"  <{res.status}>"
            if res.status != "ok":
                failures.append(f"{res.config_id}: {res.status} ({res.error})")
            print(
                f"[{done_count}/{len(pending)}] {res.config_id} "
                f"cov={cov} vHat={ent} {res.wall_time_s:.0f}s{flag}"
            )

    print(f"\n[sweep] finished. results -> {csv_path}")
    if failures:
        print(f"[sweep] {len(failures)} non-ok runs:")
        for f in failures:
            print("   - " + f)
    return csv_path


def _build_config_from_args(args: argparse.Namespace) -> SweepConfig:
    if args.quick:
        cfg = config.quick_config()
    else:
        cfg = SweepConfig()

    overrides = {}
    if args.budget is not None:
        overrides["search_budget_s"] = args.budget
    if args.parallelism is not None:
        overrides["parallelism"] = args.parallelism
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.k_grid is not None:
        overrides["k_grid"] = tuple(args.k_grid)
    if args.timeout is not None:
        overrides["run_timeout_s"] = args.timeout

    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)
    return cfg


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diversity-aware fitness sweep")
    parser.add_argument("--quick", action="store_true", help="fast smoke sweep")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--budget", type=int, help="search budget seconds/run")
    parser.add_argument("--timeout", type=int, help="hard wall-clock per run (s)")
    parser.add_argument("--parallelism", type=int, help="concurrent EvoSuite runs")
    parser.add_argument("--seeds", type=int, nargs="+", help="random seeds")
    parser.add_argument("--k-grid", dest="k_grid", type=float, nargs="+")
    parser.set_defaults(build=True, resume=True)
    args = parser.parse_args(argv)

    cfg = _build_config_from_args(args)
    run_sweep(cfg, do_build=args.build, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
