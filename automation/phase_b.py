from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis import metrics, parser
from automation import config, env_manager
from automation.sweep import run_sweep

PHASE_B_SEEDS = (42, 1234, 2024, 7, 13, 99, 256, 777, 31337, 2718)
SAFE_TIER = [("ADDITIVE", 0.25), ("ADDITIVE", 0.5), ("MULTIPLICATIVE", 0.5), ("MULTIPLICATIVE", 1.0)]


def select_promising(
    df,
    min_ms_delta: float = -0.02,
    min_ent_delta: float = 0.0,
):
    d = metrics.deltas_vs_baseline(df)
    if d.empty:
        return set()

    promising = set(SAFE_TIER)

    agg = (
        d.groupby(["coupling", "k"])
        .agg(
            mean_delta_ms=("delta_mutation_score", "mean"),
            pct_positive_ms=(
                "delta_mutation_score",
                lambda x: (x >= min_ms_delta).mean(),
            ),
            mean_delta_ent=("delta_entropy_norm", "mean"),
        )
        .reset_index()
    )

    agg["composite"] = 2 * agg["mean_delta_ent"] + agg["mean_delta_ms"]
    top = agg.sort_values("composite", ascending=False).head(4)

    for _, row in top.iterrows():
        if row["mean_delta_ms"] >= min_ms_delta:
            promising.add((str(row["coupling"]), float(row["k"])))

    return promising


def build_phase_b_config(
    promising,
    seeds: tuple,
    budget: int = 40,
    parallelism: int = 8,
) -> config.SweepConfig:
    return config.SweepConfig(
        benchmarks=config.DEFAULT_BENCHMARKS,
        couplings=tuple(sorted({c for c, _ in promising})),
        k_grid=tuple(sorted({k for _, k in promising})),
        seeds=seeds,
        search_budget_s=budget,
        run_timeout_s=240,
        parallelism=parallelism,
    )


def main(argv=None):
    parser_cli = argparse.ArgumentParser(description="Phase B confirmatory sweep")
    parser_cli.add_argument("--dry-run", action="store_true")
    parser_cli.add_argument("--seeds", type=int, nargs="+", default=list(PHASE_B_SEEDS))
    parser_cli.add_argument("--min-ms-delta", type=float, default=-0.02)
    parser_cli.add_argument("--min-ent-delta", type=float, default=0.0)
    parser_cli.add_argument("--budget", type=int, default=40)
    parser_cli.add_argument("--parallelism", type=int, default=8)
    parser_cli.add_argument("--no-build", dest="build", action="store_false")
    parser_cli.set_defaults(build=True)
    args = parser_cli.parse_args(argv)

    try:
        df = parser.load_results()
    except (FileNotFoundError, ValueError) as e:
        print(f"[phase_b] Cannot load Phase A results: {e}")
        return 1

    n_seeds_a = df["seed"].nunique()
    print(f"[phase_b] Phase A data: {len(df)} rows, {n_seeds_a} seeds, "
          f"{df['benchmark'].nunique()} benchmarks")

    promising = select_promising(df, args.min_ms_delta, args.min_ent_delta)
    print(f"[phase_b] Promising (coupling, k) pairs: {sorted(promising)}")

    seeds_b = tuple(args.seeds)
    cfg = build_phase_b_config(promising, seeds_b, args.budget, args.parallelism)

    all_runs = config.enumerate_runs(cfg)
    import csv as _csv
    done: set = set()
    if config.AGGREGATED_CSV.exists():
        for row in _csv.DictReader(open(config.AGGREGATED_CSV)):
            if row.get("status") == "ok":
                done.add(row["config_id"])
    pending = [r for r in all_runs if r.config_id not in done]

    print(f"[phase_b] Phase B total={len(all_runs)}, already_done={len(done & {r.config_id for r in all_runs})}, "
          f"pending={len(pending)}")
    import math
    est_min = math.ceil(len(pending) * 60 / 8 / 60)
    print(f"[phase_b] Estimated time: ~{est_min} min at 8-way parallelism, 60s/run")

    if args.dry_run:
        print("[phase_b] --dry-run: not executing.")
        return 0

    run_sweep(cfg, do_build=args.build, resume=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
