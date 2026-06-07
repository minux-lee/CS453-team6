from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

BASELINE = "NONE"
_AGG_METRICS = ["mutation_score", "entropy", "entropy_norm", "suite_size", "wall_time_s"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg_funcs = {m: ["median", "mean", "std", "count"] for m in _AGG_METRICS if m in df}
    grouped = df.groupby(["benchmark", "coupling", "k"], as_index=False).agg(agg_funcs)
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    return grouped


def baseline_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    base = df[df["coupling"] == BASELINE]
    out: Dict[str, Dict[str, float]] = {}
    for bench, sub in base.groupby("benchmark"):
        out[bench] = {
            "mutation_score": float(sub["mutation_score"].median()),
            "entropy": float(sub["entropy"].median()),
            "entropy_norm": float(sub["entropy_norm"].median()),
            "suite_size": float(sub["suite_size"].median()),
        }
    return out


def deltas_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    base = baseline_lookup(df)
    agg = aggregate(df)
    rows: List[dict] = []
    for _, r in agg.iterrows():
        bench, coupling, k = r["benchmark"], r["coupling"], r["k"]
        if coupling == BASELINE or bench not in base:
            continue
        b = base[bench]
        rows.append(
            {
                "benchmark": bench,
                "coupling": coupling,
                "k": k,
                "mutation_score": r["mutation_score_median"],
                "baseline_mutation_score": b["mutation_score"],
                "delta_mutation_score": r["mutation_score_median"] - b["mutation_score"],
                "entropy_norm": r["entropy_norm_median"],
                "baseline_entropy_norm": b["entropy_norm"],
                "delta_entropy_norm": r["entropy_norm_median"] - b["entropy_norm"],
                "entropy": r["entropy_median"],
                "delta_entropy": r["entropy_median"] - b["entropy"],
            }
        )
    return pd.DataFrame(rows)


def coupling_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    return (
        d.groupby("coupling", as_index=False)
        .agg(
            mean_delta_mutation_score=("delta_mutation_score", "mean"),
            mean_delta_entropy_norm=("delta_entropy_norm", "mean"),
            mean_delta_entropy=("delta_entropy", "mean"),
            n=("delta_mutation_score", "count"),
        )
        .sort_values("mean_delta_entropy_norm", ascending=False)
    )


def best_configurations(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    safe = d[d["delta_mutation_score"] >= -1e-9]
    pool = safe if not safe.empty else d
    idx = pool.groupby("benchmark")["delta_entropy_norm"].idxmax()
    return pool.loc[idx].sort_values("benchmark")


def domain_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    meta = df[["benchmark", "domain", "library"]].drop_duplicates()
    d = d.merge(meta, on="benchmark", how="left")
    return (
        d.groupby(["domain", "coupling"], as_index=False)
        .agg(
            mean_delta_mutation_score=("delta_mutation_score", "mean"),
            mean_delta_entropy_norm=("delta_entropy_norm", "mean"),
            n=("delta_mutation_score", "count"),
        )
        .sort_values(["domain", "coupling"])
    )


def library_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    meta = df[["benchmark", "domain", "library"]].drop_duplicates()
    d = d.merge(meta, on="benchmark", how="left")
    return (
        d.groupby(["library", "coupling"], as_index=False)
        .agg(
            mean_delta_mutation_score=("delta_mutation_score", "mean"),
            mean_delta_entropy_norm=("delta_entropy_norm", "mean"),
            n=("delta_mutation_score", "count"),
        )
        .sort_values(["library", "coupling"])
    )


def characteristics_table(df: pd.DataFrame) -> pd.DataFrame:
    base = baseline_lookup(df)
    sizes = df.groupby("benchmark")["total_goals"].median()
    meta = df[["benchmark", "domain", "library"]].drop_duplicates().set_index("benchmark")
    d = deltas_vs_baseline(df)
    low = d[(d["k"] <= 2.0) & (d["coupling"].isin(["MULTIPLICATIVE", "ADDITIVE"]))]

    rows: List[dict] = []
    for bench, b in base.items():
        sub = low[low["benchmark"] == bench]
        if not sub.empty:
            best = sub.sort_values(
                ["delta_mutation_score", "delta_entropy_norm"], ascending=False
            ).iloc[0]
            best_ms = best["delta_mutation_score"]
            best_ent = best["delta_entropy_norm"]
            best_cfg = f"{best['coupling']}(k={best['k']:g})"
        else:
            best_ms = best_ent = float("nan")
            best_cfg = "-"
        rows.append(
            {
                "benchmark": bench,
                "library": meta.loc[bench, "library"] if bench in meta.index else "?",
                "domain": meta.loc[bench, "domain"] if bench in meta.index else "?",
                "total_goals": float(sizes.get(bench, float("nan"))),
                "baseline_mutation_score": b["mutation_score"],
                "baseline_entropy_norm": b["entropy_norm"],
                "best_lowk_config": best_cfg,
                "best_lowk_delta_ms": best_ms,
                "best_lowk_delta_entropy_norm": best_ent,
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "benchmark"])


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    ch = characteristics_table(df)
    out: List[dict] = []
    if ch.empty or ch["best_lowk_delta_ms"].notna().sum() < 3:
        return pd.DataFrame()
    valid = ch.dropna(subset=["best_lowk_delta_ms"])
    for feat in ["baseline_mutation_score", "baseline_entropy_norm", "total_goals"]:
        try:
            corr = valid[feat].corr(valid["best_lowk_delta_ms"], method="spearman")
        except Exception:
            corr = float("nan")
        out.append({"characteristic": feat, "spearman_rho_vs_best_lowk_delta_ms": corr})
    return pd.DataFrame(out)


def significance(df: pd.DataFrame, metric: str = "mutation_score") -> pd.DataFrame:
    if not _HAVE_SCIPY:
        return pd.DataFrame()
    rows: List[dict] = []
    for bench, sub in df.groupby("benchmark"):
        base_vals = sub[sub["coupling"] == BASELINE][metric].dropna().values
        if len(base_vals) < 2:
            continue
        for (coupling, k), grp in sub.groupby(["coupling", "k"]):
            if coupling == BASELINE:
                continue
            vals = grp[metric].dropna().values
            if len(vals) < 2:
                continue
            try:
                stat, p = mannwhitneyu(vals, base_vals, alternative="two-sided")
            except ValueError:
                continue
            rows.append(
                {
                    "benchmark": bench,
                    "coupling": coupling,
                    "k": k,
                    "metric": metric,
                    "median_config": float(np.median(vals)),
                    "median_baseline": float(np.median(base_vals)),
                    "u_stat": float(stat),
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(rows)


def best_per_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    idx = d.groupby("benchmark")["mutation_score"].idxmax()
    return d.loc[idx].sort_values("benchmark")


def joint_winners(df: pd.DataFrame) -> pd.DataFrame:
    d = deltas_vs_baseline(df)
    if d.empty:
        return d
    win = d[(d["delta_mutation_score"] > 1e-9) & (d["delta_entropy_norm"] > 1e-9)]
    return win.sort_values(["benchmark", "delta_mutation_score"], ascending=[True, False])


def honest_summary(df: pd.DataFrame) -> dict:
    best = best_per_benchmark(df)
    joint = joint_winners(df)
    n_bench = len(best)
    ms_up = int((best["delta_mutation_score"] > 1e-9).sum()) if not best.empty else 0
    joint_at_best = 0
    if not best.empty:
        joint_at_best = int(
            ((best["delta_mutation_score"] > 1e-9) & (best["delta_entropy_norm"] > 1e-9)).sum()
        )
    any_joint_bench = joint["benchmark"].nunique() if not joint.empty else 0
    sig_n = 0
    if _HAVE_SCIPY and not best.empty:
        for _, row in best.iterrows():
            bench, coup, k = row["benchmark"], row["coupling"], row["k"]
            base_vals = df[(df["benchmark"] == bench) & (df["coupling"] == BASELINE)][
                "mutation_score"
            ].dropna().values
            cfg_vals = df[
                (df["benchmark"] == bench)
                & (df["coupling"] == coup)
                & (df["k"] == k)
            ]["mutation_score"].dropna().values
            if len(base_vals) >= 4 and len(cfg_vals) >= 4:
                try:
                    _, p = mannwhitneyu(base_vals, cfg_vals, alternative="two-sided")
                    if p < 0.05:
                        sig_n += 1
                except ValueError:
                    pass
    d = deltas_vs_baseline(df)
    pooled_mean = float(d["delta_mutation_score"].mean()) if not d.empty else float("nan")
    return {
        "n_benchmarks": n_bench,
        "ms_improved_best_config": ms_up,
        "joint_at_best_ms_config": joint_at_best,
        "any_joint_win_benchmarks": any_joint_bench,
        "significant_best_config": sig_n,
        "pooled_mean_delta_ms": pooled_mean,
    }


def domain_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    best = best_per_benchmark(df)
    if best.empty:
        return best
    meta = df[["benchmark", "domain"]].drop_duplicates()
    best = best.merge(meta, on="benchmark", how="left")
    return (
        best.groupby(["domain", "coupling"], as_index=False)
        .agg(
            mean_best_delta_ms=("delta_mutation_score", "mean"),
            mean_best_delta_entropy=("delta_entropy_norm", "mean"),
            n=("benchmark", "count"),
        )
        .sort_values(["domain", "coupling"])
    )
