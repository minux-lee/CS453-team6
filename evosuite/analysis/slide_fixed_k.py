
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SLIDE_FIG_DIR = REPO / "analysis" / "report" / "slides"
SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE_C_CSV = REPO / "automation" / "results" / "sweep_results_phc.csv"
PHASE_D_CSV = REPO / "automation" / "results" / "sweep_results_phd.csv"

COLORS = {
    "NONE":                  "#555555",
    "MULTIPLICATIVE":        "#2166ac",
    "ADDITIVE":              "#33a02c",
    "CAPPED_MULTIPLICATIVE": "#ff7f00",
}
MARKERS = {
    "MULTIPLICATIVE":        "o",
    "ADDITIVE":              "s",
    "CAPPED_MULTIPLICATIVE": "D",
}
LABEL = {
    "NONE":                  "NONE (baseline)",
    "MULTIPLICATIVE":        "MULTIPLICATIVE",
    "ADDITIVE":              "ADDITIVE",
    "CAPPED_MULTIPLICATIVE": "CAPPED-MULT",
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})
W, H = 8.0, 4.5

def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    renames = {
        "es_Coverage":              "mutation_score",
        "es_MutantTypeEntropyNorm": "entropy_norm",
        "es_MutantTypeEntropy":     "entropy",
        "es_Generations":           "generations",
        "es_Total_Goals":           "total_goals",
    }
    df = df.rename(columns={v: k for k, v in renames.items() if v in df.columns})
    for col in ["mutation_score", "entropy_norm", "k", "seed",
                "total_goals", "generations", "wall_time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["benchmark"] = df["target_class"].str.rsplit(".", n=1).str[-1]
    df["diversity_cap"] = df["config_id"].str.extract(r"cap([\d.]+)").astype(float).fillna(1.0)
    return df[df["status"] == "ok"].copy()

def _baseline(df: pd.DataFrame) -> pd.Series:
    return df[df["coupling"] == "NONE"].groupby("benchmark")["mutation_score"].median()

def _delta_table(df: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    base = _baseline(df)
    rows = []
    mask = (df["coupling"] != "NONE") & (df["diversity_cap"] == cap)
    for (bench, coup, k), grp in df[mask].groupby(["benchmark", "coupling", "k"]):
        if bench not in base.index:
            continue
        ms_vals = grp["mutation_score"].dropna()
        rows.append({
            "benchmark": bench,
            "coupling":  coup,
            "k":         k,
            "delta_ms":  ms_vals.median() - base[bench],
            "delta_std": ms_vals.std(),
            "n":         len(ms_vals),
        })
    return pd.DataFrame(rows)

def _find_optimal_k(df_delta: pd.DataFrame) -> Dict[str, float]:
    optimal: Dict[str, float] = {}
    for coup, sub in df_delta.groupby("coupling"):
        g = sub.groupby("k").agg(
            med=("delta_ms", "median"),
            pct_pos=("delta_ms", lambda x: (x > 0).mean()),
        )
        g["composite"] = g["med"] * g["pct_pos"]
        optimal[coup] = float(g["composite"].idxmax())
    return optimal

def _save(fig, name: str) -> Path:
    out = SLIDE_FIG_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out

def fig_k_selection(df: pd.DataFrame, optimal_k: Dict[str, float],
                    phase_label: str, fname: str) -> Path:
    d = _delta_table(df, cap=1.0)
    couplings = [c for c in ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
                 if c in d["coupling"].unique()]

    fig, ax = plt.subplots(figsize=(W * 0.85, H))
    for coup in couplings:
        sub = d[d["coupling"] == coup]
        pooled = sub.groupby("k")["delta_ms"].agg(["median", "std"]).sort_index()
        ax.plot(pooled.index, pooled["median"] * 100,
                marker=MARKERS[coup], color=COLORS[coup],
                label=LABEL[coup], lw=2.2, zorder=3)
        ax.fill_between(
            pooled.index,
            (pooled["median"] - pooled["std"].fillna(0)) * 100,
            (pooled["median"] + pooled["std"].fillna(0)) * 100,
            alpha=0.13, color=COLORS[coup],
        )
        k_opt = optimal_k.get(coup)
        if k_opt is not None and k_opt in pooled.index:
            y_opt = pooled.loc[k_opt, "median"] * 100
            ax.axvline(k_opt, color=COLORS[coup], ls=":", lw=1.4, alpha=0.7)
            ax.scatter([k_opt], [y_opt], s=120, color=COLORS[coup],
                       edgecolors="black", lw=1.2, zorder=5)

    ax.axhline(0, color="black", lw=0.9, ls="--", label="NONE baseline (0%)")
    ax.set_xlabel("Diversity pressure  $k$")
    ax.set_ylabel("Pooled median  $\\Delta$ MS  (%)")
    ax.set_title(f"{phase_label}: Pooled $\\Delta$ MS across all benchmarks\n"
                 f"(shading = ±1 std;  dots = selected $k$)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, fname)

def fig_fixed_k_comparison(df: pd.DataFrame, optimal_k: Dict[str, float],
                            phase_label: str, fname: str) -> Path:
    couplings = [c for c in ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
                 if c in df["coupling"].unique()]
    d = _delta_table(df, cap=1.0)
    base = _baseline(df)
    benches = sorted(base.index)

    n_coups = len(couplings)
    x = np.arange(len(benches))
    total_w = 0.72
    w = total_w / n_coups

    fig, ax = plt.subplots(figsize=(W * 1.1, H))

    for ci, coup in enumerate(couplings):
        k_opt = optimal_k.get(coup)
        sub = d[(d["coupling"] == coup) & (np.abs(d["k"] - k_opt) < 0.01)]
        vals   = [sub[sub["benchmark"] == b]["delta_ms"].values[0] * 100
                  if not sub[sub["benchmark"] == b].empty else float("nan")
                  for b in benches]
        stds   = [sub[sub["benchmark"] == b]["delta_std"].values[0] * 100
                  if not sub[sub["benchmark"] == b].empty else float("nan")
                  for b in benches]
        offset = ci * w - total_w / 2 + w / 2
        bars = ax.bar(x + offset, vals, w, label=f"{LABEL[coup]}  (k={k_opt:g})",
                      color=COLORS[coup], edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.errorbar(x + offset, vals, yerr=stds, fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=3)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=25, ha="right")
    ax.set_ylabel("$\\Delta$ Mutation Score  (%,  vs. NONE baseline)")
    ax.set_title(f"{phase_label}: All couplings at globally-optimal $k$  "
                 f"(error bars = ±1 std over seeds)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, fname)

def run_phase(csv_path: Path, phase_label: str,
              k_sel_fname: str, fixed_k_fname: str) -> Optional[Dict[str, float]]:
    df = _load(csv_path)
    if df.empty:
        print(f"  [{phase_label}] No data found at {csv_path}")
        return None
    d = _delta_table(df, cap=1.0)
    if d.empty:
        print(f"  [{phase_label}] No delta rows computed")
        return None

    optimal_k = _find_optimal_k(d)
    print(f"  [{phase_label}] Globally optimal k: {optimal_k}")

    for coup, k_opt in optimal_k.items():
        sub = d[(d["coupling"] == coup) & (np.abs(d["k"] - k_opt) < 0.01)]
        n_benches = sub["benchmark"].nunique()
        print(f"    {coup}: k={k_opt:g},  n_benchmarks={n_benches},  "
              f"pooled_median={sub['delta_ms'].median()*100:+.2f}%")

    fig_k_selection(df, optimal_k, phase_label, k_sel_fname)
    fig_fixed_k_comparison(df, optimal_k, phase_label, fixed_k_fname)
    return optimal_k

def main():
    print("Phase C —")
    opt_c = run_phase(
        PHASE_C_CSV,
        "Phase C  (pop=5, 60s budget)",
        "s11_k_selection_phc.png",
        "s12_fixed_k_phc.png",
    )
    print("\nPhase D —")
    opt_d = run_phase(
        PHASE_D_CSV,
        "Phase D  (pop=5, 120s budget, large classes)",
        "s13_k_selection_phd.png",
        "s14_fixed_k_phd.png",
    )
    print(f"\nAll figures written to: {SLIDE_FIG_DIR}")
    return opt_c, opt_d

if __name__ == "__main__":
    main()
