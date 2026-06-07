
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import parser, metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "analysis" / "report" / "figures"
PHASE_C_CSV = REPO_ROOT / "automation" / "results" / "sweep_results_phc.csv"
PHASE_D_CSV = REPO_ROOT / "automation" / "results" / "sweep_results_phd.csv"

_COUPLING_COLORS = {
    "NONE": "#444444",
    "MULTIPLICATIVE": "#1f77b4",
    "ADDITIVE": "#2ca02c",
    "EXPONENTIAL": "#d62728",
    "CAPPED_MULTIPLICATIVE": "#ff7f0e",
}
_COUPLING_MARKERS = {
    "MULTIPLICATIVE": "o",
    "ADDITIVE": "s",
    "CAPPED_MULTIPLICATIVE": "D",
    "EXPONENTIAL": "^",
}

def _ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_phase_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        for col in ["mutation_score", "entropy", "entropy_norm", "k", "seed",
                    "total_goals", "covered_goals", "suite_size", "wall_time_s",
                    "total_time_ms"]:
            rename_try = {
                "mutation_score": "es_Coverage",
                "entropy": "es_MutantTypeEntropy",
                "entropy_norm": "es_MutantTypeEntropyNorm",
                "total_goals": "es_Total_Goals",
                "covered_goals": "es_Covered_Goals",
                "suite_size": "es_Size",
                "total_time_ms": "es_Total_Time",
            }
        df = df.rename(columns={v: k for k, v in rename_try.items() if v in df.columns})
        for col in ["mutation_score", "entropy_norm", "entropy", "total_goals",
                    "covered_goals", "suite_size", "k", "seed", "wall_time_s",
                    "total_time_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            es_col = f"es_{col.title().replace('_', '')}"
        lib_map, dom_map = parser._benchmark_metadata()
        df["benchmark"] = df["target_class"].str.rsplit(".", n=1).str[-1]
        df["library"] = df["benchmark"].map(lib_map).fillna("unknown")
        df["domain"] = df["benchmark"].map(dom_map).fillna("unknown")
        df["population"] = df["config_id"].str.extract(r"_p(\d+)$")[0].fillna("50")
        df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(50).astype(int)
        df["diversity_cap"] = df["config_id"].str.extract(r"cap([\d.]+)")[0].fillna("1.0")
        df["diversity_cap"] = pd.to_numeric(df["diversity_cap"], errors="coerce").fillna(1.0)
        return df[df["status"] == "ok"].copy()
    except Exception as e:
        print(f"[multi_phase] Warning: could not load {path}: {e}")
        return None

def _phase_c_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["coupling"] == "NONE"].groupby("benchmark")["mutation_score"].median()
    out = []
    for (bench, coupling, k), grp in df[df["coupling"] != "NONE"].groupby(["benchmark", "coupling", "k"]):
        if bench not in base.index:
            continue
        ms = grp["mutation_score"].median()
        ent = grp["entropy_norm"].median()
        b_ms = base[bench]
        out.append({
            "benchmark": bench, "coupling": coupling, "k": k,
            "mutation_score": ms, "entropy_norm": ent,
            "delta_ms": ms - b_ms, "baseline_ms": b_ms,
        })
    return pd.DataFrame(out)

def plot_dense_k_response(df: pd.DataFrame, fname: str = "phc_dense_k.png") -> Path:
    _ensure_fig_dir()
    benches = sorted(df["benchmark"].unique())
    active = [c for c in ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
              if c in df["coupling"].unique()]
    ncol = 4
    nrow = int(np.ceil(len(benches) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False)

    for i, bench in enumerate(benches):
        ax = axes[i // ncol][i % ncol]
        base_val = df[(df["benchmark"] == bench) & (df["coupling"] == "NONE")]["mutation_score"].median()
        if not np.isnan(base_val):
            ax.axhline(base_val, color=_COUPLING_COLORS["NONE"], ls="--", lw=1.5,
                       label="NONE (baseline)")
        for coupling in active:
            sub = df[(df["benchmark"] == bench) & (df["coupling"] == coupling)]
            if sub.empty:
                continue
            if coupling == "CAPPED_MULTIPLICATIVE":
                for cap, csub in sub.groupby("diversity_cap"):
                    g = csub.groupby("k")["mutation_score"].agg(["median", "std"]).sort_index()
                    ax.plot(g.index, g["median"], marker="D",
                            color=_COUPLING_COLORS[coupling], lw=1.5,
                            linestyle="--" if cap < 1.0 else "-",
                            label=f"CAPPED(cap={cap:g})", alpha=0.85)
                    ax.fill_between(g.index,
                                    g["median"] - g["std"].fillna(0),
                                    g["median"] + g["std"].fillna(0),
                                    alpha=0.12, color=_COUPLING_COLORS[coupling])
            else:
                g = sub.groupby("k")["mutation_score"].agg(["median", "std"]).sort_index()
                ax.plot(g.index, g["median"], marker=_COUPLING_MARKERS.get(coupling, "o"),
                        color=_COUPLING_COLORS.get(coupling), lw=2, label=coupling)
                ax.fill_between(g.index,
                                g["median"] - g["std"].fillna(0),
                                g["median"] + g["std"].fillna(0),
                                alpha=0.13, color=_COUPLING_COLORS.get(coupling))

        gens = df[(df["benchmark"] == bench) & (df["coupling"] == "NONE")]["es_Generations"] if "es_Generations" in df.columns else None
        gen_med = pd.to_numeric(gens, errors="coerce").median() if gens is not None else float("nan")
        title = bench + (f" (~{gen_med:.0f}gen)" if not np.isnan(gen_med) else "")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("k")
        ax.set_ylabel("mutation score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    for j in range(len(benches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Phase C: mutation score vs k  (pop=5, real evolution; shading = ±1 std)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

def plot_pop_comparison(df_old: pd.DataFrame, df_new: pd.DataFrame,
                        fname: str = "comparison_pop50_vs_pop5.png") -> Path:
    _ensure_fig_dir()
    compare_configs = [("MULTIPLICATIVE", 0.5), ("MULTIPLICATIVE", 1.0),
                       ("ADDITIVE", 0.25), ("ADDITIVE", 0.5)]

    d_old = _phase_c_deltas(df_old) if df_old is not None else pd.DataFrame()
    d_new = _phase_c_deltas(df_new) if df_new is not None else pd.DataFrame()

    benches = sorted(
        set(d_new["benchmark"].unique() if not d_new.empty else []) |
        set(d_old["benchmark"].unique() if not d_old.empty else [])
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(benches))
    w = 0.18
    slot = 0
    for (coup, k) in compare_configs[:2]:  # just MULT k=0.5, MULT k=1
        for tag, d, color in [("pop=50", d_old, "#aabbdd"), ("pop=5", d_new, "#1f77b4")]:
            if d.empty:
                continue
            sub = d[(d["coupling"] == coup) & (np.abs(d["k"] - k) < 0.01)]
            vals = [sub[sub["benchmark"] == b]["delta_ms"].median()
                    if not sub[sub["benchmark"] == b].empty else float("nan")
                    for b in benches]
            ax.bar(x + slot * w, vals, w, label=f"{coup}(k={k:g}) {tag}", color=color,
                   edgecolor="black", linewidth=0.4,
                   alpha=0.7 if tag == "pop=50" else 1.0)
            slot += 1

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + slot * w / 2 - w / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Δ mutation score vs baseline")
    ax.set_title("Effect of real evolution (pop=5) vs random selection (pop=50)\n"
                 "for MULTIPLICATIVE coupling k=0.5 and k=1.0")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

def plot_phd_large_classes(df: pd.DataFrame, fname: str = "phd_large_classes.png") -> Path:
    _ensure_fig_dir()
    benches = sorted(df["benchmark"].unique())
    active = [c for c in ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
              if c in df["coupling"].unique()]
    ncol = 3
    nrow = int(np.ceil(len(benches) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4 * nrow), squeeze=False)
    for i, bench in enumerate(benches):
        ax = axes[i // ncol][i % ncol]
        base_val = df[(df["benchmark"] == bench) & (df["coupling"] == "NONE")]["mutation_score"].median()
        if not np.isnan(base_val):
            ax.axhline(base_val, color=_COUPLING_COLORS["NONE"], ls="--", lw=1.5, label="NONE")
        for coupling in active:
            sub = df[(df["benchmark"] == bench) & (df["coupling"] == coupling)]
            if coupling == "CAPPED_MULTIPLICATIVE":
                for cap, csub in sub.groupby("diversity_cap"):
                    g = csub.groupby("k")["mutation_score"].agg(["median", "std"]).sort_index()
                    ax.plot(g.index, g["median"], marker="D",
                            color=_COUPLING_COLORS[coupling], lw=1.5,
                            linestyle="--" if cap < 1.0 else "-",
                            label=f"CAPPED(cap={cap:g})")
            else:
                g = sub.groupby("k")["mutation_score"].agg(["median", "std"]).sort_index()
                ax.plot(g.index, g["median"], marker=_COUPLING_MARKERS.get(coupling, "o"),
                        color=_COUPLING_COLORS.get(coupling), lw=2, label=coupling)
                ax.fill_between(g.index, g["median"] - g["std"].fillna(0),
                                g["median"] + g["std"].fillna(0),
                                alpha=0.13, color=_COUPLING_COLORS.get(coupling))
        ax.set_title(bench, fontsize=9)
        ax.set_xlabel("k"); ax.set_ylabel("mutation score")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for j in range(len(benches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Phase D: large classes (pop=5, budget=120s, real evolution)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

def capped_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cap_rows = df[df["coupling"] == "CAPPED_MULTIPLICATIVE"].copy()
    mult_rows = df[df["coupling"] == "MULTIPLICATIVE"].copy()
    if cap_rows.empty or mult_rows.empty:
        return pd.DataFrame()
    base = df[df["coupling"] == "NONE"].groupby("benchmark")["mutation_score"].median()
    rows = []
    for (bench, k, cap), grp in cap_rows.groupby(["benchmark", "k", "diversity_cap"]):
        if bench not in base.index:
            continue
        ms_cap = grp["mutation_score"].median()
        mult_match = mult_rows[(mult_rows["benchmark"] == bench) & (np.abs(mult_rows["k"] - k) < 0.01)]
        ms_mult = mult_match["mutation_score"].median() if not mult_match.empty else float("nan")
        rows.append({
            "benchmark": bench, "k": k, "cap": cap,
            "delta_ms_capped": ms_cap - base[bench],
            "delta_ms_mult": ms_mult - base[bench] if not np.isnan(ms_mult) else float("nan"),
            "advantage_of_cap": (ms_cap - base[bench]) - (ms_mult - base[bench])
                                 if not np.isnan(ms_mult) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["k", "cap", "benchmark"])

def generate_phase_c_report(df_c: pd.DataFrame, df_ab: Optional[pd.DataFrame]) -> Dict[str, Path]:
    figs = {}
    figs["phc_dense_k"] = plot_dense_k_response(df_c)
    if df_ab is not None:
        c_benches = set(df_c["benchmark"].unique())
        df_ab_filtered = df_ab[df_ab["benchmark"].isin(c_benches)]
        figs["comparison"] = plot_pop_comparison(df_ab_filtered, df_c)
    return figs

def generate_phase_d_report(df_d: pd.DataFrame) -> Dict[str, Path]:
    figs = {}
    figs["phd_large"] = plot_phd_large_classes(df_d)
    return figs
