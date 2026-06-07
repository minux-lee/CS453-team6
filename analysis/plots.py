from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics, parser  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "analysis" / "report" / "figures"

_COUPLING_COLORS = {
    "NONE": "#444444",
    "MULTIPLICATIVE": "#1f77b4",
    "ADDITIVE": "#2ca02c",
    "EXPONENTIAL": "#d62728",
}
_COUPLING_MARKERS = {
    "MULTIPLICATIVE": "o",
    "ADDITIVE": "s",
    "EXPONENTIAL": "^",
}


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _median_curve(df: pd.DataFrame, bench: str, coupling: str, metric: str):
    sub = df[(df["benchmark"] == bench) & (df["coupling"] == coupling)]
    g = sub.groupby("k")[metric].median().sort_index()
    return g.index.values, g.values


def plot_metric_vs_k(df: pd.DataFrame, metric: str, ylabel: str, fname: str) -> Path:
    _ensure_dir()
    benches = parser.benchmarks(df)
    active = [c for c in parser.couplings(df) if c != "NONE"]
    ncol = 3 if len(benches) > 6 else 2
    nrow = int(np.ceil(len(benches) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)

    for i, bench in enumerate(benches):
        ax = axes[i // ncol][i % ncol]
        base = df[(df["benchmark"] == bench) & (df["coupling"] == "NONE")][metric].median()
        if not np.isnan(base):
            ax.axhline(base, color=_COUPLING_COLORS["NONE"], ls="--", lw=1.5,
                       label="NONE (baseline)")
        for coupling in active:
            ks, vals = _median_curve(df, bench, coupling, metric)
            if len(ks) == 0:
                continue
            ax.plot(ks, vals, marker=_COUPLING_MARKERS.get(coupling, "o"),
                    color=_COUPLING_COLORS.get(coupling), label=coupling, lw=2)
        ax.set_title(bench)
        ax.set_xlabel("diversity scale  k")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(benches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"{ylabel} vs diversity scale (median over seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_tradeoff(df: pd.DataFrame, fname: str = "tradeoff_scatter.png") -> Path:
    _ensure_dir()
    agg = df.groupby(["benchmark", "coupling", "k"], as_index=False).agg(
        mutation_score=("mutation_score", "median"),
        entropy_norm=("entropy_norm", "median"),
    )
    benches = parser.benchmarks(df)
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for bi, bench in enumerate(benches):
        sub = agg[agg["benchmark"] == bench]
        for _, r in sub.iterrows():
            is_base = r["coupling"] == "NONE"
            ax.scatter(
                r["entropy_norm"], r["mutation_score"],
                marker=markers[bi % len(markers)],
                s=160 if is_base else 70,
                color=_COUPLING_COLORS.get(r["coupling"], "#888"),
                edgecolors="black" if is_base else "none",
                linewidths=1.4 if is_base else 0,
                alpha=0.9,
            )
    bench_handles = [
        plt.Line2D([0], [0], marker=markers[i % len(markers)], color="gray",
                   linestyle="", label=b) for i, b in enumerate(benches)
    ]
    coup_handles = [
        plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=k)
        for k, c in _COUPLING_COLORS.items()
    ]
    leg1 = ax.legend(handles=bench_handles, title="benchmark (shape)",
                     loc="lower left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=coup_handles, title="coupling (color); large=baseline",
              loc="lower right", fontsize=8)
    ax.set_xlabel("normalised killed-mutant-type entropy  v\u0302")
    ax.set_ylabel("mutation score")
    ax.set_title("Diversity vs mutation-score trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_coupling_summary(df: pd.DataFrame, fname: str = "coupling_summary.png") -> Path:
    _ensure_dir()
    summ = metrics.coupling_summary(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    if summ.empty:
        ax.text(0.5, 0.5, "no data", ha="center")
        out = FIG_DIR / fname
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
    x = np.arange(len(summ))
    w = 0.38
    ax.bar(x - w / 2, summ["mean_delta_entropy_norm"], w,
           label="\u0394 normalised entropy", color="#1f77b4")
    ax.bar(x + w / 2, summ["mean_delta_mutation_score"], w,
           label="\u0394 mutation score", color="#ff7f0e")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summ["coupling"], rotation=10)
    ax.set_ylabel("mean change vs baseline (pooled)")
    ax.set_title("Effect of coupling strategy (averaged over benchmarks & k)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


_DOMAIN_COLORS = {
    "numeric": "#1f77b4",
    "string": "#2ca02c",
    "boolean": "#9467bd",
    "encoding": "#d62728",
    "unknown": "#888888",
}


def plot_domain_summary(df: pd.DataFrame, fname: str = "domain_summary.png") -> Path:
    _ensure_dir()
    ds = metrics.domain_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=True)
    if ds.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center")
        out = FIG_DIR / fname
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out

    domains = sorted(ds["domain"].unique())
    couplings = [c for c in ["MULTIPLICATIVE", "ADDITIVE", "EXPONENTIAL"]
                 if c in set(ds["coupling"])]
    x = np.arange(len(domains))
    w = 0.8 / max(len(couplings), 1)
    for metric, ax, title in (
        ("mean_delta_entropy_norm", axes[0], "\u0394 normalised entropy"),
        ("mean_delta_mutation_score", axes[1], "\u0394 mutation score"),
    ):
        for ci, coup in enumerate(couplings):
            vals = [
                ds[(ds["domain"] == d) & (ds["coupling"] == coup)][metric].mean()
                for d in domains
            ]
            ax.bar(x + ci * w - 0.4 + w / 2, vals, w,
                   label=coup, color=_COUPLING_COLORS.get(coup))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=10)
        ax.set_title(f"{title} by domain (pooled)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Coupling effect across functional domains", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_characteristics(df: pd.DataFrame, fname: str = "characteristics.png") -> Path:
    _ensure_dir()
    ch = metrics.characteristics_table(df).dropna(subset=["best_lowk_delta_ms"])
    feats = [
        ("baseline_entropy_norm", "baseline normalised entropy v\u0302"),
        ("baseline_mutation_score", "baseline mutation score"),
        ("total_goals", "class size (total goals)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=True)
    for ax, (feat, label) in zip(axes, feats):
        for _, r in ch.iterrows():
            ax.scatter(r[feat], r["best_lowk_delta_ms"], s=90,
                       color=_DOMAIN_COLORS.get(r["domain"], "#888"),
                       edgecolors="black", linewidths=0.5)
            ax.annotate(r["benchmark"], (r[feat], r["best_lowk_delta_ms"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(label)
        ax.set_ylabel("best low-k \u0394 mutation score")
        ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=d)
               for d, c in _DOMAIN_COLORS.items() if d != "unknown"]
    axes[-1].legend(handles=handles, title="domain", fontsize=8)
    fig.suptitle("Where diversity coupling helps: effect vs class characteristics",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def generate_all(df: pd.DataFrame) -> Dict[str, Path]:
    return {
        "mutation_vs_k": plot_metric_vs_k(
            df, "mutation_score", "mutation score", "mutation_score_vs_k.png"
        ),
        "entropy_vs_k": plot_metric_vs_k(
            df, "entropy_norm", "normalised entropy v\u0302", "entropy_vs_k.png"
        ),
        "tradeoff": plot_tradeoff(df),
        "coupling_summary": plot_coupling_summary(df),
        "domain_summary": plot_domain_summary(df),
        "characteristics": plot_characteristics(df),
    }
