
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SLIDE_FIG_DIR = REPO / "analysis" / "report" / "slides"
SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE_AB_CSV  = REPO / "automation" / "results" / "sweep_results.csv"
PHASE_C_CSV   = REPO / "automation" / "results" / "sweep_results_phc.csv"
PHASE_D_CSV   = REPO / "automation" / "results" / "sweep_results_phd.csv"
PHASE_POP_CSV = REPO / "automation" / "results" / "sweep_results_pop_control.csv"

COLORS = {
    "NONE":                 "#555555",
    "MULTIPLICATIVE":       "#2166ac",
    "ADDITIVE":             "#33a02c",
    "EXPONENTIAL":          "#e31a1c",
    "CAPPED_MULTIPLICATIVE":"#ff7f00",
}
MARKERS = {"MULTIPLICATIVE":"o","ADDITIVE":"s","EXPONENTIAL":"^",
           "CAPPED_MULTIPLICATIVE":"D","NONE":"X"}
KOREAN = {
    "NONE":                 "NONE (baseline)",
    "MULTIPLICATIVE":       "MULTIPLICATIVE",
    "ADDITIVE":             "ADDITIVE",
    "EXPONENTIAL":          "EXPONENTIAL",
    "CAPPED_MULTIPLICATIVE":"CAPPED-MULT (new)",
}
DOMAIN_KO = {"numeric":"Numeric","string":"String","boolean":"Boolean","encoding":"Encoding"}
DOMAIN_COLORS = {"numeric":"#2166ac","string":"#33a02c",
                 "boolean":"#9467bd","encoding":"#d62728","unknown":"#aaaaaa"}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

W, H = 8.0, 4.5  # slide-safe inches

def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rename = {
        "es_Coverage":             "mutation_score",
        "es_MutantTypeEntropy":    "entropy",
        "es_MutantTypeEntropyNorm":"entropy_norm",
        "es_Total_Goals":          "total_goals",
        "es_Covered_Goals":        "covered_goals",
        "es_Generations":          "generations",
    }
    df = df.rename(columns={v:k for k,v in rename.items() if v in df.columns})
    es_renames = {c: c.replace("es_", "").lower() for c in df.columns if c.startswith("es_")}
    for es_col, tidy_col in [("es_Coverage","mutation_score"),("es_MutantTypeEntropyNorm","entropy_norm"),
                              ("es_MutantTypeEntropy","entropy"),("es_Generations","generations"),
                              ("es_Total_Goals","total_goals"),("es_Covered_Goals","covered_goals")]:
        if es_col in df.columns and tidy_col not in df.columns:
            df[tidy_col] = pd.to_numeric(df[es_col], errors="coerce")
    for c in ["mutation_score","entropy_norm","entropy","k","seed",
              "total_goals","covered_goals","generations","wall_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["benchmark"] = df["target_class"].str.rsplit(".", n=1).str[-1]
    df["population"] = df["config_id"].str.extract(r"_p(\d+)").astype(float).fillna(50)
    df["diversity_cap"] = df["config_id"].str.extract(r"cap([\d.]+)").astype(float).fillna(1.0)

    try:
        from automation import config as _cfg
        lib_m = {b.short_name: b.library for b in _cfg.DEFAULT_BENCHMARKS}
        dom_m = {b.short_name: b.domain  for b in _cfg.DEFAULT_BENCHMARKS}
        df["library"] = df["benchmark"].map(lib_m).fillna("unknown")
        df["domain"]  = df["benchmark"].map(dom_m).fillna("unknown")
    except Exception:
        df["library"] = "unknown"
        df["domain"]  = "unknown"

    return df[df["status"] == "ok"].copy()

def _deltas(df: pd.DataFrame) -> pd.DataFrame:
    ms_col = "mutation_score" if "mutation_score" in df.columns else "es_Coverage"
    ent_col = "entropy_norm" if "entropy_norm" in df.columns else "es_MutantTypeEntropyNorm"
    df = df.copy()
    if ms_col != "mutation_score":
        df["mutation_score"] = pd.to_numeric(df[ms_col], errors="coerce")
    if ent_col != "entropy_norm":
        df["entropy_norm"] = pd.to_numeric(df[ent_col], errors="coerce")
    base = df[df["coupling"] == "NONE"].groupby("benchmark")["mutation_score"].median()
    rows = []
    for (b, c, k), g in df[df["coupling"] != "NONE"].groupby(["benchmark","coupling","k"]):
        if b not in base.index:
            continue
        ms  = g["mutation_score"].median()
        ent = g["entropy_norm"].median()
        cap = g["diversity_cap"].mode()[0] if "diversity_cap" in g.columns else 1.0
        rows.append({"benchmark":b,"coupling":c,"k":k,"cap":cap,
                     "mutation_score":ms,"entropy_norm":ent,
                     "delta_ms":ms-base[b],"baseline_ms":base[b]})
    return pd.DataFrame(rows)

def _save(fig, name: str) -> Path:
    out = SLIDE_FIG_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out

def fig_formula_concept():
    fig, ax = plt.subplots(figsize=(W, H))
    delta = np.linspace(0, 1, 300)
    k, F_base, cap = 2.0, 1.0, 1.0

    ax.plot(delta, F_base * (1 + k * delta),
            color=COLORS["MULTIPLICATIVE"], label=f"MULTIPLICATIVE  $F_b(1+k\\delta)$", lw=2.5)
    ax.plot(delta, F_base + k * delta,
            color=COLORS["ADDITIVE"], label=f"ADDITIVE  $F_b+k\\delta$", lw=2.5, ls="--")
    ax.plot(delta, F_base * (1 + np.minimum(k * delta, cap)),
            color=COLORS["CAPPED_MULTIPLICATIVE"],
            label=f"CAPPED-MULT  $F_b(1+\\min(k\\delta, c))$", lw=2.5, ls="-.")
    ax.axvline(0.5, color="gray", ls=":", lw=1.2)
    ax.axhline(F_base, color=COLORS["NONE"], ls=":", lw=1.5,
               label="$F_{base}$ (baseline, perfect diversity)")

    ax.set_xlabel("Diversity deficiency  $\\delta = 1 - \\hat{v}$   (0=fully diverse, 1=single type)")
    ax.set_ylabel("Fitness value  $F$  (EvoSuite minimises)")
    ax.set_title(f"Coupling Mode Comparison  ($k={k:g},\\ c={cap:g}$)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0.8, 3.3)
    return _save(fig, "s01_formula_concept.png")

def fig_pop_ablation():
    df = _load(PHASE_POP_CSV)
    if df.empty:
        return _fig_ga_stall_fallback()

    benches = sorted(df["benchmark"].unique())
    pops = [5, 10, 50]
    pop_colors = {5: "#2166ac", 10: "#33a02c", 50: "#aabbcc"}

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.15, H), sharex=True)
    metrics_plot = [
        ("generations", "Median Generations", axes[0]),
        ("mutation_score", "Median Mutation Score", axes[1]),
    ]
    x = np.arange(len(benches))
    w = 0.25
    offsets = [-w, 0, w]

    for pop, off in zip(pops, offsets):
        vals_gen, vals_ms = [], []
        for b in benches:
            sub = df[(df["benchmark"] == b) & (df["population"] == pop)]
            vals_gen.append(sub["generations"].median() if not sub.empty else 0)
            vals_ms.append(sub["mutation_score"].median() if not sub.empty else 0)
        axes[0].bar(x + off, vals_gen, w, color=pop_colors[pop], edgecolor="black",
                    lw=0.5, label=f"pop={pop}")
        axes[1].bar(x + off, vals_ms, w, color=pop_colors[pop], edgecolor="black",
                    lw=0.5, label=f"pop={pop}")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(benches, rotation=20, ha="right")
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)

    axes[0].set_title("Generations (40s, NONE, controlled)")
    axes[1].set_title("Mutation Score (40s, NONE, controlled)")
    fig.suptitle("Population Ablation at Fixed 40s Budget  (only pop varies)",
                 fontsize=14)
    fig.tight_layout()
    return _save(fig, "s02_ga_stall.png")

def _fig_ga_stall_fallback():
    df_ab = _load(PHASE_AB_CSV)
    df_c = _load(PHASE_C_CSV)
    benches = sorted(set(df_c["benchmark"].unique()) & set(df_ab["benchmark"].unique()))

    def med(df, col, benches):
        if df.empty or col not in df.columns:
            return [0] * len(benches)
        return [df[df["benchmark"] == b][col].median() for b in benches]

    gen50 = med(df_ab, "generations", benches)
    gen5 = med(df_c, "generations", benches)
    fig, ax = plt.subplots(figsize=(W, H))
    x = np.arange(len(benches))
    w = 0.38
    ax.bar(x - w / 2, gen50, w, color="#aabbcc", label="pop=50 (40s) [Phase A/B]")
    ax.bar(x + w / 2, gen5, w, color="#2166ac", label="pop=5 (60s) [Phase C]")
    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=25, ha="right")
    ax.set_ylabel("Median Generations")
    ax.set_title("GA Stall (legacy; budget confounded — see pop ablation)")
    ax.legend()
    return _save(fig, "s02_ga_stall.png")

def fig_phc_best_delta(df_c: pd.DataFrame):
    d = _deltas(df_c)
    best = d.loc[d.groupby("benchmark")["delta_ms"].idxmax()].copy()
    best = best.sort_values("delta_ms")

    fig, ax = plt.subplots(figsize=(W, H))
    colors = [COLORS.get(c, "#888") for c in best["coupling"]]
    bars = ax.barh(best["benchmark"], best["delta_ms"] * 100,
                   color=colors, edgecolor="black", linewidth=0.5)

    for bar, (_, row) in zip(bars, best.iterrows()):
        v = row["delta_ms"] * 100
        ha = "left" if v >= 0 else "right"
        offset = 0.15 if v >= 0 else -0.15
        ax.text(v + offset, bar.get_y() + bar.get_height()/2,
                f"{v:+.1f}%  ({KOREAN.get(row['coupling'],'')}, k={row['k']:g})",
                va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Delta Mutation Score (%, vs. NONE baseline)")
    ax.set_title("Phase C: Best Improvement per Benchmark  (pop=5, real evolution)")

    legend_handles = [mpatches.Patch(color=COLORS[c], label=KOREAN[c])
                      for c in ["MULTIPLICATIVE","ADDITIVE","CAPPED_MULTIPLICATIVE"]]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11)
    ax.set_xlim(min(best["delta_ms"].min() * 100 - 2, -8), max(best["delta_ms"].max() * 100 + 4, 12))
    return _save(fig, "s03_phc_best_delta.png")

def fig_phc_representative(df_c: pd.DataFrame):
    focus = ["Fraction", "Precision", "NumberUtils"]
    couplings = ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
    linestyles = {"MULTIPLICATIVE":"-","ADDITIVE":"--","CAPPED_MULTIPLICATIVE":"-."}

    fig, axes = plt.subplots(1, 3, figsize=(W * 1.3, H), sharey=False)
    for ax, bench in zip(axes, focus):
        sub_all = df_c[df_c["benchmark"] == bench]
        base_val = sub_all[sub_all["coupling"] == "NONE"]["mutation_score"].median()
        ax.axhline(base_val, color=COLORS["NONE"], ls=":", lw=1.8,
                   label="NONE (baseline)")
        for coup in couplings:
            sub = sub_all[(sub_all["coupling"] == coup) &
                          (sub_all["diversity_cap"] == 1.0)]
            if sub.empty:
                continue
            g = sub.groupby("k")["mutation_score"].agg(["median","std"]).sort_index()
            ax.plot(g.index, g["median"],
                    marker=MARKERS[coup], color=COLORS[coup],
                    ls=linestyles[coup], lw=2.2, label=KOREAN[coup])
            ax.fill_between(g.index,
                            g["median"] - g["std"].fillna(0),
                            g["median"] + g["std"].fillna(0),
                            alpha=0.15, color=COLORS[coup])
        ax.set_title(bench, fontsize=14)
        ax.set_xlabel("Diversity pressure  $k$")
        ax.set_ylabel("Mutation Score")
    axes[0].legend(fontsize=10, loc="lower left")
    fig.suptitle("Mutation Score vs. k  (shading = +/-1 std dev, pop=5, real evolution)",
                 fontsize=14)
    fig.tight_layout()
    return _save(fig, "s04_phc_representative.png")

def fig_overamplification(df_c: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)

    titles = ["MULTIPLICATIVE — Score Collapse at High k", "CAPPED-MULT — Stable at High k"]
    coup_list = ["MULTIPLICATIVE", "CAPPED_MULTIPLICATIVE"]

    for ax, title, coup in zip(axes, titles, coup_list):
        for bench in ["Strings", "NumberUtils", "Soundex", "IEEE754rUtils"]:
            sub = df_c[(df_c["benchmark"] == bench) & (df_c["coupling"] == coup)
                       & (df_c["diversity_cap"] == 1.0)]
            if sub.empty:
                continue
            base = df_c[(df_c["benchmark"] == bench) &
                        (df_c["coupling"] == "NONE")]["mutation_score"].median()
            g = sub.groupby("k")["mutation_score"].median().sort_index()
            ax.plot(g.index, g.values / base, label=bench, lw=2.0)
        ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Baseline (=1)")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Diversity pressure  $k$")
        ax.set_ylim(0.0, 1.6)
    axes[0].set_ylabel("Mutation Score / Baseline")
    axes[0].legend(fontsize=10)
    fig.suptitle("Over-Amplification: Coupling Mode Comparison at High k", fontsize=14)
    fig.tight_layout()
    return _save(fig, "s05_overamplification.png")

def fig_phd_best_delta(df_d: pd.DataFrame):
    d = _deltas(df_d)
    best = d.loc[d.groupby("benchmark")["delta_ms"].idxmax()].copy()
    best = best.sort_values("delta_ms")

    fig, ax = plt.subplots(figsize=(W, H))
    colors = [COLORS.get(c, "#888") for c in best["coupling"]]
    bars = ax.barh(best["benchmark"], best["delta_ms"] * 100,
                   color=colors, edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars, best.iterrows()):
        v = row["delta_ms"] * 100
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2,
                f"{v:+.1f}%  ({KOREAN.get(row['coupling'],'')}, k={row['k']:g})",
                va="center", ha="left", fontsize=9)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Delta Mutation Score (%, vs. NONE baseline)")
    ax.set_title("Phase D: Large Class Improvement (pop=5, 120s budget)")
    legend_handles = [mpatches.Patch(color=COLORS[c], label=KOREAN[c])
                      for c in ["MULTIPLICATIVE","ADDITIVE","CAPPED_MULTIPLICATIVE"]]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11)
    ax.set_xlim(0, best["delta_ms"].max() * 100 + 14)
    return _save(fig, "s06_phd_best_delta.png")

def fig_domain_heatmap(df_c: pd.DataFrame, df_d: pd.DataFrame):
    df_all = pd.concat([df_c, df_d], ignore_index=True)
    d = _deltas(df_all)
    try:
        from automation import config as _cfg
        dom_m = {b.short_name: b.domain for b in _cfg.DEFAULT_BENCHMARKS}
        d["domain"] = d["benchmark"].map(dom_m).fillna("unknown")
    except Exception:
        d["domain"] = "unknown"

    best_bc = d.loc[d.groupby(["benchmark", "coupling"])["delta_ms"].idxmax()]
    couplings = ["MULTIPLICATIVE", "ADDITIVE", "CAPPED_MULTIPLICATIVE"]
    domains = ["numeric", "string", "encoding", "boolean"]

    matrix = np.full((len(couplings), len(domains)), np.nan)
    counts = np.zeros_like(matrix, dtype=int)
    for ri, coup in enumerate(couplings):
        for ci, dom in enumerate(domains):
            sub = best_bc[(best_bc["coupling"] == coup) & (best_bc["domain"] == dom)]
            if not sub.empty:
                matrix[ri, ci] = sub["delta_ms"].mean() * 100
                counts[ri, ci] = len(sub)

    fig, ax = plt.subplots(figsize=(W * 0.75, H))
    vmax = max(np.nanmax(np.abs(matrix)), 1.0)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="mean best-k Delta MS (%)")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([DOMAIN_KO[d] for d in domains], fontsize=12)
    ax.set_yticks(range(len(couplings)))
    ax.set_yticklabels([KOREAN[c] for c in couplings], fontsize=12)
    ax.set_title("Coupling Effect by Domain  (best k per class, domain mean)",
                 fontsize=13)

    for ri in range(len(couplings)):
        for ci in range(len(domains)):
            v = matrix[ri, ci]
            n = counts[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.1f}\n(n={n})", ha="center", va="center",
                        fontsize=11, color="black",
                        fontweight="bold" if abs(v) > 2 else "normal")
    fig.tight_layout()
    return _save(fig, "s07_domain_heatmap.png")

def fig_best_vs_baseline(df_c: pd.DataFrame, df_d: pd.DataFrame):
    df_all = pd.concat([df_c, df_d], ignore_index=True)
    d = _deltas(df_all)
    best = d.loc[d.groupby("benchmark")["delta_ms"].idxmax()].copy()
    try:
        from automation import config as _cfg
        dom_m = {b.short_name: b.domain for b in _cfg.DEFAULT_BENCHMARKS}
        best["domain"] = best["benchmark"].map(dom_m).fillna("unknown")
    except Exception:
        best["domain"] = "unknown"

    n_up = int((best["delta_ms"] > 0).sum())
    n_total = len(best)

    fig, ax = plt.subplots(figsize=(H * 1.15, H))
    ref = np.linspace(0, 1, 100)
    ax.plot(ref, ref, color="gray", ls="--", lw=1.2)
    ax.fill_between(ref, ref, 1.0, alpha=0.06, color="green")
    ax.text(0.62, 0.96, "improved region", color="green", fontsize=11,
            transform=ax.transAxes, va="top")
    ax.text(0.05, 0.04, f"{n_up}/{n_total} classes improved (best-config)",
            transform=ax.transAxes, fontsize=11, fontweight="bold")

    used_domains = set()
    for _, row in best.iterrows():
        bms, ams = row["baseline_ms"], row["mutation_score"]
        dom = row.get("domain", "unknown")
        col = DOMAIN_COLORS.get(dom, "#888")
        used_domains.add(dom)
        ax.scatter(bms, ams, s=130, color=col, edgecolors="black", lw=0.7, zorder=3)
        ax.annotate(row["benchmark"], (bms, ams),
                    xytext=(5, 4), textcoords="offset points", fontsize=8.5)

    domain_handles = [
        mpatches.Patch(color=DOMAIN_COLORS[d], label=DOMAIN_KO.get(d, d))
        for d in ["numeric", "string", "encoding", "boolean"]
        if d in used_domains
    ]
    ax.legend(handles=domain_handles, title="Domain", fontsize=10, loc="lower right")
    ax.set_xlabel("Baseline Mutation Score (NONE)")
    ax.set_ylabel("Best Achieved Mutation Score")
    ax.set_title("Best Configuration vs. Baseline  (Phase C+D, 13 classes)")
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.1, 0.95)
    fig.tight_layout()
    return _save(fig, "s08_best_vs_baseline.png")

def fig_joint_ms_entropy(df_c: pd.DataFrame, df_d: pd.DataFrame):
    df_all = pd.concat([df_c, df_d], ignore_index=True)
    d = _deltas(df_all)
    base_ent = (
        df_all[df_all["coupling"] == "NONE"]
        .groupby("benchmark")["entropy_norm"]
        .median()
    )
    best = d.loc[d.groupby("benchmark")["delta_ms"].idxmax()].copy()
    best["delta_entropy"] = best.apply(
        lambda r: r["entropy_norm"] - base_ent.get(r["benchmark"], r["entropy_norm"]),
        axis=1,
    )
    joint = (best["delta_ms"] > 0) & (best["delta_entropy"] > 0)

    fig, ax = plt.subplots(figsize=(W, H))
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.scatter(best.loc[~joint, "delta_entropy"] * 100,
               best.loc[~joint, "delta_ms"] * 100,
               s=100, color="#d62728", edgecolors="black", lw=0.5,
               label="MS up only or neither", zorder=2)
    ax.scatter(best.loc[joint, "delta_entropy"] * 100,
               best.loc[joint, "delta_ms"] * 100,
               s=110, color="#33a02c", edgecolors="black", lw=0.5,
               label="both MS and entropy up", zorder=3)
    for _, row in best.iterrows():
        ax.annotate(row["benchmark"],
                    (row["delta_entropy"] * 100, row["delta_ms"] * 100),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Delta normalised entropy (%, vs baseline)")
    ax.set_ylabel("Delta mutation score (%, vs baseline)")
    n_joint = int(joint.sum())
    ax.set_title(f"Hypothesis axes at best-config: {n_joint}/{len(best)} joint wins")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return _save(fig, "s15_joint_ms_entropy.png")

def fig_recommendation_table():
    rows = [
        ["Safe default",           "ADDITIVE",              "0.5 -- 1.0", "small gain; never collapses"],
        ["Tuned per class",        "MULTIPLICATIVE",        "0.2 -- 0.75","best MS on some classes"],
        ["High k, capped",         "CAPPED-MULT (c=1.0)",   "1 -- 4",     "class-dependent"],
        ["Large class + budget",   "ADDITIVE",              "0.5",        "needs 120s budget"],
        ["[WARNING] Avoid",        "MULT / EXPONENTIAL",    ">= 2",       "score collapse risk"],
        ["Not validated",          "any",                   "n/a",        "real-fault detection"],
    ]
    cols = ["Use Case", "Recommended Coupling", "k Range", "Expected effect"]

    fig, ax = plt.subplots(figsize=(W * 1.05, H * 0.7))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.2)

    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#f7fbff","#ffffff","#fff7ec","#f7fbff","#fee0d2"]
    for i, rc in enumerate(row_colors, 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(rc)

    ax.set_title("Optimal Configuration Guidelines", fontsize=15, pad=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "s09_recommendation_table.png")

def fig_experiment_pipeline():
    fig, ax = plt.subplots(figsize=(W, H * 0.72))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    phases = [
        ("Phase A/B\n탐색 스윕",
         "14 벤치마크\nk∈{0.25…8}\n3→10 seeds\n집단=50, 예산=40s\n→ 방향 결정",
         "#d0e8ff", 0.5),
        ("Phase C\n정밀 스윕",
         "8 벤치마크\nk∈{0.05…1.5}\nCAPPED 추가\n8 seeds, 집단=5\n예산=60s",
         "#d0f0d0", 3.0),
        ("Phase D\n대형 클래스",
         "6 대형 클래스\n집단=5, 예산=120s\nk∈{0.25…4}\n6 seeds",
         "#fde8c8", 5.5),
        ("최적 설정\n도출",
         "ADDITIVE k=0.5\nMULT k=0.5\nCAPPED(c=1) k≤4\n→ 13/14 개선",
         "#f0d0f0", 8.0),
    ]
    for title, body, color, xc in phases:
        rect = mpatches.FancyBboxPatch((xc - 1.1, 0.6), 2.2, 2.8,
                                        boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="#555555", lw=1.2)
        ax.add_patch(rect)
        ax.text(xc, 2.8, title, ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(xc, 2.3, body, ha="center", va="top", fontsize=9.5,
                multialignment="center", linespacing=1.5)

    for x_start in [1.6, 4.1, 6.6]:
        ax.annotate("", xy=(x_start + 0.28, 2.0), xytext=(x_start - 0.28, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    ax.text(5.0, 0.25, "총 ~3,790회 EvoSuite 실행  ·  14 벤치마크  ·  6 라이브러리  ·  4 도메인",
            ha="center", fontsize=11, style="italic", color="#444")
    ax.set_title("단계별 실험 파이프라인 구성", fontsize=14, pad=8)
    fig.tight_layout()
    return _save(fig, "s10_experiment_pipeline.png")

def main():
    print("Loading data …")
    df_c  = _load(PHASE_C_CSV)
    df_d  = _load(PHASE_D_CSV)

    print(f"  Phase C: {len(df_c)} rows | Phase D: {len(df_d)} rows")
    if df_c.empty:
        print("ERROR: Phase C data not found."); return

    print("Generating slide figures …")
    fig_formula_concept()
    fig_pop_ablation()
    fig_phc_best_delta(df_c)
    fig_phc_representative(df_c)
    fig_overamplification(df_c)
    if not df_d.empty:
        fig_phd_best_delta(df_d)
    fig_domain_heatmap(df_c, df_d if not df_d.empty else pd.DataFrame())
    fig_best_vs_baseline(df_c, df_d if not df_d.empty else pd.DataFrame())
    fig_joint_ms_entropy(df_c, df_d if not df_d.empty else pd.DataFrame())
    fig_recommendation_table()
    fig_experiment_pipeline()
    print(f"\nAll figures written to: {SLIDE_FIG_DIR}")

if __name__ == "__main__":
    main()
