from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = REPO_ROOT / "automation" / "results" / "sweep_results.csv"
PHASE_C_CSV = REPO_ROOT / "automation" / "results" / "sweep_results_phc.csv"
PHASE_D_CSV = REPO_ROOT / "automation" / "results" / "sweep_results_phd.csv"
PHASE_POP_CSV = REPO_ROOT / "automation" / "results" / "sweep_results_pop_control.csv"


def _benchmark_metadata():
    try:
        from automation import config as _cfg

        lib = {b.short_name: b.library for b in _cfg.DEFAULT_BENCHMARKS}
        dom = {b.short_name: b.domain for b in _cfg.DEFAULT_BENCHMARKS}
        return lib, dom
    except Exception:
        return {}, {}


_RENAME = {
    "es_Coverage": "mutation_score",
    "es_MutantTypeEntropy": "entropy",
    "es_MutantTypeEntropyNorm": "entropy_norm",
    "es_Total_Goals": "total_goals",
    "es_Covered_Goals": "covered_goals",
    "es_Size": "suite_size",
    "es_Length": "suite_length",
    "es_Total_Time": "total_time_ms",
}

_NUMERIC = [
    "k",
    "seed",
    "search_budget_s",
    "wall_time_s",
    "mutation_score",
    "entropy",
    "entropy_norm",
    "total_goals",
    "covered_goals",
    "suite_size",
    "suite_length",
    "total_time_ms",
]


def load_results(csv_path: Optional[Path] = None, only_ok: bool = True) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else DEFAULT_RESULTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns=_RENAME)

    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if only_ok and "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if df.empty:
        raise ValueError(f"No usable (status==ok) rows in {path}")

    df["benchmark"] = df["target_class"].str.rsplit(".", n=1).str[-1]
    if "config_id" in df.columns:
        df["population"] = (
            df["config_id"].str.extract(r"_p(\d+)$")[0]
            .astype(float)
            .fillna(50)
            .astype(int)
        )
        df["diversity_cap"] = (
            df["config_id"].str.extract(r"cap([\d.]+)")[0].astype(float).fillna(1.0)
        )
    lib_map, dom_map = _benchmark_metadata()
    df["library"] = df["benchmark"].map(lib_map).fillna("unknown")
    df["domain"] = df["benchmark"].map(dom_map).fillna("unknown")
    df["config_label"] = df.apply(
        lambda r: r["coupling"]
        if r["coupling"] == "NONE"
        else f"{r['coupling']}(k={r['k']:g})",
        axis=1,
    )
    return df


def benchmarks(df: pd.DataFrame) -> list:
    return sorted(df["benchmark"].unique())


def couplings(df: pd.DataFrame) -> list:
    order = ["NONE", "MULTIPLICATIVE", "ADDITIVE", "EXPONENTIAL", "CAPPED_MULTIPLICATIVE"]
    present = set(df["coupling"].unique())
    return [c for c in order if c in present]


def load_phase_cd(only_ok: bool = True) -> pd.DataFrame:
    frames = []
    for path in (PHASE_C_CSV, PHASE_D_CSV):
        if path.exists():
            frames.append(load_results(path, only_ok=only_ok))
    if not frames:
        raise FileNotFoundError("No Phase C/D CSVs found")
    return pd.concat(frames, ignore_index=True)


def load_all_phases(only_ok: bool = True) -> dict:
    out = {}
    if DEFAULT_RESULTS_CSV.exists():
        out["ab"] = load_results(DEFAULT_RESULTS_CSV, only_ok=only_ok)
    if PHASE_C_CSV.exists():
        out["c"] = load_results(PHASE_C_CSV, only_ok=only_ok)
    if PHASE_D_CSV.exists():
        out["d"] = load_results(PHASE_D_CSV, only_ok=only_ok)
    if PHASE_POP_CSV.exists():
        out["pop"] = load_results(PHASE_POP_CSV, only_ok=only_ok)
    return out
