from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVOSUITE_DIR = REPO_ROOT / "evosuite"
EVOSUITE_JAR = EVOSUITE_DIR / "master" / "target" / "evosuite-master-1.2.1-SNAPSHOT.jar"
BENCHMARK_LIB_DIR = EVOSUITE_DIR / "Benchmark_Commons" / "lib"

RESULTS_DIR = REPO_ROOT / "automation" / "results"
RUN_LOG_DIR = RESULTS_DIR / "logs"
RAW_STATS_DIR = RESULTS_DIR / "raw"
AGGREGATED_CSV = RESULTS_DIR / "sweep_results.csv"
PHASE_C_CSV = RESULTS_DIR / "sweep_results_phc.csv"
PHASE_D_CSV = RESULTS_DIR / "sweep_results_phd.csv"
PHASE_POP_CSV = RESULTS_DIR / "sweep_results_pop_control.csv"

FITNESS_SOURCE = (
    EVOSUITE_DIR
    / "client/src/main/java/org/evosuite/coverage/mutation/StrongMutationSuiteFitness.java"
)

OUTPUT_VARIABLES: Sequence[str] = (
    "TARGET_CLASS",
    "criterion",
    "Coverage",
    "Total_Goals",
    "Covered_Goals",
    "MutantTypeEntropy",
    "MutantTypeEntropyNorm",
    "Size",
    "Length",
    "Generations",
    "Total_Time",
    "Random_Seed",
)


@dataclass(frozen=True)
class Benchmark:
    target_class: str
    classpath: List[str]
    library: str
    domain: str
    note: str

    @property
    def short_name(self) -> str:
        return self.target_class.rsplit(".", 1)[-1]


def _all_jars() -> List[str]:
    jars = sorted(str(p) for p in BENCHMARK_LIB_DIR.glob("*.jar"))
    return jars


_CP = _all_jars()

D_NUMERIC = "numeric"
D_STRING = "string"
D_BOOLEAN = "boolean"
D_ENCODING = "encoding"

DEFAULT_BENCHMARKS: List[Benchmark] = [
    Benchmark("org.apache.commons.math3.util.ArithmeticUtils", _CP,
              "commons-math3", D_NUMERIC,
              "Integer arithmetic / gcd / lcm / pow: arithmetic, bitwise, comparison."),
    Benchmark("org.apache.commons.math3.fraction.Fraction", _CP,
              "commons-math3", D_NUMERIC,
              "Rational arithmetic: arithmetic, comparison, constants, conditionals."),
    Benchmark("org.apache.commons.math3.util.Precision", _CP,
              "commons-math3", D_NUMERIC,
              "Floating-point comparison/rounding: arithmetic, bitwise, comparison."),
    Benchmark("org.apache.commons.math3.complex.Complex", _CP,
              "commons-math3", D_NUMERIC,
              "Complex arithmetic: arithmetic, comparison, NaN/Inf conditionals."),
    Benchmark("com.google.common.math.IntMath", _CP,
              "guava", D_NUMERIC,
              "Checked integer math: arithmetic, bitwise, overflow comparison."),
    Benchmark("org.apache.commons.lang3.math.IEEE754rUtils", _CP,
              "commons-lang3", D_NUMERIC,
              "IEEE min/max: comparison-dominated, NaN handling."),
    Benchmark("org.apache.commons.lang3.math.NumberUtils", _CP,
              "commons-lang3", D_NUMERIC,
              "Numeric parsing / min-max: comparison, conditionals, constants."),
    Benchmark("org.apache.commons.lang3.CharUtils", _CP,
              "commons-lang3", D_STRING,
              "Character classification: comparison, constants, conditionals."),
    Benchmark("org.apache.commons.text.WordUtils", _CP,
              "commons-text", D_STRING,
              "Word wrapping / capitalisation: conditionals, comparison, loops."),
    Benchmark("com.google.common.base.Strings", _CP,
              "guava", D_STRING,
              "String padding / null handling: comparison, conditionals, constants."),
    Benchmark("org.apache.commons.text.similarity.LevenshteinDistance", _CP,
              "commons-text", D_STRING,
              "Edit-distance DP: arithmetic + comparison; low operator diversity."),
    Benchmark("org.apache.commons.lang3.BooleanUtils", _CP,
              "commons-lang3", D_BOOLEAN,
              "Boolean logic: negate-condition, comparison, constants."),
    Benchmark("org.apache.commons.codec.binary.Hex", _CP,
              "commons-codec", D_ENCODING,
              "Hex encode/decode: bitwise, arithmetic, comparison."),
    Benchmark("org.apache.commons.codec.language.Soundex", _CP,
              "commons-codec", D_ENCODING,
              "Phonetic encoding: comparison, conditionals, char arithmetic."),
]

COUPLING_NONE = "NONE"
COUPLING_MULT = "MULTIPLICATIVE"
COUPLING_ADD = "ADDITIVE"
COUPLING_EXP = "EXPONENTIAL"

ACTIVE_COUPLINGS = (COUPLING_MULT, COUPLING_ADD, COUPLING_EXP)


@dataclass(frozen=True)
class SweepConfig:
    benchmarks: List[Benchmark] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS))
    couplings: Sequence[str] = ACTIVE_COUPLINGS
    k_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    seeds: Sequence[int] = (42, 1234, 2024)
    search_budget_s: int = 40
    run_timeout_s: int = 240
    criterion: str = "STRONGMUTATION"
    parallelism: int = 8
    client_mem_mb: int = 1500
    population: int = 50


@dataclass(frozen=True)
class RunSpec:
    benchmark: Benchmark
    coupling: str
    k: float
    seed: int
    search_budget_s: int
    run_timeout_s: int
    criterion: str
    client_mem_mb: int
    population: int = 50
    diversity_cap: float = 1.0

    @property
    def config_id(self) -> str:
        k_token = "na" if self.coupling == COUPLING_NONE else f"k{self.k:g}"
        pop_token = f"p{self.population}" if self.population != 50 else ""
        cap_token = f"cap{self.diversity_cap:g}" if self.coupling == "CAPPED_MULTIPLICATIVE" else ""
        suffix = ("_" + pop_token + cap_token).rstrip("_") if (pop_token or cap_token) else ""
        return f"{self.benchmark.short_name}__{self.coupling}__{k_token}__seed{self.seed}{suffix}"


def enumerate_runs(cfg: SweepConfig) -> List[RunSpec]:
    runs: List[RunSpec] = []

    def _mk(bench: Benchmark, coupling: str, k: float, seed: int, cap: float = 1.0) -> RunSpec:
        return RunSpec(
            benchmark=bench,
            coupling=coupling,
            k=k,
            seed=seed,
            search_budget_s=cfg.search_budget_s,
            run_timeout_s=cfg.run_timeout_s,
            criterion=cfg.criterion,
            client_mem_mb=cfg.client_mem_mb,
            population=cfg.population,
            diversity_cap=cap,
        )

    for bench, seed in product(cfg.benchmarks, cfg.seeds):
        runs.append(_mk(bench, COUPLING_NONE, 0.0, seed))
        for coupling, k in product(cfg.couplings, cfg.k_grid):
            runs.append(_mk(bench, coupling, k, seed))

    return runs


def quick_config() -> SweepConfig:
    return SweepConfig(
        benchmarks=[DEFAULT_BENCHMARKS[1]],
        couplings=(COUPLING_MULT,),
        k_grid=(2.0,),
        seeds=(42,),
        search_budget_s=15,
    )


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, RUN_LOG_DIR, RAW_STATS_DIR):
        os.makedirs(d, exist_ok=True)
