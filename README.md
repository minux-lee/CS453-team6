# Evolutionary Test Generation with Diversity-Aware Mutation Fitness

**CS453 Team 6** — Search-Based Software Testing (SBST)

This project extends **EvoSuite** with a mutation-based fitness function that rewards not only how many mutants a test suite kills, but also **how diverse the killed mutant operator types are**, measured with Shannon entropy. The repository contains the modified EvoSuite source, a Python experiment automation pipeline, and analysis scripts used to evaluate the approach on 14 Java benchmark classes (~3,790 EvoSuite runs).

---

## Motivation

Standard mutation scores count **how many** mutants are killed, not **which operator types** they belong to. Two suites with the same score can differ sharply in fault-model coverage:

| Case | Killed mutants by type | Mutation score | Coverage profile |
|------|------------------------|----------------|------------------|
| A | $M_1{=}90,\ M_2{=}0,\ M_3{=}0$ | Same | Single operator type |
| B | $M_1{=}30,\ M_2{=}30,\ M_3{=}30$ | Same | Balanced across types |

**Hypothesis:** steering search toward diverse killed-mutant distributions improves mutation score without sacrificing it, and yields broader fault-model coverage.

**Verdict (after experiments):** *Conditional confirmation.* Tuned coupling improves mutation score on most classes, but effects are class-specific, rarely statistically significant.

---

## Approach

### Diversity metric

Let $c_i$ be killed mutants of operator type $i$, $p_i = c_i / \sum_j c_j$, and $n$ the number of operator types in the class:

$$
v = -\sum_{i=1}^{n} p_i \log p_i, \qquad
\hat{v} = \frac{v}{\log n} \in [0, 1], \qquad
\delta = 1 - \hat{v}
$$

Entropy is computed over **killed** mutants (not alive ones). Nine operator types are tracked (arithmetic, comparison, bitwise, conditional, etc.).

### Fitness function

EvoSuite minimises fitness (0 = all mutants killed):

$$
F_\text{base} = f_\text{branch} + \sum_{m \in \text{alive}} d_\text{min}(m)
$$

| Mode | Formula |
|------|---------|
| `NONE` | $F_\text{base}$ |
| `MULTIPLICATIVE` | $F_\text{base} \cdot (1 + k\delta)$ |
| `ADDITIVE` | $F_\text{base} + k\delta$ |
| `EXPONENTIAL` | $F_\text{base} \cdot e^{k\delta}$ |
| `CAPPED_MULTIPLICATIVE` | $F_\text{base} \cdot (1 + \min(k\delta, c))$ |

`CAPPED_MULTIPLICATIVE` was added because plain multiplicative coupling at large $k$ can collapse mutation scores—the penalty multiplier swamps $F_\text{base}$ and the GA optimises diversity alone.

### EvoSuite changes

| File | Change |
|------|--------|
| `evosuite/client/.../StrongMutationSuiteFitness.java` | Redesigned fitness; fixed 7 skeleton defects (NPE, NaN, inverted objective, etc.) |
| `evosuite/client/.../Properties.java` | `diversity_coupling`, `diversity_k`, `diversity_cap` |
| `evosuite/client/.../RuntimeVariable.java` | `MutantTypeEntropy`, `MutantTypeEntropyNorm` in `statistics.csv` |

**CLI parameters:** `-Ddiversity_coupling=`, `-Ddiversity_k=`, `-Ddiversity_cap=`

---

## Results

### Experimental setup

- **Benchmarks:** 14 classes from commons-math3, commons-lang3, commons-codec, commons-text, and Guava (numeric, string, encoding, boolean domains).
- **Phases:** A/B (directional scan, pop=50, 40s) → C (dense $k$, pop=5, 60s, 8 classes) → D (large classes, pop=5, 120s).
- **Control:** `-Dtest_archive=false` so suite-level fitness is the sole search driver.

Phase A/B used `population=50`, which exhausted the time budget before any evolution (`Generations=0` on all 14 classes). Phase C/D switched to `population=5`, yielding real evolution (~35 generations on average). **Primary evidence comes from Phase C/D.**

### Summary metrics (Phase C+D, 13 classes)

| Metric | Result |
|--------|--------|
| Mutation score improved at best-config | **12/13** |
| MS + entropy improved jointly | **9/13** |

### Notable per-class gains (best tuned config)

| Class | Best coupling | $k$ | Δ mutation score |
|-------|--------------|-----|------------------|
| Fraction | MULTIPLICATIVE | 0.75 | **+0.091** |
| Precision | ADDITIVE | 1.5 | **+0.079** |
| Soundex | MULTIPLICATIVE | 0.5 | **+0.044** |
| ArithmeticUtils | CAPPED_MULT | 0.5 | **+0.051** |
| CharUtils | — | — | **−0.054** (sole regression) |

### Practical recommendations

| Use case | Config |
|----------|--------|
| Safe default | `ADDITIVE`, $k = 0.5–1.0$ |
| Strongest gains | `MULTIPLICATIVE`, $k = 0.5–0.75$ |
| High pressure without collapse | `CAPPED_MULTIPLICATIVE`, $k = 2–4$, `cap=1.0` |
| Avoid | `MULTIPLICATIVE`, $k \geq 2$ (score collapse risk) |

### Limitations

- Results at `population=5` may not transfer to EvoSuite's default pop=50 without a larger budget.
- 40–120s budgets are short for large classes.
- `test_archive=false` lowers absolute scores vs default EvoSuite (deltas are relative to the same-setting baseline).
- Operator entropy was not validated against real faults (e.g. Defects4J).

---

## Repository layout

```
evosuite/              Modified EvoSuite + Benchmark_Commons JARs
automation/            Build, run, and resumable parameter sweeps
  config.py            Benchmark registry and sweep parameters
  sweep.py             Phase A sweep (default entry point)
  phase_b.py … phase_pop.py   Later experiment phases
  results/             Aggregated CSVs and per-run logs
analysis/              Result parsing, metrics, and plotting
  honest_audit.py      Print Phase C+D summary to stdout
  parser.py  metrics.py  plots.py  multi_phase.py
```

---

## How to run

### Prerequisites

Java 8+, Maven, Python 3.10+

### Setup

After cloning, build EvoSuite before running sweeps. 

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cd evosuite && mvn -q -pl client,master clean install -DskipTests
cd ..
```

### Experiment sweeps

```bash
# Phase A — broad scan (14 classes, pop=50, 40s)
./.venv/bin/python -m automation.sweep

# Smoke test (one class, 15s)
./.venv/bin/python -m automation.sweep --quick

# Phase B — confirmatory (needs automation/results/sweep_results.csv)
./.venv/bin/python -m automation.phase_b

# Phase C — dense k, pop=5, 60s (primary evidence)
./.venv/bin/python -m automation.phase_c

# Phase D — large classes, pop=5, 120s
./.venv/bin/python -m automation.phase_d

# Population ablation (pop 5 / 10 / 50)
./.venv/bin/python -m automation.phase_pop
```

Sweep flags: `--budget <s>`, `--parallelism <n>`, `--seeds 42 1234`, `--k-grid 0.5 1.0`, `--no-build`, `--no-resume`.

Results are written to `automation/results/` (e.g. `sweep_results.csv`, `sweep_results_phc.csv`, `sweep_results_phd.csv`). Sweeps are resumable: completed runs are skipped based on `config_id` in the CSV.

### Single EvoSuite run

```bash
java -jar evosuite/master/target/evosuite-master-1.2.1-SNAPSHOT.jar \
  -generateSuite \
  -class org.apache.commons.codec.language.Soundex \
  -projectCP "$(echo evosuite/Benchmark_Commons/lib/*.jar | tr ' ' ':')" \
  -seed 42 \
  -Dsearch_budget=60 \
  -Dpopulation=5 \
  -Ddiversity_coupling=ADDITIVE \
  -Ddiversity_k=0.5 \
  -Dtest_archive=false
```

### View results

After Phase C/D CSVs exist:

```bash
./.venv/bin/python -m analysis.honest_audit
```

Prints per-class best configs and the pooled honest summary (MS/entropy deltas, significance counts).

---

## References

1. McMinn, P. (2004). Search-based software test data generation: a survey. *STVR*, 14(2), 105–156.
2. Fraser, G., & Arcuri, A. (2011). EvoSuite: automatic test suite generation for Java classes. *ISSTA*.
3. Fraser, G., & Arcuri, A. (2014). Achieving higher test quality with mutation-based test generation. *TSE*, 40(9), 1041–1059.
4. Just, R., et al. (2014). Are mutants a valid substitute for real faults? *FSE*.
5. Papadakis, M., et al. (2019). Mutation testing advances. *Advances in Computers*, 112, 1–75.
