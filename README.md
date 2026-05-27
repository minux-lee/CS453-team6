# Evolutionary Test Generation with Diversity-Aware Mutation Fitness

This repository contains a course research project in **Search-Based Software Testing (SBST)**. We use **EvoSuite** (and only EvoSuite) to generate tests with a genetic algorithm, while experimenting with **mutation-based fitness functions** that reward not only how many mutants are killed, but also **how diverse the killed mutant types are**.

---

## Tech stack

- **Test generation / GA engine**: EvoSuite (source modification / extension where needed)
- **Mutation instrumentation**: EvoSuite's built-in mutation infrastructure (no external mutation tool)

Useful EvoSuite source references:
- `TestFitnessFunction`: `https://github.com/EvoSuite/evosuite/blob/master/client/src/main/java/org/evosuite/testcase/TestFitnessFunction.java`
- `MutationInstrumentation`: `https://github.com/EvoSuite/evosuite/blob/6d2e848c683e15ce9eb9a7ace506993ea46db022/client/src/main/java/org/evosuite/instrumentation/coverage/MutationInstrumentation.java#L66`
- `MutationFactory`: `https://github.com/EvoSuite/evosuite/blob/6d2e848c683e15ce9eb9a7ace506993ea46db022/client/src/main/java/org/evosuite/coverage/mutation/MutationFactory.java#L115`

---

## Project direction (updated)

### Why mutant diversity?

Mutation testing is commonly motivated by the hypothesis that **tests that kill mutants are also likely to reveal real faults**. Since real-world faults appear in many forms, a test suite that kills **a wide range of mutant operator types** may be more robust than a suite that overfits to a single operator type.

Example (3 mutation types \(M_1, M_2, M_3\)):

- Case A: \(M_1=100, M_2=0, M_3=0\)
- Case B: \(M_1=30, M_2=30, M_3=30\)

Both cases can have a similar overall mutation score, but Case B is hypothesized to be **safer** due to broader fault-model coverage.

### Quantifying diversity with entropy

Let \(c_i\) be the number of killed mutants of type \(i\), and \(p_i = \frac{c_i}{\sum_j c_j}\).
We define the mutant-type diversity value \(v\) using **Shannon entropy**:

\[
v = -\sum_{i=1}^{n} p_i \log(p_i)
\]

Properties we want from \(v\):
- **Continuity** and smooth changes as the distribution changes
- **Monotonicity** under the uniform distribution (more covered types increases diversity)
- A bounded range: \(0 \le v \le \log(n)\), where \(n\) is the number of mutation operator types considered

Reference: `https://en.wikipedia.org/wiki/Entropy_(information_theory)`

---

## Fitness function (diversity-aware)

We start from an existing mutation-based fitness baseline (e.g., a strong-mutation-style objective and/or RIP-inspired components used within EvoSuite), and then incorporate diversity.

We will initially evaluate multiplicative coupling:

- **Baseline**: \(F_{base}\)
- **Diversity-weighted**: \(F = F_{base}\cdot(1 + k\cdot v)\)

where \(k\) controls how strongly diversity affects selection pressure.

If needed, we will also compare alternative combinations (additive, exponentiation, progressive schedules), for example:
- \(F = F_{base} + k\cdot v\)
- \(F = v^{\alpha}\cdot F_{base}^{\beta}\)

---

## Evaluation plan

We will:
- **Sweep \(k\)** across multiple values and run EvoSuite for a fixed budget (time/fitness evaluations) per configuration.
- Track:
  - overall mutation score (final evaluation)
  - diversity value \(v\) during/after evolution
  - resulting fitness trends across iterations
- Plot curves to identify an **optimal region** where diversity improves without degrading overall effectiveness.

The final comparison will still report standard mutation-testing metrics (e.g., mutation score), with diversity used as an internal optimization signal.

---

## References

1. McMinn, P. (2004). Search-based software test data generation: a survey. *Software Testing, Verification and Reliability*, 14(2), 105-156.
2. Fraser, G., & Arcuri, A. (2011). EvoSuite: automatic test suite generation for Java classes. In *Proceedings of the 19th ACM SIGSOFT symposium*.
3. Fraser, G., & Arcuri, A. (2014). Achieving Higher Test Quality with Mutation-Based Test Generation. *IEEE Transactions on Software Engineering*, 40(9), 1041-1059.
4. Just, R., Jalali, S., Inozemtseva, L., Ernst, M. D., Holmes, R., & Fraser, G. (2014). Are mutants a valid substitute for real faults in software testing?. In *Proceedings of the 22nd ACM SIGSOFT International Symposium on Foundations of Software Engineering* (pp. 654-665).
5. Papadakis, M., et al. (2019). Mutation testing advances: from software quality to software engineering. *Advances in Computers*, 112, 1-75.