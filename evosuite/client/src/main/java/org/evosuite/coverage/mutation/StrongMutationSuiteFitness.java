/*
 * Copyright (C) 2010-2018 Gordon Fraser, Andrea Arcuri and EvoSuite
 * contributors
 *
 * This file is part of EvoSuite.
 *
 * EvoSuite is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * EvoSuite is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
 */
package org.evosuite.coverage.mutation;

import org.evosuite.Properties;
import org.evosuite.rmi.ClientServices;
import org.evosuite.statistics.RuntimeVariable;
import org.evosuite.instrumentation.mutation.*;
import org.evosuite.testcase.TestCase;
import org.evosuite.testcase.TestChromosome;
import org.evosuite.testcase.execution.ExecutionResult;
import org.evosuite.testcase.execution.ExecutionTrace;
import org.evosuite.testsuite.TestSuiteChromosome;

import java.util.*;
import java.util.Map.Entry;

/**
 * <p>
 * StrongMutationSuiteFitness class.
 * </p>
 *
 * @author fraser
 */
public class StrongMutationSuiteFitness extends MutationSuiteFitness {

    private static final long serialVersionUID = -9124328839917834720L;

    // Added by CS453 Team 6
    private static final String[] MUTATION_OPERATORS = {
            DeleteField.NAME, DeleteStatement.NAME, InsertUnaryOperator.NAME, ReplaceVariable.NAME,
            ReplaceConstant.NAME, ReplaceArithmeticOperator.NAME, ReplaceComparisonOperator.NAME,
            ReplaceBitwiseOperator.NAME, NegateCondition.NAME
    };

    // Added by CS453 Team 6
    private static final int NUM_BUCKETS = MUTATION_OPERATORS.length + 1;

    // Added by CS453 Team 6
    private final Map<Integer, Integer> mutantTypeCache = new HashMap<>();

    // Added by CS453 Team 6
    private int operatorBucketsPresent = -1;

    // Added by CS453 Team 6
    private double lastEntropy = 0.0;

    // Added by CS453 Team 6
    private double lastEntropyNorm = 0.0;

    public StrongMutationSuiteFitness() {
        super(Properties.Criterion.STRONGMUTATION);
    }

    // Added by CS453 Team 6
    private static int operatorBucketOf(Mutation mutant) {
        String name = mutant.getMutationName();
        for (int i = 0; i < MUTATION_OPERATORS.length; i++) {
            if (name.startsWith(MUTATION_OPERATORS[i])) {
                return i;
            }
        }
        return MUTATION_OPERATORS.length; // OTHER
    }

    // Added by CS453 Team 6
    private double computeDiversityDeficiency(Set<Integer> newKilled) {
        for (Map.Entry<Integer, MutationTestFitness> entry : this.mutantMap.entrySet()) {
            mutantTypeCache.computeIfAbsent(entry.getKey(),
                    id -> operatorBucketOf(entry.getValue().getMutation()));
        }

        if (operatorBucketsPresent < 0) {
            Set<Integer> present = new LinkedHashSet<>(mutantTypeCache.values());
            operatorBucketsPresent = Math.max(present.size(), 1);
        }

        Set<Integer> killed = new LinkedHashSet<>(removedMutants);
        killed.addAll(newKilled);

        long[] counts = new long[NUM_BUCKETS];
        long total = 0;
        for (Integer id : killed) {
            Integer bucket = mutantTypeCache.get(id);
            if (bucket != null) {
                counts[bucket]++;
                total++;
            }
        }

        if (total == 0 || operatorBucketsPresent <= 1) {
            this.lastEntropy = 0.0;
            this.lastEntropyNorm = 0.0;
            return 0.0;
        }

        double entropy = 0.0;
        for (long count : counts) {
            if (count > 0) {
                double p = (double) count / (double) total;
                entropy -= p * Math.log(p);
            }
        }

        double vHat = entropy / Math.log(operatorBucketsPresent);
        if (vHat < 0.0) {
            vHat = 0.0;
        } else if (vHat > 1.0) {
            vHat = 1.0;
        }

        this.lastEntropy = entropy;
        this.lastEntropyNorm = vHat;
        return 1.0 - vHat;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ExecutionResult runTest(TestCase test) {
        return runTest(test, null);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ExecutionResult runTest(TestCase test, Mutation mutant) {

        return StrongMutationTestFitness.runTest(test, mutant);
    }

    /**
     * Create a list of test cases ordered by their execution time. The
     * precondition is that all TestChromomes have been executed such that they
     * have an ExecutionResult.
     *
     * @param individual
     * @return
     */
    private List<TestChromosome> prioritizeTests(TestSuiteChromosome individual) {
        List<TestChromosome> executionOrder = new ArrayList<>(individual.getTestChromosomes());
        executionOrder.sort(Comparator.comparingLong(tch -> tch.getLastExecutionResult().getExecutionTime()));
        return executionOrder;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.evosuite.ga.FitnessFunction#getFitness(org.evosuite.ga.Chromosome)
     */

    /**
     * {@inheritDoc}
     */
    @Override
    public double getFitness(TestSuiteChromosome suite) {
        runTestSuite(suite);

        // Set<MutationTestFitness> uncoveredMutants =
        // MutationTestPool.getUncoveredFitnessFunctions();

        for (TestChromosome test : suite.getTestChromosomes()) {
            ExecutionResult result = test.getLastExecutionResult();

            if (result.hasTimeout() || result.hasTestException()) {
                logger.debug("Skipping test with timeout");
                double fitness = branchFitness.totalGoals * 2
                        + branchFitness.totalMethods + 3 * this.numMutants;
                updateIndividual(suite, fitness);
                suite.setCoverage(this, 0.0);
                logger.info("Test case has timed out, setting fitness to max value "
                        + fitness);
                return fitness;
            }
        }

        // First objective: achieve branch coverage
        logger.debug("Calculating branch fitness: ");
        boolean archive = Properties.TEST_ARCHIVE;
        Properties.TEST_ARCHIVE = false;
        double branchValue = branchFitness.getFitness(suite);
        Properties.TEST_ARCHIVE = archive;

        Set<Integer> touchedMutants = new LinkedHashSet<>();
        Map<Mutation, Double> minMutantFitness = new LinkedHashMap<>();

        // For each mutant that is not in the archive:
        // 3 -> not covered
        // 1..2 -> infection distance
        // 0..1 -> propagation distance
        for (Integer mutantId : this.mutantMap.keySet()) {
            MutationTestFitness mutantFitness = mutantMap.get(mutantId);
            minMutantFitness.put(mutantFitness.getMutation(), 3.0);
        }

        int mutantsChecked = 0;
        int numKilled = removedMutants.size();
        Set<Integer> newKilled = new LinkedHashSet<>();

        List<TestChromosome> executionOrder = prioritizeTests(suite); // Quicker tests first
        for (TestChromosome test : executionOrder) {
            ExecutionResult result = test.getLastExecutionResult();
            // Using private reflection can lead to false positives
            // that represent unrealistic behaviour. Thus, we only
            // use reflection for basic criteria, not for mutation
            if (result.calledReflection())
                continue;

            ExecutionTrace trace = result.getTrace();
            touchedMutants.addAll(trace.getTouchedMutants());
            logger.debug("Tests touched " + touchedMutants.size() + " mutants");

            Map<Integer, Double> touchedMutantsDistances = trace.getMutationDistances();
            if (touchedMutantsDistances.isEmpty()) {
                // if 'result' does not touch any mutant, no need to continue
                continue;
            }

            for (final Entry<Integer, MutationTestFitness> entry : this.mutantMap.entrySet()) {
                int mutantID = entry.getKey();
                if (newKilled.contains(mutantID)) {
                    continue;
                }
                MutationTestFitness goal = entry.getValue();

                if (MutationTimeoutStoppingCondition.isDisabled(goal.getMutation())) {
                    logger.debug("Skipping timed out mutation " + goal.getMutation().getId());
                    continue;
                }

                mutantsChecked++;

                double mutantInfectionDistance = 3.0;
                boolean hasBeenTouched = touchedMutantsDistances.containsKey(mutantID);

                if (hasBeenTouched) {
                    // Infection happened, so we need to check propagation
                    if (touchedMutantsDistances.get(mutantID) == 0.0) {
                        logger.debug("Executing test against mutant " + goal.getMutation());

                        mutantInfectionDistance = goal.getFitness(test, result); // archive is updated by the
                                                                                 // TestFitnessFunction class
                    } else {
                        // We can skip calling the test fitness function since we already know
                        // fitness is 1.0 (for propagation) + infection distance
                        mutantInfectionDistance = 1.0 + normalize(touchedMutantsDistances.get(mutantID));
                    }
                } else {
                    mutantInfectionDistance = goal.getFitness(test, result); // archive is updated by the
                                                                             // TestFitnessFunction class
                }

                if (mutantInfectionDistance == 0.0) {
                    numKilled++;
                    newKilled.add(mutantID);
                    result.test.addCoveredGoal(goal); // update list of covered goals
                    this.toRemoveMutants.add(mutantID); // goal to not be considered by the next iteration of the
                                                        // evolutionary algorithm
                } else {
                    minMutantFitness.put(goal.getMutation(),
                            Math.min(mutantInfectionDistance, minMutantFitness.get(goal.getMutation())));
                }
            }
        }

        // ============================================================
        // Diversity-Aware Mutation Fitness (CS453 Team6)
        // ============================================================

        double fBase = branchValue;
        for (Double fit : minMutantFitness.values()) {
            fBase += fit;
        }

        double diversityDeficiency = computeDiversityDeficiency(newKilled);
        ClientServices.getInstance().getClientNode().trackOutputVariable(
                RuntimeVariable.MutantTypeEntropy, this.lastEntropy);
        ClientServices.getInstance().getClientNode().trackOutputVariable(
                RuntimeVariable.MutantTypeEntropyNorm, this.lastEntropyNorm);

        double fitness;
        double k = Properties.DIVERSITY_K;
        switch (Properties.DIVERSITY_COUPLING) {
            case MULTIPLICATIVE:
                fitness = fBase * (1.0 + k * diversityDeficiency);
                break;
            case ADDITIVE:
                fitness = fBase + k * diversityDeficiency;
                break;
            case EXPONENTIAL:
                fitness = fBase * Math.exp(k * diversityDeficiency);
                break;
            case CAPPED_MULTIPLICATIVE:
                fitness = fBase * (1.0 + Math.min(k * diversityDeficiency, Properties.DIVERSITY_CAP));
                break;
            case NONE:
            default:
                fitness = fBase;
                break;
        }

        logger.debug("Mutants killed: {}, Checked: {}, Goals: {}, diversityDeficiency: {})", numKilled,
                mutantsChecked, this.numMutants, diversityDeficiency);

        updateIndividual(suite, fitness);

        assert numKilled == newKilled.size() + removedMutants.size();
        assert numKilled <= this.numMutants;
        double coverage = (double) numKilled / (double) this.numMutants;
        assert coverage >= 0.0 && coverage <= 1.0;
        suite.setCoverage(this, coverage);
        suite.setNumOfCoveredGoals(this, numKilled);

        return fitness;
    }

}
