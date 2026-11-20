
# ADB Quantum Risk Model — Evaluation and Benchmarking

## 1. Objective

This evaluation investigates whether Quantum Amplitude Estimation (QAE)
can improve the estimation of joint disaster risk metrics relative
to classical sampling methods.

The evaluation focuses on estimation accuracy, computational
resources, scalability, and sensitivity to quantum noise.

The two-country benchmark is based on the ADB challenge reference
case involving Kyrgyzstan and Tajikistan.

---

## 2. Evaluation Experiments

### Experiment 1 — Estimation Error vs Computational Resources

Methods compared:

- Monte Carlo
- Stratified Monte Carlo
- Importance Sampling
- Sobol sampling
- Quantum Maximum Likelihood Amplitude Estimation (QMLAE)

The objective is to investigate estimation accuracy as the
computational budget increases.

Five risk metrics are evaluated:

- Expected principal loss
- Probability that both countries pay
- Probability that portfolio loss exceeds $80 million
- Expected stop-loss
- Probability that both bonds are exhausted

The QMLAE results represent an ideal statistical simulation
and are not results from physical quantum hardware.

---

### Experiment 2 — Circuit Depth vs Number of Countries

This experiment investigates how the quantum circuit resources
increase as the number of modeled countries increases.

Resources examined include:

- Number of qubits
- Circuit depth
- Total gates
- CX gates
- U gates

The scaling experiment uses synthetic probability distributions
to investigate state-preparation complexity.

It is not a fully calibrated multi-country ADB risk model.

---

### Experiment 3 — Estimation Error vs Quantum Noise

This experiment investigates the sensitivity of QMLAE to
simulated gate and readout noise.

Noise is applied to the compiled circuits through simulated
depolarizing error models.

The experiment is intended to investigate hardware feasibility
and sensitivity to noise.

These noise parameters are simulation settings and do not
represent the measured performance of a real quantum processor.

---

### Experiment 4 — Computational Cost Comparison

Classical computational runtime is measured for:

- Monte Carlo
- Stratified Monte Carlo
- Importance Sampling
- Sobol sampling

Quantum resource estimation includes:

- Qubits
- Grover oracle applications
- State preparations
- CX gate executions
- Total gate executions
- Circuit depth

The quantum resource results are based on compiled logical
circuits and do not represent actual quantum hardware runtime.

The ideal statistical QMLAE benchmark runtime must not be
interpreted as quantum processor execution time.

---

## 3. Scientific Limitations

1. Quantum experiments are simulator-based.
2. No physical quantum hardware experiment has been conducted.
3. Classical methods use a precomputed probability distribution.
4. Equal sample or shot budgets are not equivalent to equal
   computational costs.
5. State-preparation complexity is a significant consideration.
6. Quantum noise is simulated and does not fully represent
   actual hardware behavior.
7. Circuit scaling experiments use synthetic distributions.
8. No quantum advantage is claimed.

---

## 4. Research Goal

The objective is to provide a transparent and reproducible
benchmark of quantum amplitude estimation for joint disaster
risk estimation.

The investigation evaluates whether QAE offers an accuracy
advantage under selected resource constraints and whether
its resource requirements and noise sensitivity suggest
potential future feasibility.
