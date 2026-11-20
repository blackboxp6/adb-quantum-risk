
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

BASE = "outputs/"

# Load state preparation
with open(BASE + "three_country_state_preparation.qpy", "rb") as f:
    A = qpy.load(f)[0]

# Load Grover operator
with open(BASE + "three_country_grover.qpy", "rb") as f:
    G = qpy.load(f)[0]

# Load marked outcomes
data = np.load(BASE + "three_country_quantum_input.npz")
marked = data["marked"]

n = A.num_qubits
dimension = 2 ** n

marked_padded = np.zeros(dimension, dtype=bool)
marked_padded[:len(marked)] = marked

# Initial state
initial = Statevector.from_instruction(A)

p = np.sum(np.abs(initial.data[marked_padded]) ** 2)
theta = np.arcsin(np.sqrt(p))

print("=" * 60)
print("THREE-COUNTRY GROVER ITERATION EXPERIMENT")
print("=" * 60)

print(f"Initial probability: {p:.10%}")

results = []

# Test zero through twelve iterations
for k in range(13):

    circuit = QuantumCircuit(n)
    circuit.compose(A, inplace=True)

    for _ in range(k):
        circuit.compose(G, inplace=True)

    state = Statevector.from_instruction(circuit)

    actual = np.sum(
        np.abs(state.data[marked_padded]) ** 2
    )

    theoretical = np.sin((2 * k + 1) * theta) ** 2

    results.append((k, actual, theoretical))

    print(
        f"Iterations: {k:2d} | "
        f"Actual: {actual:.10%} | "
        f"Theoretical: {theoretical:.10%} | "
        f"Error: {abs(actual-theoretical):.3e}"
    )

# Find best tested iteration
best = max(results, key=lambda x: x[1])

print("\nBEST TESTED RESULT")
print(f"Iterations: {best[0]}")
print(f"Success probability: {best[1]:.10%}")

# Save results
np.savetxt(
    BASE + "three_country_grover_iterations.csv",
    np.array(results),
    delimiter=",",
    header="iterations,actual_probability,theoretical_probability",
    comments=""
)

# Plot
iterations = [r[0] for r in results]
actual = [r[1] * 100 for r in results]
theoretical = [r[2] * 100 for r in results]

plt.figure(figsize=(9, 5))
plt.plot(iterations, actual, "o-", label="Actual statevector")
plt.plot(iterations, theoretical, "--", label="Theory")

plt.xlabel("Grover iterations")
plt.ylabel("Marked probability (%)")
plt.title("Three-Country Tail-Risk Amplitude Amplification")
plt.xticks(iterations)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE + "three_country_grover_iterations.png",
    dpi=300
)

plt.show()

print("\nResults saved.")
print("GROVER ITERATION EXPERIMENT COMPLETE.")
