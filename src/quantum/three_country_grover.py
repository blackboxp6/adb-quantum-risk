
import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector

# --------------------------------------------------
# 1. LOAD STATE PREPARATION AND ORACLE
# --------------------------------------------------

BASE = "outputs/"

with open(BASE + "three_country_state_preparation.qpy", "rb") as f:
    A = qpy.load(f)[0]

with open(BASE + "three_country_loss_oracle.qpy", "rb") as f:
    oracle = qpy.load(f)[0]

n = A.num_qubits
dimension = 2 ** n

print("=" * 55)
print("THREE-COUNTRY GROVER OPERATOR")
print("=" * 55)
print(f"Qubits: {n}")
print(f"Basis states: {dimension}")

# --------------------------------------------------
# 2. BUILD THE ZERO-STATE REFLECTION
# --------------------------------------------------
# This changes the phase of |000...000>.
#
# X gates temporarily convert the all-zero state
# into the all-one state.
# H-MCX-H creates a multi-controlled Z operation.
# The final X gates restore the original basis.

zero_reflection = QuantumCircuit(n, name="S0")

zero_reflection.x(range(n))
zero_reflection.h(n - 1)
zero_reflection.mcx(list(range(n - 1)), n - 1)
zero_reflection.h(n - 1)
zero_reflection.x(range(n))

# --------------------------------------------------
# 3. CONSTRUCT ONE GROVER ITERATION
# --------------------------------------------------
# Circuit application order:
# Oracle -> A-dagger -> zero reflection -> A
#
# This implements G = A S0 A-dagger O.
# A global phase does not affect measurement probabilities.

grover = QuantumCircuit(n, name="Grover")

grover.compose(oracle, inplace=True)
grover.compose(A.inverse(), inplace=True)
grover.compose(zero_reflection, inplace=True)
grover.compose(A, inplace=True)

print("\nGrover operator constructed.")
print("Operations:", grover.count_ops())

# --------------------------------------------------
# 4. APPLY ONE ITERATION TO THE PREPARED STATE
# --------------------------------------------------

test_circuit = QuantumCircuit(n)
test_circuit.compose(A, inplace=True)
test_circuit.compose(grover, inplace=True)

state = Statevector.from_instruction(test_circuit)

# --------------------------------------------------
# 5. THEORETICAL MARKED-PROBABILITY CHECK
# --------------------------------------------------

initial_probability = 0.007263678322602455

theta = np.arcsin(np.sqrt(initial_probability))

expected_probability = np.sin(3 * theta) ** 2

print("\nIdeal amplitude amplification")
print(f"Initial marked probability: {initial_probability:.10%}")
print(f"After one iteration:         {expected_probability:.10%}")

# --------------------------------------------------
# 6. TRANSPILATION AND RESOURCE ESTIMATES
# --------------------------------------------------

print("\nTranspiling Grover operator...")

compiled = transpile(
    grover,
    basis_gates=["u", "cx"],
    optimization_level=1
)

print("\nGrover resource estimates")
print(f"Depth: {compiled.depth()}")
print(f"Total gates: {compiled.size()}")
print(f"U gates: {compiled.count_ops().get('u', 0)}")
print(f"CX gates: {compiled.count_ops().get('cx', 0)}")

# --------------------------------------------------
# 7. SAVE THE CIRCUIT
# --------------------------------------------------

output_path = BASE + "three_country_grover.qpy"

with open(output_path, "wb") as f:
    qpy.dump(compiled, f)

print("\nSaved Grover operator:")
print(output_path)

print("\nGROVER OPERATOR CONSTRUCTION COMPLETE.")
