
from pathlib import Path
import numpy as np
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

# Inspect the quantum input data
input_file = OUTPUTS / "three_country_quantum_input.npz"

print("\n=== QUANTUM INPUT DATA ===")

with np.load(input_file) as data:
    for key in data.files:
        arr = data[key]
        print(
            f"{key}: shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

# Inspect the saved circuits
circuit_files = [
    "three_country_state_preparation.qpy",
    "three_country_loss_oracle.qpy",
    "three_country_grover.qpy",
]

for filename in circuit_files:
    path = OUTPUTS / filename

    print(f"\n=== {filename} ===")

    with path.open("rb") as file:
        circuits = qpy.load(file)

    circuit = circuits[0]

    print("Qubits:", circuit.num_qubits)
    print("Classical bits:", circuit.num_clbits)
    print("Depth:", circuit.depth())
    print("Gate count:", circuit.size())
    print("Operations:", circuit.count_ops())
