import pennylane as qml
from pennylane import numpy as pnp

NUM_WIRES = 4
dev = qml.device("default.qubit", wires=NUM_WIRES)

@qml.qnode(dev)
def quantum_kernel(vec1, vec2):
    """
    Quantum kernel circuit that compares vec1 and vec2 via overlap.

    Returns the probability of measuring the all-zero state.
    """
    # Encode vec1
    for i in range(NUM_WIRES):
        qml.RY(vec1[i], wires=i)

    # Apply entanglement (optional but helpful)
    for i in range(NUM_WIRES - 1):
        qml.CNOT(wires=[i, i + 1])

    # Inverse encode vec2
    for i in range(NUM_WIRES):
        qml.RY(-vec2[i], wires=i)

    return qml.probs(wires=range(NUM_WIRES))

def compute_similarity(vec1, vec2):
    """
    Return the quantum similarity score between vec1 and vec2.

    Score = probability of returning to |0000> state.
    """
    probs = quantum_kernel(pnp.array(vec1), pnp.array(vec2))
    return float(probs[0])
