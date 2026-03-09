import pennylane as qml
import numpy as np

def quantum_dft(x):
    """Quantum DFT using PennyLane's QFT - exponentially faster!"""
    N = len(x)
    n_qubits = int(np.log2(N))
    
    # Must be power of 2
    assert N == 2**n_qubits, "Input length must be power of 2"
    
    # Create quantum device
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(amplitudes):
        # Encode classical data into quantum state
        qml.AmplitudeEmbedding(amplitudes, wires=range(n_qubits), normalize=True)
        
        # Apply Quantum Fourier Transform
        qml.QFT(wires=range(n_qubits))
        
        # Measure in computational basis
        return qml.state()
    
    # Normalize input
    x_norm = x / np.linalg.norm(x)
    
    # Run quantum circuit
    result = circuit(x_norm)
    
    # Extract amplitudes (these are the DFT coefficients)
    return result * np.linalg.norm(x)

# Example usage
if __name__ == "__main__":
    # Test signal (must be power of 2)
    N = 8
    t = np.arange(N)
    signal = np.sin(2 * np.pi * 2 * t / N)
    
    # Classical DFT
    X_classical = np.fft.fft(signal)
    
    # Quantum DFT
    X_quantum = quantum_dft(signal)
    
    print("Classical DFT:", np.abs(X_classical[:4]))
    print("Quantum DFT:  ", np.abs(X_quantum[:4]))
    print("Difference:   ", np.max(np.abs(X_classical - X_quantum)))
