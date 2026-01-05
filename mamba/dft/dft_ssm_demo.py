import numpy as np
import matplotlib.pyplot as plt

def dft_minimal(x):
    """Minimal DFT implementation - O(N²)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X

def ssm_frequency_response(A, B, C, frequencies):
    """How SSM responds to different frequencies"""
    responses = []
    for freq in frequencies:
        # z = e^(jω) for frequency ω
        z = np.exp(1j * freq)
        # Transfer function: H(z) = C(zI - A)^(-1)B
        H = C @ np.linalg.inv(z * np.eye(len(A)) - A) @ B
        responses.append(H)
    return np.array(responses)

def demo_dft_ssm_connection():
    """Show how DFT connects to SSM frequency analysis"""
    
    print("=== DFT & SSM Connection Demo ===")
    print("This shows how DFT reveals frequencies and SSMs filter them")
    print()
    
    # Create a simple signal with known frequencies
    N = 64
    t = np.arange(N)
    signal = np.sin(2 * np.pi * 3 * t / N) + 0.5 * np.sin(2 * np.pi * 7 * t / N)
    
    print("1. Created signal with:")
    print("   - 3 cycles per 64 samples (frequency bin 3)")
    print("   - 7 cycles per 64 samples (frequency bin 7)")
    print()
    
    # DFT analysis
    X = dft_minimal(signal)
    
    # Find strong frequencies
    magnitudes = np.abs(X)
    strong_freqs = np.where(magnitudes > N/4)[0]
    
    print("2. DFT Analysis Results:")
    print(f"   Signal length: {N} samples")
    print(f"   Strong frequencies found at bins: {strong_freqs}")
    print(f"   Magnitude at bin 3: {magnitudes[3]:.1f}")
    print(f"   Magnitude at bin 7: {magnitudes[7]:.1f}")
    print()
    
    # Simple SSM that acts like a filter
    A = np.array([[0.9]])  # Decay factor
    B = np.array([[1.0]])  # Input scaling  
    C = np.array([[1.0]])  # Output scaling
    
    # SSM frequency response
    omega = np.linspace(0, np.pi, N//2)
    H_ssm = ssm_frequency_response(A, B, C, omega)
    
    print("3. SSM Frequency Response:")
    print("   SSM acts as a frequency filter")
    print(f"   Response at low frequencies: {np.abs(H_ssm[0]).item():.2f}")
    print(f"   Response at high frequencies: {np.abs(H_ssm[-1]).item():.2f}")
    print()
    
    print("=== Key Insights ===")
    print("• DFT decomposes signals into frequency components")
    print("• SSMs can be analyzed in frequency domain")
    print("• Both work with the same frequency representations")
    print("• This connection enables fast SSM computation using FFT")
    
    # Visualize the connection
    visualize_dft_ssm_connection(signal, X, H_ssm, omega, N)

def visualize_dft_ssm_connection(signal, X, H_ssm, omega, N):
    """Visualize the DFT-SSM connection"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original signal
    axes[0,0].plot(signal[:32], 'b-', linewidth=2)
    axes[0,0].set_title('Original Signal (first 32 samples)')
    axes[0,0].set_xlabel('Sample')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True, alpha=0.3)
    
    # DFT magnitude spectrum
    freqs = np.arange(N//2)
    axes[0,1].stem(freqs, np.abs(X[:N//2]), basefmt=' ')
    axes[0,1].set_title('DFT: Frequency Content')
    axes[0,1].set_xlabel('Frequency bin')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].grid(True, alpha=0.3)
    
    # SSM frequency response
    axes[1,0].plot(omega, np.abs(H_ssm).flatten(), 'r-', linewidth=2)
    axes[1,0].set_title('SSM: Frequency Response')
    axes[1,0].set_xlabel('Frequency (rad)')
    axes[1,0].set_ylabel('|H(ω)|')
    axes[1,0].grid(True, alpha=0.3)
    
    # Combined view - simplified without alpha
    axes[1,1].stem(freqs, np.abs(X[:N//2]), basefmt=' ', label='Signal spectrum')
    ax2 = axes[1,1].twinx()
    ax2.plot(omega * N / (2*np.pi), np.abs(H_ssm).flatten(), 'r-', linewidth=2, label='SSM response')
    axes[1,1].set_title('DFT + SSM: Signal & Filter')
    axes[1,1].set_xlabel('Frequency bin')
    axes[1,1].set_ylabel('Signal magnitude', color='b')
    ax2.set_ylabel('SSM response', color='r')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_dft_ssm_connection()