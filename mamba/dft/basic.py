import numpy as np
import matplotlib.pyplot as plt

def dft_minimal(x):
    """Minimal DFT implementation - O(N²)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            ''' 
            This is the DFT formula! It comes from Fourier analysis theory.
            x[n] * np.exp(-2j * np.pi * k * n / N)

            Breaking down the formula:

            x[n] = your signal value at time n
            k = the frequency we're testing for (0, 1, 2, ..., N-1)
            n = time index (0, 1, 2, ..., N-1)
            exp(-2j * π * k * n / N) = a "test wave" at frequency k

            The intuition:

            1. Create a test wave that oscillates k times: exp(-2j * π * k * n / N)
            2. Multiply your signal by this test wave: x[n] * test_wave[n]
            3. Sum up all the products

            '''
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X


def fft_minimal(x):
    """Minimal FFT implementation - O(N log N)"""
    N = len(x)
    
    # Base case: if N is 1, return the input
    if N <= 1:
        return x
    
    # Divide: split into even and odd indices
    even = fft_minimal(x[0::2])  # x[0], x[2], x[4], ...
    odd = fft_minimal(x[1::2])   # x[1], x[3], x[5], ...
    
    # Conquer: combine results
    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        # Twiddle factor: e^(-2πjk/N)
        twiddle = np.exp(-2j * np.pi * k / N)
        
        # Butterfly operation
        X[k] = even[k] + twiddle * odd[k]
        X[k + N//2] = even[k] - twiddle * odd[k]
    
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
    
    # Create a simple signal
    N = 64
    t = np.arange(N)
    signal = np.sin(2 * np.pi * 3 * t / N) + 0.5 * np.sin(2 * np.pi * 7 * t / N)
    
    # DFT analysis
    X = dft_minimal(signal)
    freqs = np.fft.fftfreq(N, 1/N)
    
    # Simple SSM that acts like a filter
    A = np.array([[0.9]])  # Decay factor
    B = np.array([[1.0]])  # Input scaling
    C = np.array([[1.0]])  # Output scaling
    
    # SSM frequency response
    omega = np.linspace(0, np.pi, N//2)
    H_ssm = ssm_frequency_response(A, B, C, omega)
    
    print("=== DFT & SSM Connection ===")
    print(f"Signal length: {N}")
    print(f"DFT reveals frequencies at bins: {np.where(np.abs(X) > N/4)[0]}")
    print(f"SSM acts as frequency filter with response shape")
    print("Key insight: Both analyze signals in frequency domain!")

def visualize_signal():
    N = 64
    t = np.arange(N)
    
    # Individual components
    wave1 = np.sin(2 * np.pi * 3 * t / N)
    wave2 = 0.5 * np.sin(2 * np.pi * 7 * t / N)
    signal = wave1 + wave2
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, wave1, '--', label='3 Hz component')
    plt.plot(t, wave2, '--', label='7 Hz component (0.5x)')
    plt.plot(t, signal, 'k-', linewidth=2, label='Combined signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Signal = sin(3ω) + 0.5*sin(7ω)')
    plt.grid(True, alpha=0.3)
    plt.show()

def show_continuous_and_discrete():
    """Show both continuous waves and discrete samples"""
    N = 16  # More points for better visualization
    n_discrete = np.arange(N)
    n_continuous = np.linspace(0, N-1, 200)  # Smooth curve
    
    # Signal (1 cycle)
    signal_discrete = np.sin(2 * np.pi * 1 * n_discrete / N)
    signal_continuous = np.sin(2 * np.pi * 1 * n_continuous / N)
    
    # Test waves
    test_k1_discrete = np.real(np.exp(-2j * np.pi * 1 * n_discrete / N))
    test_k1_continuous = np.real(np.exp(-2j * np.pi * 1 * n_continuous / N))
    
    test_k2_discrete = np.real(np.exp(-2j * np.pi * 2 * n_discrete / N))
    test_k2_continuous = np.real(np.exp(-2j * np.pi * 2 * n_continuous / N))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Matching case (k=1)
    axes[0,0].plot(n_continuous, signal_continuous, 'b-', alpha=0.7, label='Continuous')
    axes[0,0].plot(n_continuous, test_k1_continuous, 'r-', alpha=0.7, label='Test k=1')
    axes[0,0].stem(n_discrete, signal_discrete, basefmt=' ', linefmt='b-', markerfmt='bo')
    axes[0,0].stem(n_discrete, test_k1_discrete, basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[0,0].set_title('MATCHING: Signal vs Test k=1')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Products for k=1
    products_k1 = signal_discrete * test_k1_discrete
    axes[0,1].stem(n_discrete, products_k1, basefmt=' ')
    axes[0,1].set_title(f'Products → Sum = {np.sum(products_k1):.1f} ✓')
    axes[0,1].grid(True, alpha=0.3)
    
    # Canceling case (k=2)
    axes[1,0].plot(n_continuous, signal_continuous, 'b-', alpha=0.7, label='Continuous')
    axes[1,0].plot(n_continuous, test_k2_continuous, 'r-', alpha=0.7, label='Test k=2')
    axes[1,0].stem(n_discrete, signal_discrete, basefmt=' ', linefmt='b-', markerfmt='bo')
    axes[1,0].stem(n_discrete, test_k2_discrete, basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[1,0].set_title('CANCELING: Signal vs Test k=2')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Products for k=2
    products_k2 = signal_discrete * test_k2_discrete
    axes[1,1].stem(n_discrete, products_k2, basefmt=' ')
    axes[1,1].set_title(f'Products → Sum = {np.sum(products_k2):.1f} ✗')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_dft_vs_fft():
    """Compare DFT vs FFT - same results, different speed!"""
    import time
    
    print("=== DFT vs FFT Comparison ===")
    print("Key point: IDENTICAL results, different algorithms!")
    print()
    
    # Test with power-of-2 sizes (FFT requirement)
    sizes = [16, 64, 256]
    
    for N in sizes:
        print(f"--- Testing with N = {N} samples ---")
        
        # Create test signal
        t = np.arange(N)
        signal = np.sin(2 * np.pi * 2 * t / N) + 0.3 * np.sin(2 * np.pi * 5 * t / N)
        
        # Time DFT (our slow implementation)
        start_time = time.time()
        X_dft = dft_minimal(signal)
        dft_time = time.time() - start_time
        
        # Time FFT (our fast implementation)
        start_time = time.time()
        X_fft = fft_minimal(signal)
        fft_time = time.time() - start_time
        
        # Check if results are identical (within floating point precision)
        max_diff = np.max(np.abs(X_dft - X_fft))
        
        print(f"DFT time: {dft_time*1000:.2f} ms")
        print(f"FFT time: {fft_time*1000:.2f} ms")
        print(f"Speedup: {dft_time/fft_time:.1f}x faster")
        print(f"Max difference: {max_diff:.2e} (should be ~0)")
        print(f"Results identical: {max_diff < 1e-10}")
        print()
    
    # Show the complexity difference
    print("=== Why FFT is faster ===")
    print("DFT: O(N²) operations")
    print("FFT: O(N log N) operations")
    print()
    for N in [64, 256, 1024]:
        dft_ops = N**2
        fft_ops = N * np.log2(N)
        speedup = dft_ops / fft_ops
        print(f"N={N}: DFT={dft_ops:,} ops, FFT={fft_ops:.0f} ops, speedup={speedup:.0f}x")

def show_identical_results():
    """Visually show DFT and FFT give identical results"""
    N = 32
    t = np.arange(N)
    signal = np.sin(2 * np.pi * 3 * t / N) + 0.5 * np.sin(2 * np.pi * 7 * t / N)
    
    # Compute both
    X_dft = dft_minimal(signal)
    X_fft = fft_minimal(signal)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Magnitude comparison
    axes[0,0].stem(range(N//2), np.abs(X_dft[:N//2]), basefmt=' ', label='DFT')
    axes[0,0].set_title('DFT Magnitude')
    axes[0,0].set_ylabel('Magnitude')
    
    axes[0,1].stem(range(N//2), np.abs(X_fft[:N//2]), basefmt=' ', label='FFT')
    axes[0,1].set_title('FFT Magnitude')
    
    # Difference (should be ~0)
    diff = np.abs(X_dft - X_fft)
    axes[1,0].stem(range(N//2), diff[:N//2], basefmt=' ')
    axes[1,0].set_title('Difference (DFT - FFT)')
    axes[1,0].set_ylabel('Difference')
    axes[1,0].set_xlabel('Frequency bin')
    
    # Both overlaid
    axes[1,1].stem(range(N//2), np.abs(X_dft[:N//2]), basefmt=' ', 
                   linefmt='b-', markerfmt='bo', label='DFT')
    axes[1,1].stem(range(N//2), np.abs(X_fft[:N//2]), basefmt=' ', 
                   linefmt='r--', markerfmt='rx', label='FFT')
    axes[1,1].set_title('Both Overlaid (should overlap perfectly)')
    axes[1,1].set_xlabel('Frequency bin')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Maximum difference: {np.max(diff):.2e}")
    print("If difference is ~1e-15 or smaller, they're identical!")

if __name__ == "__main__":
    print("Choose what to run:")
    print("1. DFT vs FFT speed comparison")
    print("2. Visual proof they're identical")
    print("3. Both")
    
    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice == "1":
        compare_dft_vs_fft()
    elif choice == "2":
        show_identical_results()
    elif choice == "3":
        compare_dft_vs_fft()
        show_identical_results()
    else:
        print("Running speed comparison by default...")
        compare_dft_vs_fft()