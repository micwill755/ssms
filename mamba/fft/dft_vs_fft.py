import numpy as np
import matplotlib.pyplot as plt
import time

def dft_minimal(x):
    """Minimal DFT implementation - O(N²)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X

'''
example: signal [1, 0, 1, 0] (N=4), n = 0-3, k = 0-3

After recursion, we have:

Even FFT: [2, 0] (from [1, 1
Odd FFT: [0, 0] (from [0, 0])

Now we need to build the full 4-point FFT from these 2-point pieces.

The butterfly operations are:

For k=0:
twiddle = e^(-2π*0/4) = e^0 = 1
X[0] = even[0] + 1 * odd[0] = 2 + 1*0 = 2
X[2] = even[0] - 1 * odd[0] = 2 - 1*0 = 2

For k=1:
twiddle = e^(-2π*1/4) = e^(-π/2) = -j
X[1] = even[1] + (-j) * odd[1] = 0 + (-j)*0 = 0
X[3] = even[1] - (-j) * odd[1] = 0 - (-j)*0 = 0

Result: [2, 0, 2, 0] 

'''

def fft_minimal(x):
    """Minimal FFT implementation - O(N log N)"""
    N = len(x)
    
    # Base case: if N is 1, return the input
    if N <= 1:
        return x
    
    '''
    DFT Formula: X[k] = Σ(n=0 to N-1) x[n] * e^(-2πjkn/N)
    Split the sum into even and odd indices:
    X[k] = Σ(even n) x[n] * e^(-2πjkn/N) + Σ(odd n) x[n] * e^(-2πjkn/N)
    
    '''
    # Divide: split into even and odd indices
    even = fft_minimal(x[0::2])  # x[0], x[2], x[4], ...
    odd = fft_minimal(x[1::2])   # x[1], x[3], x[5], ...
    
    # Conquer: combine results
    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        # Twiddle factor: e^(-2πjk/N)
        twiddle = np.exp(-2j * np.pi * k / N)
        
        # Butterfly operation - X[k] = Even[k] + e^(-2πjk/N) * Odd[k]
        X[k] = even[k] + twiddle * odd[k]
        X[k + N//2] = even[k] - twiddle * odd[k]
    
    return X

def compare_dft_vs_fft():
    """Compare DFT vs FFT - same results, different speed!"""
    
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
        if fft_time > 0:
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
    
    print("=== Visual Proof: DFT and FFT are Identical ===")
    
    # Compute both
    X_dft = dft_minimal(signal)
    X_fft = fft_minimal(signal)
    
    # Check difference
    diff = np.abs(X_dft - X_fft)
    max_diff = np.max(diff)
    
    print(f"Maximum difference: {max_diff:.2e}")
    print("If difference is ~1e-15 or smaller, they're identical!")
    print()
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Magnitude comparison
    axes[0,0].stem(range(N//2), np.abs(X_dft[:N//2]), basefmt=' ')
    axes[0,0].set_title('DFT Magnitude')
    axes[0,0].set_ylabel('Magnitude')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].stem(range(N//2), np.abs(X_fft[:N//2]), basefmt=' ')
    axes[0,1].set_title('FFT Magnitude')
    axes[0,1].grid(True, alpha=0.3)
    
    # Difference (should be ~0)
    axes[1,0].stem(range(N//2), diff[:N//2], basefmt=' ')
    axes[1,0].set_title('Difference (DFT - FFT)')
    axes[1,0].set_ylabel('Difference')
    axes[1,0].set_xlabel('Frequency bin')
    axes[1,0].grid(True, alpha=0.3)
    
    # Both overlaid
    axes[1,1].stem(range(N//2), np.abs(X_dft[:N//2]), basefmt=' ', 
                   linefmt='b-', markerfmt='bo', label='DFT')
    axes[1,1].stem(range(N//2), np.abs(X_fft[:N//2]), basefmt=' ', 
                   linefmt='r--', markerfmt='rx', label='FFT')
    axes[1,1].set_title('Both Overlaid (should overlap perfectly)')
    axes[1,1].set_xlabel('Frequency bin')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def explain_fft_algorithm():
    """Explain how FFT works step by step"""
    print("=== How FFT Works ===")
    print()
    print("FFT uses 'Divide and Conquer':")
    print("1. DIVIDE: Split signal into even/odd indices")
    print("   - Even: x[0], x[2], x[4], x[6], ...")
    print("   - Odd:  x[1], x[3], x[5], x[7], ...")
    print()
    print("2. CONQUER: Recursively compute FFT of smaller pieces")
    print("   - FFT(even) and FFT(odd)")
    print()
    print("3. COMBINE: Use 'butterfly operations'")
    print("   - X[k] = Even[k] + twiddle * Odd[k]")
    print("   - X[k+N/2] = Even[k] - twiddle * Odd[k]")
    print("   - twiddle = e^(-2πjk/N)")
    print()
    print("Key insight: Reuses calculations from smaller subproblems!")
    print("Instead of N² operations, only N log N needed.")

if __name__ == "__main__":
    print("Choose what to run:")
    print("1. Speed comparison")
    print("2. Visual proof they're identical")
    print("3. Explain FFT algorithm")
    print("4. All of the above")
    
    choice = input("Enter 1, 2, 3, or 4: ").strip()
    
    if choice == "1":
        compare_dft_vs_fft()
    elif choice == "2":
        show_identical_results()
    elif choice == "3":
        explain_fft_algorithm()
    elif choice == "4":
        compare_dft_vs_fft()
        print("\n" + "="*50 + "\n")
        show_identical_results()
        print("\n" + "="*50 + "\n")
        explain_fft_algorithm()
    else:
        print("Running speed comparison by default...")
        compare_dft_vs_fft()