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

def show_continuous_and_discrete():
    """Show both continuous waves and discrete samples"""
    print("\n=== Visual Comparison: Continuous vs Discrete ===")
    
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

if __name__ == "__main__":
    show_continuous_and_discrete()