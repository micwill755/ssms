import numpy as np

def dft_minimal(x):
    """Minimal DFT implementation - O(N²)"""
    
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    ''' 
        this is exactly the Vandermonde matrix multiplication written out as nested loops.

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        
        is exactly:
        V[k,n] = ω^(kn)  where ω = e^(-2πi/N)
        X = V @ x
    '''
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X

def dft_vandermonde(x):
    """Same DFT using explicit Vandermonde matrix"""
    N = len(x)
    omega = np.exp(-2j * np.pi / N)
    
    # Build Vandermonde matrix: V[k,n] = ω^(kn)
    V = omega ** np.outer(range(N), range(N))
    
    return V @ x

def show_equivalence():
    """Show your loops = Vandermonde multiplication"""
    x = np.array([1, 2, 3, 4])
    
    result1 = dft_minimal(x)
    result2 = dft_vandermonde(x)
    
    print("Your nested loops:", result1)
    print("Vandermonde matrix:", result2)
    print("Identical:", np.allclose(result1, result2))
    
    # Show the matrix
    N = len(x)
    omega = np.exp(-2j * np.pi / N)
    V = omega ** np.outer(range(N), range(N))
    
    print(f"\nVandermonde matrix (ω = e^(-2πi/{N})):")
    print("V[k,n] = ω^(kn)")
    print(V)

if __name__ == "__main__":
    show_equivalence()