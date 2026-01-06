import numpy as np
import time

def sequential_scan(A, B, u, h0=0):
    """Sequential scan - O(n) time, can't parallelize"""
    # SSM: h[t] = A * h[t-1] + B * u[t]
    n = len(u)
    h = np.zeros(n + 1)
    h[0] = h0
    
    for t in range(n):
        h[t + 1] = A * h[t] + B * u[t]
    
    return h[1:]  # Return h[1] to h[n]

def parallel_scan_naive(A, B, u, h0=0):
    """Parallel scan using closed form - O(1) per element if precomputed"""
    # Key insight: h[t] = A^t * h0 + Σ(i=0 to t-1) A^(t-1-i) * B * u[i]
    n = len(u)
    h = np.zeros(n)
    
    # Precompute powers of A
    A_powers = np.array([A**i for i in range(n)])
    
    for t in range(n):
        # Direct formula: no dependencies!
        h[t] = A_powers[t] * h0
        for i in range(t):
            h[t] += A_powers[t-1-i] * B * u[i]
    
    return h

def parallel_scan_efficient(elements):
    """Efficient parallel scan using divide-and-conquer
    
    Elements should be tuples (a, b) representing operations:
    result = a * prev_result + b
    
    This is the general form that includes SSM recurrence.
    """
    n = len(elements)
    if n == 1:
        return elements
    
    # Divide: split into pairs
    # Combine adjacent elements using associativity
    combined = []
    for i in range(0, n-1, 2):
        a1, b1 = elements[i]
        a2, b2 = elements[i+1]
        # Combine: (a2*a1, a2*b1 + b2)
        combined.append((a2 * a1, a2 * b1 + b2))
    
    # Handle odd length
    if n % 2 == 1:
        combined.append(elements[-1])
    
    # Conquer: recursively solve smaller problem
    if len(combined) > 1:
        combined = parallel_scan_efficient(combined)
    
    # Expand back to original size
    result = [None] * n
    result[0] = elements[0]
    
    for i in range(1, n):
        if i % 2 == 1:
            # Odd indices: use combined result
            a_combined, b_combined = combined[i // 2]
            if i // 2 == 0:
                result[i] = (elements[0][0] * a_combined, elements[0][0] * b_combined + elements[0][1])
            else:
                prev_a, prev_b = result[i-2] if i >= 2 else (1, 0)
                result[i] = (prev_a * a_combined, prev_a * b_combined + prev_b)
        else:
            # Even indices: compute from previous
            a_prev, b_prev = result[i-1]
            a_curr, b_curr = elements[i]
            result[i] = (a_prev * a_curr, a_prev * b_curr + b_prev)
    
    return result

def ssm_to_scan_elements(A, B, u):
    """Convert SSM parameters to scan elements"""
    # Each step: h[t] = A * h[t-1] + B * u[t]
    # This is in the form: result = a * prev + b
    # where a = A, b = B * u[t]
    return [(A, B * ut) for ut in u]

def parallel_scan_ssm(A, B, u, h0=0):
    """Parallel scan for SSM using efficient algorithm"""
    # Convert to scan elements
    elements = ssm_to_scan_elements(A, B, u)
    
    # Apply parallel scan
    scan_result = parallel_scan_efficient(elements)
    
    # Extract final values
    h = np.zeros(len(u))
    current_val = h0
    
    for i, (a, b) in enumerate(scan_result):
        current_val = a * current_val + b
        h[i] = current_val
    
    return h

def demonstrate_scan_concept():
    """Show basic scan concept with simple examples"""
    print("=== Understanding Scan ===")
    print()
    
    # Basic cumulative sum
    x = [1, 2, 3, 4, 5]
    cumsum = []
    acc = 0
    for val in x:
        acc += val
        cumsum.append(acc)
    
    print(f"Input:     {x}")
    print(f"Cumsum:    {cumsum}")
    print(f"NumPy:     {list(np.cumsum(x))}")
    print()
    
    # Cumulative product
    x = [2, 3, 2, 2]
    cumprod = []
    acc = 1
    for val in x:
        acc *= val
        cumprod.append(acc)
    
    print(f"Input:     {x}")
    print(f"Cumprod:   {cumprod}")
    print(f"NumPy:     {list(np.cumprod(x))}")
    print()
    
    print("Scan = generalized 'running operation' on sequences")
    print("Key: Each output depends on ALL previous inputs")

def compare_sequential_vs_parallel():
    """Compare sequential vs parallel scan for SSMs"""
    print("=== Sequential vs Parallel Scan for SSMs ===")
    print()
    
    # SSM parameters
    A, B = 0.9, 1.0
    u = np.array([1, 2, 1, 3, 2, 1, 4])
    h0 = 0
    
    print(f"SSM: h[t] = {A} * h[t-1] + {B} * u[t]")
    print(f"Input sequence: {u}")
    print()
    
    # Sequential computation
    print("Sequential computation (must go step by step):")
    h_seq = sequential_scan(A, B, u, h0)
    for t in range(len(u)):
        if t == 0:
            print(f"h[{t+1}] = {A} * {h0} + {B} * {u[t]} = {h_seq[t]:.3f}")
        else:
            print(f"h[{t+1}] = {A} * {h_seq[t-1]:.3f} + {B} * {u[t]} = {h_seq[t]:.3f}")
    
    print()
    
    # Parallel computation
    print("Parallel computation (closed form):")
    h_par = parallel_scan_naive(A, B, u, h0)
    print("Each h[t] computed independently using:")
    print("h[t] = A^t * h0 + Σ(i=0 to t-1) A^(t-1-i) * B * u[i]")
    print()
    
    # Verify they're identical
    print("Results comparison:")
    print(f"Sequential: {h_seq}")
    print(f"Parallel:   {h_par}")
    print(f"Identical:  {np.allclose(h_seq, h_par)}")
    print(f"Max diff:   {np.max(np.abs(h_seq - h_par)):.2e}")

def show_parallel_scan_advantage():
    """Show why parallel scan is faster for long sequences"""
    print("\n=== Parallel Scan Performance Advantage ===")
    print()
    
    # Test different sequence lengths
    A, B = 0.95, 1.0
    lengths = [100, 1000, 5000]
    
    for n in lengths:
        print(f"--- Sequence length: {n} ---")
        
        # Generate random input
        np.random.seed(42)
        u = np.random.randn(n)
        
        # Time sequential scan
        start = time.time()
        h_seq = sequential_scan(A, B, u)
        seq_time = time.time() - start
        
        # Time parallel scan (naive version for simplicity)
        start = time.time()
        h_par = parallel_scan_naive(A, B, u)
        par_time = time.time() - start
        
        print(f"Sequential time: {seq_time*1000:.2f} ms")
        print(f"Parallel time:   {par_time*1000:.2f} ms")
        print(f"Results match:   {np.allclose(h_seq, h_par)}")
        print()
    
    print("Note: True parallel scan advantage comes with GPU/parallel hardware!")
    print("Sequential: O(n) time, must wait for each step")
    print("Parallel:   O(log n) time with n processors")

def explain_ssm_scan_connection():
    """Explain how scan relates to SSM computation"""
    print("=== Why Scan Matters for SSMs ===")
    print()
    
    print("SSM Recurrence: h[t] = A * h[t-1] + B * u[t]")
    print()
    print("Problem: This looks inherently sequential!")
    print("- h[1] depends on h[0]")
    print("- h[2] depends on h[1]") 
    print("- h[3] depends on h[2]")
    print("- Can't compute h[100] without computing h[1] through h[99]")
    print()
    
    print("Solution: Rewrite using scan!")
    print("- Express each h[t] in terms of initial state h[0] and all inputs")
    print("- h[t] = A^t * h[0] + Σ A^(t-1-i) * B * u[i]")
    print("- Now h[t] can be computed independently (in parallel)")
    print()
    
    print("This is exactly what modern SSM libraries do:")
    print("- Training: Use parallel scan (fast, GPU-friendly)")
    print("- Inference: Use recurrent form (memory efficient)")
    print()
    
    print("Analogy to FFT:")
    print("- FFT: Makes convolution O(n log n) instead of O(n²)")
    print("- Scan: Makes recurrence O(log n) instead of O(n) sequential")

def demo_associative_property():
    """Show why scan works - associativity"""
    print("=== Why Parallel Scan Works: Associativity ===")
    print()
    
    print("Key insight: SSM operations are associative!")
    print()
    
    # Simple example with 4 elements
    A, B = 0.8, 1.0
    u = [1, 2, 3, 4]
    
    print("Sequential computation:")
    print("h[1] = A*h[0] + B*u[0]")
    print("h[2] = A*h[1] + B*u[1] = A*(A*h[0] + B*u[0]) + B*u[1]")
    print("     = A²*h[0] + A*B*u[0] + B*u[1]")
    print("h[3] = A*h[2] + B*u[2] = A*(A²*h[0] + A*B*u[0] + B*u[1]) + B*u[2]")
    print("     = A³*h[0] + A²*B*u[0] + A*B*u[1] + B*u[2]")
    print()
    
    print("Parallel computation (divide and conquer):")
    print("Split [u[0], u[1], u[2], u[3]] into pairs:")
    print("Pair 1: [u[0], u[1]] → combined effect")
    print("Pair 2: [u[2], u[3]] → combined effect") 
    print("Then combine the two pairs!")
    print()
    
    print("This works because matrix multiplication is associative:")
    print("(AB)C = A(BC)")

if __name__ == "__main__":
    print("Parallel Scan for State Space Models")
    print("=" * 40)
    
    # Run all demonstrations
    demonstrate_scan_concept()
    print("\n" + "="*50 + "\n")
    
    compare_sequential_vs_parallel()
    print("\n" + "="*50 + "\n")
    
    explain_ssm_scan_connection()
    print("\n" + "="*50 + "\n")
    
    demo_associative_property()
    print("\n" + "="*50 + "\n")
    
    show_parallel_scan_advantage()