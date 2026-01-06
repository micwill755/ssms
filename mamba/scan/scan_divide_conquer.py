import numpy as np

def sequential_ssm_scan(u, A, B):
    """Sequential SSM scan - O(N) time"""
    h = []
    h_prev = 0
    for ut in u:
        h_next = A * h_prev + B * ut
        h.append(h_next)
        h_prev = h_next
    return h

def naive_parallel_ssm_scan(u, A, B):
    """Naive parallel SSM - O(N²) time, shows mathematical equivalence"""
    h = []
    for t in range(len(u)):
        h_t = 0
        for i in range(t + 1):
            h_t += (A ** (t - i)) * B * u[i]
        h.append(h_t)
    return h

def efficient_parallel_ssm_scan(u, A, B):
    """Efficient parallel SSM scan - O(log N) time with N processors"""
    n = len(u)
    if n == 1:
        return [B * u[0]]
    
    # Step 1: Create pairs with SSM operations
    pairs = []
    pair_As = []  # Track A values for each pair
    
    for i in range(0, n, 2):
        if i + 1 < n:
            # Combine two consecutive SSM operations:
            # h[i] = B * u[i]
            # h[i+1] = A * h[i] + B * u[i+1] = A * B * u[i] + B * u[i+1]
            pair_sum = A * B * u[i] + B * u[i+1]
            pairs.append(pair_sum)
            pair_As.append(A * A)  # A^2 for the combined operation
        else:
            pairs.append(B * u[i])
            pair_As.append(A)
    
    # Step 2: Recursively solve smaller problem
    if len(pairs) > 1:
        # For simplicity, use A^2 as the new A parameter
        pair_results = efficient_parallel_ssm_scan(pairs, A*A, 1.0)
    else:
        pair_results = pairs
    
    # Step 3: Reconstruct full result
    result = [0] * n
    result[0] = B * u[0]
    
    for i in range(1, n):
        if i % 2 == 1:  # Odd positions
            result[i] = pair_results[i // 2]
        else:  # Even positions
            result[i] = A * pair_results[(i-1) // 2] + B * u[i]
    
    return result

def ssm_scan_demo():
    """Demonstrate all three SSM scan methods"""
    A, B = 0.9, 1.0
    u = [1, 2, 1, 3]
    
    print(f"SSM: h[t] = {A} * h[t-1] + {B} * u[t]")
    print(f"Input: {u}")
    print()
    
    # Sequential
    print("Sequential O(N):")
    h_seq = sequential_ssm_scan(u, A, B)
    print(f"Result: {[round(x, 3) for x in h_seq]}")
    print()
    
    # Naive parallel (mathematical equivalence)
    print("Naive Parallel O(N²):")
    h_naive = naive_parallel_ssm_scan(u, A, B)
    print(f"Result: {[round(x, 3) for x in h_naive]}")
    print()
    
    # Efficient parallel
    print("Efficient Parallel O(log N):")
    h_efficient = efficient_parallel_ssm_scan(u, A, B)
    print(f"Result: {[round(x, 3) for x in h_efficient]}")
    print()
    
    # Verify all methods give same result
    print("Verification:")
    print(f"Sequential == Naive: {np.allclose(h_seq, h_naive)}")
    print(f"Sequential == Efficient: {np.allclose(h_seq, h_efficient)}")

if __name__ == "__main__":
    ssm_scan_demo()