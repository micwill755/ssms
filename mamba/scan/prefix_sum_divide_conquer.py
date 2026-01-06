import numpy as np

def sequential_prefix_sum(x):
    """Basic scan example"""    
    result = []
    acc = 0
    for val in x:
        acc += val
        result.append(acc)
    
    print(f"Scan: {result}")  # [1, 3, 6, 10]

def parallel_prefix_sum(x):
    """Parallel prefix sum - O(log n) time with n processors"""
    n = len(x)
    if n == 1:
        return x
    
    # Step 1: Divide - compute pairwise sums in parallel
    # [1, 2, 3, 4] → [1+2, 3+4] = [3, 7]
    pairs = []
    for i in range(0, n, 2):
        if i + 1 < n:
            pairs.append(x[i] + x[i+1])  # Can run in parallel!
        else:
            pairs.append(x[i])
    
    # Step 2: Conquer - recursively solve smaller problem
    if len(pairs) > 1:
        pair_prefix = parallel_prefix_sum(pairs)
    else:
        pair_prefix = pairs
    
    # Step 3: Combine - expand back to original size
    result = [0] * n
    result[0] = x[0]  # First element is always x[0]
    
    for i in range(1, n):
        if i % 2 == 1:  # Odd indices: use pair result directly
            result[i] = pair_prefix[i // 2]
        else:  # Even indices: previous pair result + current element
            result[i] = pair_prefix[(i-1) // 2] + x[i]
    
    return result

if __name__ == "__main__":
    x = [1, 2, 3, 4]
    print(f"Input:  {x}")
    res = parallel_prefix_sum(x)
    print(f"Result:  {res}")