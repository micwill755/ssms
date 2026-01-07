from typing import List, Tuple

'''
| Direction        | Math           | Meaning                | Algorithm            |
| ---------------- | -------------- | ---------------------- | -------------------- |
| coeffs → samples | (y = V A)      | Evaluate polynomial    | Matrix multiply      |
| samples → coeffs | (A = V^{-1} y) | Interpolate polynomial | Gaussian elimination |

Gaussian elimination = brute-force inverse transform.

'''

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination.
    A: list of lists (n x n)
    b: list (n)
    Returns x: list (n)
    """

    n = len(A)

    # ---- Make augmented matrix [A | b] ----
    M = [A[i][:] + [b[i]] for i in range(n)]

    # ---- Forward elimination ----
    for pivot in range(n):

        # 1. Find pivot (simple version: assume non-zero)
        if abs(M[pivot][pivot]) < 1e-12:
            raise ValueError("Zero pivot encountered!")

        # 2. Eliminate rows below
        for row in range(pivot + 1, n):
            factor = M[row][pivot] / M[pivot][pivot]

            for col in range(pivot, n + 1):
                M[row][col] -= factor * M[pivot][col]

    # ---- Back substitution ----
    x = [0.0] * n

    for i in reversed(range(n)):
        s = M[i][n]   # RHS

        for j in range(i + 1, n):
            s -= M[i][j] * x[j]

        x[i] = s / M[i][i]

    return x

def gaussian_elimination_pivot(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for pivot in range(n):

        # ---- Find best pivot row ----
        max_row = pivot
        for r in range(pivot + 1, n):
            if abs(M[r][pivot]) > abs(M[max_row][pivot]):
                max_row = r

        # Swap rows
        M[pivot], M[max_row] = M[max_row], M[pivot]

        if abs(M[pivot][pivot]) < 1e-12:
            raise ValueError("Matrix is singular!")

        # ---- Eliminate below ----
        for row in range(pivot + 1, n):
            factor = M[row][pivot] / M[pivot][pivot]
            for col in range(pivot, n + 1):
                M[row][col] -= factor * M[pivot][col]

    # ---- Back substitution ----
    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x

'''

Newton Interpolation (O(n²))

Instead of solving a matrix:

Compute divided differences
Build polynomial incrementally
Convert basis to monomials

Complexity: O(n²)
Stability: better than Vandermonde
Still too slow for large n

Used in:

Numerical analysis
Small interpolation problems

'''
def newton_divided_differences(x: List[float], y: List[float]) -> List[float]:
    """
    Compute Newton-form coefficients c[i] using divided differences.
    Inputs:
      x: distinct sample locations [x0, x1, ..., x_{n-1}]
      y: sample values            [y0, y1, ..., y_{n-1}]
    Output:
      c: Newton coefficients such that:
         p(x) = c0 + c1(x-x0) + c2(x-x0)(x-x1) + ...
    Time: O(n^2), Space: O(n)
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n == 0:
        return []
    # Check distinct x's (basic check)
    if len(set(x)) != n:
        raise ValueError("x values must be distinct for interpolation")

    # Copy y into a work array for in-place divided difference updates
    dd = [float(v) for v in y]
    c = [0.0] * n
    c[0] = dd[0]

    # dd[i] will hold the current column of divided differences
    for k in range(1, n):
        for i in range(n - 1, k - 1, -1):
            denom = x[i] - x[i - k]
            if abs(denom) < 1e-15:
                raise ValueError("Nearly duplicate x values encountered")
            dd[i] = (dd[i] - dd[i - 1]) / denom
        c[k] = dd[k]

    return c

A = [
    [1, 0, 0],
    [1, 1, 1],
    [1, 2, 4],
]

b = [2, 6, 12]

x = gaussian_elimination(A, b)
print(x)