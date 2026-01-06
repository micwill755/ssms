'''

The Vandermonde matrix is fundamental to the FFT because the DFT is literally a Vandermonde matrix multiplication.

For an N-point DFT, the transformation matrix is:

'''



def vandermonde(x):
    n = len(x)
    V = [[0.0] * n for _ in range(n)]
    for j in range(n):
        val = 1.0
        for k in range(n):
            V[j][k] = val
            val *= x[j]
    return V

def matvec(V, a):
    n = len(a)
    y = [0.0] * n
    for j in range(n):
        for k in range(n):
            y[j] += V[j][k] * a[k]
    return y

coeffs = [2, 3, 1]   # a0, a1, a2
x = [0, 1, 2]

V = vandermonde(x)
samples = matvec(V, coeffs)

print(samples)  # [2.0, 6.0, 12.0]