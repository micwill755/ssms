# Naive evaluation (O(n·m))
def eval_poly_naive(coeffs, x):
    y = []
    for xi in x:
        s = 0.0
        power = 1.0
        for a in coeffs:
            s += a * power
            power *= xi
        y.append(s)
    return y

# Horner’s method (same big-O, fewer ops)
def eval_poly_horner(coeffs, x):
    y = []
    for xi in x:
        acc = 0.0
        for a in reversed(coeffs):
            acc = acc * xi + a
        y.append(acc)
    return y