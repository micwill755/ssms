What are Eigenvalues?

Simple Definition:
Eigenvalues tell you how a matrix "stretches" or "shrinks" vectors. Think of them as the "strength" or "power" of a matrix.

The Equation:
A * v = λ * v

Where:
A = matrix
v = eigenvector (special direction)
λ = eigenvalue (how much stretching)

For any n×n matrix, there are exactly n eigenvalues, eg. a 2×2 matrix: exactly 2 eigenvalues

Diagonal vs Non-Diagonal:

1. Diagonal matrices

Simple Example
A = [[2, 0],
     [0, 3]]

Eigenvalues: Just read the diagonal: [2, 3]
Eigenvectors: Just the coordinate axes: [1,0] and [0,1]

2. Non-Diagonal Matrix

A = [[2, 4],
     [5, 3]]

For non-diagonal matrices, we need to solve:
det(A - λI) = 0

e.g. For A = [[2, 4], [5, 3]]:

A - λI = [[2-λ, 4],
          [5, 3-λ]]

det = (2-λ)(3-λ) - 4×5
    = 6 - 2λ - 3λ + λ² - 20
    = λ² - 5λ - 14

Solve: λ² - 5λ - 14 = 0
Answer: λ = 7 or λ = -2

Eigenvalues: [7, -2]
For each eigenvalue λ, we solve: (A - λI)v = 0

Step 1: Find Eigenvector for λ = 7

a. Set up the equation:

(A - 7I)v = 0

A - 7I = [[2-7, 4],  = [[-5, 4],
          [5, 3-7]]    [ 5, -4]]

b. Solve [-5, 4; 5, -4] * [x, y] = [0, 0]:

-5x + 4y = 0  →  4y = 5x  →  y = (5/4)x
 5x - 4y = 0  →  same equation

c. Choose x = 4, then y = 5:

Eigenvector for λ = 7: [4, 5]

Step 2: Find Eigenvector for λ = -2

a. Set up the equation:

(A - (-2)I)v = 0

A + 2I = [[2+2, 4  ],  = [[4, 4],
          [5,   3+2]]    [5, 5]]

b. Solve [4, 4; 5, 5] * [x, y] = [0, 0]:

4x + 4y = 0  →  4y = -4x  →  y = -x
5x + 5y = 0  →  same equation

c. Choose x = 1, then y = -1:
Eigenvector for λ = -2: [1, -1]

Step 3 Verify Our Answer

a. Check λ = 7, v = [4, 5]:

i. A * v = [[2, 4],  *  [4]  = [2×4 + 4×5]  = [28]
        [5, 3]]     [5]    [5×4 + 3×5]    [35]

ii. λ * v = 7 * [4] = [28]  ✓ Match!
            [5]   [35]

b. Check λ = -2, v = [1, -1]:

i. A * v = [[2, 4],  *  [1 ]  = [2×1 + 4×(-1)]  = [-2]
        [5, 3]]     [-1]    [5×1 + 3×(-1)]    [ 2]

ii. λ * v = -2 * [1 ] = [-2]  ✓ Match!
             [-1]   [ 2]

Why Eigenvalues Matter

Eigenvalues control the "behavior" of neural networks over time.

1. Gradient Flow (Training Stability)
The Problem:
In deep learning, gradients flow backward through layers. If eigenvalues are bad, gradients explode or vanish.

Eigenvalue Effects:

|λ| > 1: Gradients explode (training becomes unstable)
|λ| < 1: Gradients vanish (network can't learn)
|λ| ≈ 1: Good gradient flow (stable training)

Example:

# Bad eigenvalues
A = [[3, 0], [0, 0.1]]  # λ = [3, 0.1]
# Problem: 3 > 1 (exploding), 0.1 << 1 (vanishing)

# Good eigenvalues (HiPPO style)
A = [[-0.5, 0], [0, -0.8]]  # λ = [-0.5, -0.8]
# Good: All |λ| < 1 and not too small

2. Memory and Information Flow
In SSMs: x(t+1) = A*x(t) + B*u(t)

Eigenvalues determine how information persists:
Large |λ|: Information persists longer (good memory)
Small |λ|: Information fades quickly (poor memory)
Negative λ: Controlled oscillation (HiPPO uses this)

# After 5 time steps, how much of original info remains?
λ = 0.9  →  0.9^5 = 0.59  (59% remains - good memory)
λ = 0.1  →  0.1^5 = 0.00001  (nothing remains - poor memory)
λ = -0.8 →  (-0.8)^5 = -0.33  (33% remains with sign flip - HiPPO style)

3. Long-Range Dependencies
The Challenge:
Neural networks struggle to remember things from long ago (like the beginning of a long sentence).

Eigenvalue Solution:
Random eigenvalues: Unpredictable memory decay
HiPPO eigenvalues: Optimal memory decay curve

Time:     0    1    2    3    4    5
Random:   1 → 0.3 → 0.8 → 0.1 → 2.1 → 0.05  (chaotic)
HiPPO:    1 → 0.8 → 0.6 → 0.5 → 0.4 → 0.3   (smooth decay)

4. Why HiPPO Works

# HiPPO eigenvalues
λ = [-0.5, -1.0, -1.5, -2.0, ...]

Transformer Architecture 

Token Embeddings:
# Each token gets an embedding vector
"hello" → [0.2, -0.1, 0.5, 0.8]  # 4-dimensional embedding
"world" → [0.1, 0.3, -0.2, 0.4]  # These ARE weights (learnable)

These embedding values are indeed trainable weights, but they don't have eigenvalues because they're just vectors, not matrices.

NOTE: Vectors Don't Have Eigenvalues, Only square matrices have eigenvalues.

# This is a vector (1D) - NO eigenvalues
embedding = [0.2, -0.1, 0.5, 0.8]

# This is a matrix (2D) - HAS eigenvalues  
W = [[0.1, 0.2],
     [0.3, 0.4]]

Why? The eigenvalue equation is A * v = λ * v

You need a matrix A to multiply a vector v
A vector can't multiply another vector in this way

Where Eigenvalues Actually Matter in Transformers

1. Attention Weight Matrices:

# In attention mechanism, you have weight matrices:
W_q = [[0.1, 0.2],   # Query weight matrix (2x2)
       [0.3, 0.4]]

W_k = [[0.5, 0.1],   # Key weight matrix (2x2)  
       [0.2, 0.6]]

W_v = [[0.3, 0.7],   # Value weight matrix (2x2)
       [0.1, 0.2]]


Each of these matrices HAS eigenvalues:

W_q eigenvalues: [0.5, 0.0] (example)
W_k eigenvalues: [0.7, 0.4] (example)
W_v eigenvalues: [0.4, 0.1] (example)

2. Feed-Forward Network Matrices:

# FFN has weight matrices too
W1 = [[0.2, 0.1, 0.3],   # First layer weights (3x3)
      [0.4, 0.2, 0.1],
      [0.1, 0.5, 0.2]]

W2 = [[0.3, 0.2, 0.4],   # Second layer weights (3x3)
      [0.1, 0.6, 0.2],
      [0.2, 0.1, 0.3]]

These matrices also have eigenvalues that affect gradient flow.

# Multiple different matrices, each with their own eigenvalues
attention_matrix = W_q @ W_k^T  # Changes for each input!
ffn_matrix1 = W1
ffn_matrix2 = W2
# Eigenvalues are different for each matrix and each input

SSM Architecture 

Why SSMs Are Simpler

Transformer problem:
- Many matrices with unpredictable eigenvalues
- Eigenvalues change with each input (attention is dynamic)
- Hard to control gradient flow

SSM solution:

- One main matrix A with controlled eigenvalues
- Eigenvalues are fixed (don't change with input)
- Easy to ensure stable training

# ONE main matrix A that controls everything
A = [[-0.5, 0,    0   ],   # This matrix's eigenvalues control
     [0,   -1.0,  0   ],   # the entire sequence processing
     [0,    0,   -1.5]]
# Eigenvalues: [-0.5, -1.0, -1.5] - fixed and controlled
