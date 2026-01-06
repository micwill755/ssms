Here's the evolution:

1. DFT (O(N²)) - Too slow

Brute force frequency analysis
Test every frequency against every sample
Impractical for long sequences

2. FFT (O(N log N)) - Fast enough for training

Clever divide-and-conquer algorithm
Same results as DFT, much faster
Enabled parallel SSM training (S4 breakthrough)

3. Mamba's Scan Operation (O(N)) - Even faster for inference

Key insight: FFT is great for training, but inference can be even faster
Selective SSMs: A, B, C matrices depend on input (context-aware)
Parallel scan: Computes recurrent operations in O(N) parallel time

The progression:

Basic SSM: Sequential O(N) per step → O(N²) total
S4 + FFT:  Parallel training O(N log N) 
Mamba:     Parallel scan O(N) + selective (input-dependent)

Why this sequence matters:

DFT/FFT: Unlocked parallel training (compete with Transformers)
Scan operation: Unlocked efficient inference (better than RNNs)
Selective mechanism: Added context-awareness (like attention)

Mamba's innovation: Not just faster computation, but selective SSMs that can focus on relevant parts of the sequence, combining the best of:

RNN efficiency (linear memory)
Transformer parallelism (training speed)
Attention selectivity (context awareness)

## Key Distinction: FFT vs Scan

**S4 (2021): Uses FFT for training**
- Fixed A, B, C matrices → can precompute convolution kernel
- Training: FFT convolution O(N log N)
- Inference: Recurrent form O(N)

**Mamba (2023): Does NOT use FFT**
- **Selective SSMs**: A, B, C depend on input → can't precompute kernel
- **Training**: Parallel scan O(N) 
- **Inference**: Parallel scan O(N)

**Why Mamba can't use FFT:**
```python
# S4: Fixed matrices (can use FFT)
A = fixed_matrix  # Same for all inputs
B = fixed_matrix
C = fixed_matrix

# Mamba: Input-dependent matrices (can't use FFT)
A = f(input)  # Changes based on input!
B = g(input)  # Changes based on input!
C = h(input)  # Changes based on input!
```

**The complete evolution:**
1. **Basic SSM**: Sequential O(N²)
2. **S4**: FFT training O(N log N) + recurrent inference O(N)
3. **Mamba**: Parallel scan for both training AND inference O(N)

**Key insight**: Mamba **gave up** the FFT speedup to gain **selectivity**. The parallel scan algorithm compensates by being even faster (O(N) vs O(N log N)) and works for both training and inference.

So FFT was crucial for S4, but Mamba moved beyond FFT to achieve input-dependent (selective) behavior!

The Evolution Timeline:

Basic SSM → Sequential, too slow
S4 → FFT for training, recurrent for inference
Mamba → Parallel scan for both, no FFT needed

The Trade-off:

S4: Fast training (FFT) but fixed behavior
Mamba: Slightly different approach (scan) but selective behavior