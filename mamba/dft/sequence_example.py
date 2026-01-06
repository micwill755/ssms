import numpy as np

def demo_text_embeddings_dft_ssm():
    """Show DFT-SSM connection with real text embeddings"""
    
    # 1. Text to embeddings (realistic 4D embeddings)
    text = "The dog jumped"
    tokens = text.split()
    
    # Simple but realistic embeddings
    embeddings = {
        "The": np.array([0.1, 0.8, -0.2, 0.5]),      # Article: low semantic, high positional
        "dog": np.array([0.9, 0.2, 0.7, -0.3]),      # Noun: high semantic content
        "jumped": np.array([0.6, -0.4, 0.8, 0.9])    # Verb: action-heavy
    }
    
    sequence = np.array([embeddings[token] for token in tokens])  # Shape: (3, 4)
    print(f"Text: '{text}'")
    print(f"Embedding shape: {sequence.shape}")
    print(f"'The': {sequence[0]}")
    print(f"'dog': {sequence[1]}")
    print(f"'jumped': {sequence[2]}")
    
    # 2. SSM matrices for 4D embeddings
    d_model = 4
    d_state = 2
    
    A = np.array([[0.8, 0.1], [0.0, 0.7]])           # 2x2 state transition
    B = np.random.randn(d_state, d_model) * 0.1       # 2x4 input projection
    C = np.random.randn(d_model, d_state) * 0.1       # 4x2 output projection
    
    print(f"\nSSM matrices: A{A.shape}, B{B.shape}, C{C.shape}")
    
    # 3. RECURRENT MODE (sequential processing)
    print("\n=== RECURRENT MODE ===")
    x = np.zeros(d_state)  # Initial state: (2,)
    outputs_recurrent = []
    
    for t, token in enumerate(tokens):
        embedding = sequence[t]  # (4,)
        x = A @ x + B @ embedding  # (2,) = (2,2)@(2,) + (2,4)@(4,)
        y = C @ x                  # (4,) = (4,2)@(2,)
        outputs_recurrent.append(y)
        print(f"'{token}': embedding{embedding.shape} -> state{x.shape} -> output{y.shape}")
        print(f"  state: [{x[0]:.3f}, {x[1]:.3f}]")
        print(f"  output: [{y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}, {y[3]:.3f}]")
    
    # 4. CONVOLUTION MODE (parallel processing)
    print("\n=== CONVOLUTION MODE ===")
    
    # Pre-compute convolution kernel K[i] = C @ A^i @ B
    seq_len = len(tokens)
    K = []
    A_power = np.eye(d_state)
    
    for i in range(seq_len):
        K_i = C @ A_power @ B  # (4,2) @ (2,2) @ (2,4) = (4,4)
        K.append(K_i)
        A_power = A_power @ A
        print(f"K[{i}] shape: {K_i.shape}")
    
    # Convolution computation
    outputs_conv = []
    for t in range(seq_len):
        y = np.zeros(d_model)
        for i in range(t + 1):
            y += K[i] @ sequence[t - i]  # (4,4) @ (4,) = (4,)
        outputs_conv.append(y)
        print(f"Output[{t}]: {[f'{val:.3f}' for val in y]}")
    
    # 5. FFT CONNECTION (frequency domain)
    print("\n=== FFT CONNECTION ===")
    
    # For each embedding dimension, apply FFT separately
    N = 8  # Pad to power of 2
    
    for dim in range(d_model):
        print(f"\nDimension {dim}:")
        
        # Extract this dimension across sequence
        u_dim = [sequence[t, dim] for t in range(seq_len)] + [0] * (N - seq_len)
        print(f"  Input sequence: {[f'{x:.2f}' for x in u_dim[:seq_len]]}")
        
        # Kernel for this output dimension (sum across input dims)
        K_dim = [np.sum(K[i][:, dim]) for i in range(seq_len)] + [0] * (N - seq_len)
        
        # FFT convolution
        U_fft = np.fft.fft(u_dim)
        K_fft = np.fft.fft(K_dim)
        Y_fft = U_fft * K_fft
        y_result = np.fft.ifft(Y_fft).real[:seq_len]
        
        print(f"  FFT result: {[f'{x:.3f}' for x in y_result]}")
    
    # 6. VERIFY EQUIVALENCE
    print(f"\n=== VERIFICATION ===")
    outputs_recurrent = np.array(outputs_recurrent)
    outputs_conv = np.array(outputs_conv)
    
    diff = np.abs(outputs_recurrent - outputs_conv).max()
    print(f"Max difference: {diff:.10f}")
    print("✓ Recurrent and Convolution identical!" if diff < 1e-10 else "✗ Methods differ!")
    
    # 7. THE EMBEDDING INSIGHT
    print(f"\n=== THE EMBEDDING INSIGHT ===")
    print("• Each embedding dimension processed independently by FFT")
    print("• SSM mixes information across dimensions through state")
    print("• FFT parallelizes the temporal convolution")
    print("• This is why S4 could handle long sequences efficiently!")
    print(f"• 'The dog jumped' -> {d_model}D embeddings -> SSM -> {d_model}D outputs")

if __name__ == "__main__":
    demo_text_embeddings_dft_ssm()
