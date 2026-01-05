import numpy as np
import math

class LinearMultiHeadAttention:
    def __init__(self, d_model, num_heads, feature_dim=None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_dim = feature_dim or self.d_k
        
        # Same weight matrices as standard attention
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    # if we want to use ReLU instead of elu
    '''def feature_map(self, x):
        """Feature map φ(x) = ReLU(x) + 1"""
        return np.maximum(0, x) + 1  # ReLU + 1 for simplicity'''

    def feature_map(self, x, alpha=1.0):
        """Feature map φ(x) = elu(x) + 1"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1)) + 1

    def linear_attention(self, Q, K, V):
        """Linear attention: O(n) complexity"""
        # Q, K, V shape: (batch, heads, seq_len, d_k)
        
        # Step 1: Apply feature map
        Q_phi = self.feature_map(Q)  # (batch, heads, seq_len, d_k)
        K_phi = self.feature_map(K)  # (batch, heads, seq_len, d_k)
        
        # Step 2: Compute K^T V first (key insight!)
        # K_phi^T: (batch, heads, d_k, seq_len)
        # V: (batch, heads, seq_len, d_k)
        # Result: (batch, heads, d_k, d_k) - much smaller than (seq_len, seq_len)!
        KV = np.matmul(K_phi.transpose(0, 1, 3, 2), V)
        
        # Step 3: Compute Q (K^T V)
        # Q_phi: (batch, heads, seq_len, d_k)
        # KV: (batch, heads, d_k, d_k)
        # Result: (batch, heads, seq_len, d_k)
        output = np.matmul(Q_phi, KV)
        
        # Step 4: Normalization (approximate softmax normalization)
        # Compute sum of keys for each query
        K_sum = np.sum(K_phi, axis=2, keepdims=True)  # (batch, heads, 1, d_k)
        normalizer = np.matmul(Q_phi, K_sum.transpose(0, 1, 3, 2))  # (batch, heads, seq_len, 1)
        normalizer = np.maximum(normalizer, 1e-6)  # Avoid division by zero
        
        output = output / normalizer
        
        return output
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections (same as standard attention)
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply linear attention
        attn_output = self.linear_attention(Q, K, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        
        return output

def compare_attention_mechanisms():
    """Compare standard vs linear attention"""
    from multihead_attention import MultiHeadAttention
    
    # Setup
    batch_size, seq_len, d_model = 2, 1000, 64  # Large sequence length
    num_heads = 8
    
    # Create same input for both
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Standard attention
    print("=== Standard Multihead Attention ===")
    mha_standard = MultiHeadAttention(d_model, num_heads)
    output_standard, attn_weights = mha_standard.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_standard.shape}")
    print(f"Attention matrix size: {seq_len}×{seq_len} = {seq_len**2:,} elements per head")
    print(f"Memory complexity: O(n²) = O({seq_len**2:,})")
    
    # Linear attention
    print("\n=== Linear Multihead Attention ===")
    mha_linear = LinearMultiHeadAttention(d_model, num_heads)
    output_linear = mha_linear.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_linear.shape}")
    print(f"KV matrix size: {mha_linear.d_k}×{mha_linear.d_k} = {mha_linear.d_k**2} elements per head")
    print(f"Memory complexity: O(n) = O({seq_len})")
    
    # Show complexity difference
    print(f"\n=== Complexity Comparison ===")
    print(f"Standard attention: {seq_len**2:,} operations")
    print(f"Linear attention: ~{seq_len * mha_linear.d_k:,} operations")
    print(f"Speedup factor: ~{(seq_len**2) / (seq_len * mha_linear.d_k):.1f}x")
    
    # Show how linear attention avoids the n×n matrix
    print(f"\n=== Key Insight ===")
    print("Standard: Q(n×d) @ K^T(d×n) = (n×n) matrix <- EXPENSIVE!")
    print("Linear:   φ(Q)(n×d) @ [φ(K)^T(d×n) @ V(n×d)] = φ(Q) @ (d×d) <- CHEAP!")
    print("Never materialize the (n×n) attention matrix!")

def demo_recurrent_form():
    """Show how linear attention can be computed recurrently"""
    print("\n=== Linear Attention as Recurrent Computation ===")
    
    seq_len, d_k = 5, 4
    
    # Sample data
    np.random.seed(42)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    Q = np.random.randn(seq_len, d_k)
    
    # Apply feature map
    K_phi = np.maximum(0, K) + 1
    V_phi = V
    Q_phi = np.maximum(0, Q) + 1
    
    print("Recurrent computation (like an SSM):")
    
    # Initialize state
    state = np.zeros((d_k, d_k))
    
    for t in range(seq_len):
        # Update state: h_t = h_{t-1} + k_t ⊗ v_t
        state = state + np.outer(K_phi[t], V_phi[t])
        
        # Compute output: o_t = q_t^T h_t
        output_t = np.dot(Q_phi[t], state)
        
        print(f"Step {t}: state shape {state.shape}, output shape {output_t.shape}")
    
    print("This is exactly how SSMs work: accumulate information in a state!")

if __name__ == "__main__":
    compare_attention_mechanisms()
    demo_recurrent_form()