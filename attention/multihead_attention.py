import numpy as np
import math

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Weight matrices for Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def scaled_dot_product_attention(self, Q, K, V):
        """Standard quadratic attention: O(n²) complexity"""
        # Q, K, V shape: (batch, heads, seq_len, d_k)
        
        # Step 1: Compute attention scores QK^T
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))  # (batch, heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_k)
        
        # Step 2: Apply softmax
        attention_weights = self.softmax(scores)  # Still (batch, heads, seq_len, seq_len)
        
        # Step 3: Apply attention to values
        output = np.matmul(attention_weights, V)  # (batch, heads, seq_len, d_k)
        
        return output, attention_weights
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.matmul(x, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        
        return output, attn_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def demo_multihead_attention():
    # Setup
    batch_size, seq_len, d_model = 2, 8, 64
    num_heads = 8
    
    # Create random input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Initialize attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attention_weights = mha.forward(x)
    
    print("=== Multihead Attention Demo ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Complexity: O(n²) where n={seq_len}")
    print(f"Attention matrix size: {seq_len}×{seq_len} = {seq_len**2} elements per head")
    
    # Show attention pattern for first head
    print(f"\nAttention pattern (first head, first batch):")
    print(attention_weights[0, 0])

if __name__ == "__main__":
    demo_multihead_attention()