import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """Traditional RNN for comparison"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # RNN weights
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)  # hidden-to-hidden
        self.W_ih = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)   # input-to-hidden
        self.W_ho = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)  # hidden-to-output
        
    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=u.device)
        outputs = []
        
        for t in range(seq_len):
            # RNN equation: h(t+1) = tanh(W_hh * h(t) + W_ih * u(t))
            h = torch.tanh(torch.matmul(h, self.W_hh.T) + torch.matmul(u[:, t], self.W_ih.T))
            
            # Output: y(t) = W_ho * h(t)
            y = torch.matmul(h, self.W_ho.T)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

class SimpleSSM(nn.Module):
    """SSM for comparison"""
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # SSM matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        
    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        x = torch.zeros(batch_size, self.state_dim, device=u.device)
        outputs = []
        
        for t in range(seq_len):
            # SSM equation: x(t+1) = A * x(t) + B * u(t)  [NO NONLINEARITY!]
            x = torch.matmul(x, self.A.T) + torch.matmul(u[:, t], self.B.T)
            
            # Output: y(t) = C * x(t)
            y = torch.matmul(x, self.C.T)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

def compare_rnn_ssm():
    """Show the key differences"""
    print("=== RNN vs SSM Comparison ===\n")
    
    # Same dimensions for fair comparison
    input_dim, hidden_dim, output_dim = 3, 5, 2
    
    rnn = SimpleRNN(input_dim, hidden_dim, output_dim)
    ssm = SimpleSSM(input_dim, hidden_dim, output_dim)
    
    # Test input
    x = torch.randn(1, 10, input_dim)
    
    rnn_out = rnn(x)
    ssm_out = ssm(x)
    
    print("SIMILARITIES:")
    print("✓ Both process sequences step-by-step")
    print("✓ Both maintain hidden state/memory")
    print("✓ Both have recurrent connections")
    print("✓ Both can handle variable-length sequences")
    
    print("\nKEY DIFFERENCES:")
    print("RNN: h(t+1) = tanh(W_hh * h(t) + W_ih * u(t))")
    print("SSM: x(t+1) = A * x(t) + B * u(t)")
    print()
    print("1. NONLINEARITY:")
    print("   RNN: Has tanh/relu (nonlinear)")
    print("   SSM: Pure linear operations")
    print()
    print("2. MATHEMATICAL FOUNDATION:")
    print("   RNN: Designed for ML from scratch")
    print("   SSM: Based on control theory (1960s)")
    print()
    print("3. TRAINING:")
    print("   RNN: Vanishing gradients with long sequences")
    print("   SSM: Better gradient flow (when done right)")
    print()
    print("4. EFFICIENCY:")
    print("   RNN: Sequential, hard to parallelize")
    print("   SSM: Can be parallelized (convolution view)")
    
    print(f"\nOutput shapes - RNN: {rnn_out.shape}, SSM: {ssm_out.shape}")

if __name__ == "__main__":
    compare_rnn_ssm()