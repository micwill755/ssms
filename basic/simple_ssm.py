import torch
import torch.nn as nn
import numpy as np

class SimpleSSM(nn.Module):
    """
    Basic State Space Model implementation
    
    The core SSM equation:
    x(t+1) = A * x(t) + B * u(t)  # State update
    y(t) = C * x(t) + D * u(t)    # Output
    """
    
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # SSM matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
    def forward(self, u):
        """
        u: input sequence [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = u.shape
        
        # Initialize state
        x = torch.zeros(batch_size, self.state_dim, device=u.device)
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            # State update: x(t+1) = A * x(t) + B * u(t)
            x = torch.matmul(x, self.A.T) + torch.matmul(u[:, t], self.B.T)
            
            # Output: y(t) = C * x(t) + D * u(t)
            y = torch.matmul(x, self.C.T) + torch.matmul(u[:, t], self.D.T)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

# Test the SSM
if __name__ == "__main__":
    # Create model
    model = SimpleSSM(input_dim=4, state_dim=8, output_dim=2)
    
    # Test input: batch_size=2, seq_len=10, input_dim=4
    x = torch.randn(2, 10, 4)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"SSM matrices shapes:")
    print(f"A: {model.A.shape}, B: {model.B.shape}")
    print(f"C: {model.C.shape}, D: {model.D.shape}")