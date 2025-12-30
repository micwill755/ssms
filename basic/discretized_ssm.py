import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretizedSSM(nn.Module):
    """
    Discretized State Space Model
    
    Converts continuous SSM to discrete form:
    A_discrete = exp(A * dt)
    B_discrete = A^(-1) * (A_discrete - I) * B
    """
    
    def __init__(self, input_dim, state_dim, output_dim, dt=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        
        # Continuous matrices (learnable)
        self.A_continuous = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B_continuous = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
    def discretize(self):
        """Convert continuous matrices to discrete"""
        # A_discrete = exp(A * dt)
        A_discrete = torch.matrix_exp(self.A_continuous * self.dt)
        
        # B_discrete = A^(-1) * (A_discrete - I) * B
        I = torch.eye(self.state_dim, device=self.A_continuous.device)
        A_inv = torch.inverse(self.A_continuous + 1e-6 * I)  # Add small epsilon for stability
        B_discrete = A_inv @ (A_discrete - I) @ self.B_continuous
        
        return A_discrete, B_discrete
    
    def forward(self, u):
        """Forward pass with discretized matrices"""
        batch_size, seq_len, _ = u.shape
        
        # Get discretized matrices
        A_d, B_d = self.discretize()
        
        # Initialize state
        x = torch.zeros(batch_size, self.state_dim, device=u.device)
        outputs = []
        
        for t in range(seq_len):
            # Discrete state update
            x = torch.matmul(x, A_d.T) + torch.matmul(u[:, t], B_d.T)
            
            # Output
            y = torch.matmul(x, self.C.T) + torch.matmul(u[:, t], self.D.T)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

# Example usage
if __name__ == "__main__":
    model = DiscretizedSSM(input_dim=3, state_dim=6, output_dim=2, dt=0.1)
    
    # Test sequence
    x = torch.randn(1, 20, 3)
    output = model(x)
    
    print(f"Discretized SSM output shape: {output.shape}")
    
    # Show the difference between continuous and discrete matrices
    A_d, B_d = model.discretize()
    print(f"Continuous A norm: {model.A_continuous.norm():.4f}")
    print(f"Discrete A norm: {A_d.norm():.4f}")