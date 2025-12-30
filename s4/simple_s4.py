import numpy as np
from scipy import linalg

class SimpleS4:
    """
    Simplified S4 (Structured State Spaces) implementation
    Key features:
    - HiPPO-inspired initialization
    - Diagonal A matrix structure
    - Dual computation modes
    """
    
    def __init__(self, input_dim, state_dim, output_dim, seq_len=100):
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # S4: Use diagonal A matrix with HiPPO-inspired initialization
        self.A = self._hippo_initialization(state_dim)
        self.B = np.random.randn(state_dim, input_dim) * 0.1
        self.C = np.random.randn(output_dim, state_dim) * 0.1
        
        # Pre-compute convolution kernel
        self.K = self._compute_convolution_kernel()
        
        print(f"S4 Model: state_dim={state_dim}, diagonal A matrix")
    
    def _hippo_initialization(self, N):
        """
        Simplified HiPPO initialization for matrix A
        Real S4 uses more complex HiPPO theory, this is a simplified version
        """
        # Create diagonal matrix with negative eigenvalues
        # This ensures stability and good memory properties
        eigenvalues = -np.arange(1, N+1, dtype=np.float32)
        A = np.diag(eigenvalues)
        
        print(f"HiPPO A eigenvalues: {eigenvalues[:5]}...")
        return A
    
    def _compute_convolution_kernel(self):
        """Compute convolution kernel K[i] = C @ A^i @ B"""
        K = []
        A_power = np.eye(self.state_dim)
        
        for i in range(self.seq_len):
            K_i = self.C @ A_power @ self.B
            K.append(K_i)
            
            # Since A is diagonal, A^i is easy to compute
            A_power = A_power @ self.A
        
        return np.array(K)
    
    def forward_recurrent(self, inputs):
        """Recurrent mode: x(t+1) = A*x(t) + B*u(t)"""
        seq_len = len(inputs)
        x = np.zeros(self.state_dim)
        outputs = []
        
        for t in range(seq_len):
            u_t = inputs[t]
            x = self.A @ x + self.B @ u_t
            y = self.C @ x
            outputs.append(y)
        
        return np.array(outputs)
    
    def forward_convolution(self, inputs):
        """Convolution mode: y = conv(u, K)"""
        seq_len = len(inputs)
        outputs = []
        
        for t in range(seq_len):
            y = np.zeros(self.C.shape[0])
            
            for i in range(min(t+1, len(self.K))):
                if t-i >= 0:
                    y += self.K[i] @ inputs[t-i]
            
            outputs.append(y)
        
        return np.array(outputs)

def compare_basic_vs_s4():
    """Compare basic SSM vs S4"""
    print("=== Basic SSM vs S4 Comparison ===\n")
    
    # Basic SSM (random A)
    state_dim = 6
    A_basic = np.random.randn(state_dim, state_dim) * 0.1
    
    # S4 (diagonal A with HiPPO)
    s4 = SimpleS4(input_dim=2, state_dim=state_dim, output_dim=1)
    A_s4 = s4.A
    
    print("Basic SSM matrix A:")
    print(A_basic)
    print(f"Eigenvalues: {np.linalg.eigvals(A_basic)}")
    
    print("\nS4 matrix A (diagonal):")
    print(A_s4)
    print(f"Eigenvalues: {np.diag(A_s4)}")
    
    print("\nKey differences:")
    print("• Basic: Dense random matrix, unstable eigenvalues")
    print("• S4: Diagonal matrix, stable negative eigenvalues")
    print("• S4: Easier to compute A^k (just diagonal^k)")
    print("• S4: Better gradient flow for long sequences")

def demo_s4():
    """Demonstrate S4 in action"""
    print("\n=== S4 Demo ===\n")
    
    s4 = SimpleS4(input_dim=2, state_dim=4, output_dim=1, seq_len=10)
    
    # Test sequence
    inputs = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [-0.5, 0.5],
        [0.0, 0.0]
    ])
    
    # Compare modes
    rec_out = s4.forward_recurrent(inputs)
    conv_out = s4.forward_convolution(inputs)
    
    print("Input sequence:")
    for i, inp in enumerate(inputs):
        print(f"  t={i}: {inp}")
    
    print(f"\nRecurrent output shape: {rec_out.shape}")
    print(f"Convolution output shape: {conv_out.shape}")
    
    diff = np.abs(rec_out - conv_out).max()
    print(f"Max difference: {diff:.10f}")
    print("✓ Same results!" if diff < 1e-10 else "✗ Different results!")

if __name__ == "__main__":
    compare_basic_vs_s4()
    demo_s4()