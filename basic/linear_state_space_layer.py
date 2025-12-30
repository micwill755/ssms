import numpy as np
from scipy import linalg

class LinearStateSpaceLayer:
    """
    Linear State Space Layer showing two computation modes:
    1. Recurrent: x(t+1) = A*x(t) + B*u(t) - for inference
    2. Convolution: y = conv(u, K) - for training (parallel)
    """
    
    def __init__(self, input_dim, state_dim, output_dim, seq_len=100):
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # SSM matrices
        self.A = np.random.randn(state_dim, state_dim) * 0.1
        self.B = np.random.randn(state_dim, input_dim) * 0.1
        self.C = np.random.randn(output_dim, state_dim) * 0.1
        
        # Pre-compute convolution kernel for training
        self.K = self._compute_convolution_kernel()
    
    def _compute_convolution_kernel(self):
        """
        Compute convolution kernel K where:
        K[i] = C @ A^i @ B
        
        This allows: y = conv(u, K) instead of recurrent computation
        """
        K = []
        A_power = np.eye(self.state_dim)  # A^0 = I
        
        for i in range(self.seq_len):
            # K[i] = C @ A^i @ B
            K_i = self.C @ A_power @ self.B
            K.append(K_i)
            
            # Update A^i for next iteration
            A_power = A_power @ self.A
        
        return np.array(K)
    
    def forward_recurrent(self, inputs):
        """
        RECURRENT MODE - for inference
        Process step by step: x(t+1) = A*x(t) + B*u(t)
        """
        print("=== RECURRENT MODE (Inference) ===")
        seq_len = len(inputs)
        
        x = np.zeros(self.state_dim)
        outputs = []
        
        for t in range(seq_len):
            u_t = inputs[t]
            
            # Recurrent update
            x = self.A @ x + self.B @ u_t
            y = self.C @ x
            
            outputs.append(y)
            
            if t < 3:  # Show first few steps
                print(f"t={t}: u={u_t} -> x={x[:2]}... -> y={y}")
        
        return np.array(outputs)
    
    def forward_convolution(self, inputs):
        """
        CONVOLUTION MODE - for training
        Parallel computation: y = conv(u, K)
        """
        print("\n=== CONVOLUTION MODE (Training) ===")
        seq_len = len(inputs)
        
        # Pad inputs for convolution
        padded_inputs = np.pad(inputs, ((self.seq_len-1, 0), (0, 0)), mode='constant')
        
        outputs = []
        for t in range(seq_len):
            # Convolution: sum over all previous inputs weighted by kernel
            y = np.zeros(self.C.shape[0])
            
            for i in range(min(t+1, self.seq_len)):
                if t-i >= 0:
                    # y += K[i] @ u[t-i]
                    y += self.K[i] @ padded_inputs[t-i + self.seq_len-1]
            
            outputs.append(y)
            
            if t < 3:  # Show first few steps
                print(f"t={t}: conv_sum -> y={y}")
        
        return np.array(outputs)
    
    def compare_modes(self, inputs):
        """Compare both computation modes"""
        print("Input sequence shape:", inputs.shape)
        
        # Run both modes
        recurrent_out = self.forward_recurrent(inputs)
        conv_out = self.forward_convolution(inputs)
        
        # Check if they're the same
        diff = np.abs(recurrent_out - conv_out).max()
        print(f"\nMax difference between modes: {diff:.10f}")
        print("✓ Same results!" if diff < 1e-10 else "✗ Different results!")
        
        return recurrent_out, conv_out

def demo_dual_computation():
    """Show the dual computation in action"""
    print("=== Linear State Space Layer: Dual Computation ===\n")
    
    # Create layer
    layer = LinearStateSpaceLayer(input_dim=2, state_dim=4, output_dim=1, seq_len=10)
    
    # Test sequence
    inputs = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [-0.5, 0.5],
        [0.0, 0.0]
    ])
    
    # Compare both modes
    rec_out, conv_out = layer.compare_modes(inputs)
    
    print(f"\nConvolution kernel shape: {layer.K.shape}")
    print(f"First few kernel values:\n{layer.K[:3, 0, :]}")

def explain_why_this_works():
    """Explain the mathematical equivalence"""
    print("\n" + "="*60)
    print("WHY CONVOLUTION = RECURRENT")
    print("="*60)
    
    print("\nRECURRENT:")
    print("x(0) = 0")
    print("x(1) = A*x(0) + B*u(0) = B*u(0)")
    print("x(2) = A*x(1) + B*u(1) = A*B*u(0) + B*u(1)")
    print("x(3) = A*x(2) + B*u(2) = A²*B*u(0) + A*B*u(1) + B*u(2)")
    print("...")
    print("y(t) = C*x(t)")
    
    print("\nCONVOLUTION:")
    print("y(t) = C*B*u(t) + C*A*B*u(t-1) + C*A²*B*u(t-2) + ...")
    print("     = K[0]*u(t) + K[1]*u(t-1) + K[2]*u(t-2) + ...")
    print("     = conv(u, K)")
    
    print("\nWhere K[i] = C @ A^i @ B")
    
    print("\n🚀 BENEFITS:")
    print("• Recurrent: Memory efficient, good for inference")
    print("• Convolution: Parallel, fast training with GPUs")

if __name__ == "__main__":
    demo_dual_computation()
    explain_why_this_works()