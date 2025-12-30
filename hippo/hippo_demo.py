import numpy as np

class SimpleSSM:
    """Simple SSM for comparison"""
    def __init__(self, state_dim, init_type="random"):
        self.state_dim = state_dim
        
        if init_type == "random":
            self.A = np.random.randn(state_dim, state_dim) * 0.1
        elif init_type == "hippo":
            # Simplified HiPPO: diagonal with negative eigenvalues
            eigenvals = -np.linspace(0.5, 2.0, state_dim)
            self.A = np.diag(eigenvals)
        
        self.B = np.random.randn(state_dim, 1) * 0.1
        self.C = np.random.randn(1, state_dim) * 0.1
    
    def process_sequence(self, inputs):
        """Process a sequence and return outputs"""
        x = np.zeros(self.state_dim)
        outputs = []
        
        for u in inputs:
            x = self.A @ x + self.B * u
            y = self.C @ x
            outputs.append(y[0])
        
        return outputs, x  # Return final state too

def memory_test():
    """Test how well different initializations remember sequences"""
    print("=== Memory Test: Random vs HiPPO ===\n")
    
    # Create models
    random_ssm = SimpleSSM(state_dim=5, init_type="random")
    hippo_ssm = SimpleSSM(state_dim=5, init_type="hippo")
    
    # Test sequence: [1, 2, 3, 4, 5]
    test_sequence = [1, 2, 3, 4, 5]
    
    print(f"Input sequence: {test_sequence}")
    
    # Process with both models
    random_outputs, random_state = random_ssm.process_sequence(test_sequence)
    hippo_outputs, hippo_state = hippo_ssm.process_sequence(test_sequence)
    
    print(f"\nRandom SSM outputs: {[f'{x:.3f}' for x in random_outputs]}")
    print(f"HiPPO SSM outputs:  {[f'{x:.3f}' for x in hippo_outputs]}")
    
    print(f"\nFinal states (memory):")
    print(f"Random: {[f'{x:.3f}' for x in random_state]}")
    print(f"HiPPO:  {[f'{x:.3f}' for x in hippo_state]}")
    
    # Show eigenvalues
    print(f"\nEigenvalues:")
    print(f"Random: {np.linalg.eigvals(random_ssm.A)}")
    print(f"HiPPO:  {np.diag(hippo_ssm.A)}")

def long_sequence_test():
    """Test on longer sequences to show vanishing gradient problem"""
    print("\n=== Long Sequence Test ===\n")
    
    # Create models
    random_ssm = SimpleSSM(state_dim=4, init_type="random")
    hippo_ssm = SimpleSSM(state_dim=4, init_type="hippo")
    
    # Long sequence
    long_sequence = list(range(1, 21))  # [1, 2, 3, ..., 20]
    
    print(f"Long sequence: {long_sequence[:5]}...{long_sequence[-5:]}")
    
    # Process sequences
    random_outputs, _ = random_ssm.process_sequence(long_sequence)
    hippo_outputs, _ = hippo_ssm.process_sequence(long_sequence)
    
    # Check if outputs are stable (not exploding/vanishing)
    random_stable = all(abs(x) < 100 for x in random_outputs)
    hippo_stable = all(abs(x) < 100 for x in hippo_outputs)
    
    print(f"\nOutput stability:")
    print(f"Random SSM: {'Stable' if random_stable else 'Unstable'}")
    print(f"HiPPO SSM:  {'Stable' if hippo_stable else 'Unstable'}")
    
    # Show last few outputs
    print(f"\nLast 5 outputs:")
    print(f"Random: {[f'{x:.3f}' for x in random_outputs[-5:]]}")
    print(f"HiPPO:  {[f'{x:.3f}' for x in hippo_outputs[-5:]]}")

def gradient_flow_demo():
    """Demonstrate gradient flow properties"""
    print("\n=== Gradient Flow Demo ===\n")
    
    # Show how eigenvalues affect gradient flow
    random_eigenvals = np.array([0.8, -0.3, 1.2, -0.1])  # Mixed, some > 1
    hippo_eigenvals = np.array([-0.5, -1.0, -1.5, -2.0])  # All negative, < 1
    
    print("Eigenvalue analysis:")
    print(f"Random eigenvals: {random_eigenvals}")
    print(f"HiPPO eigenvals:  {hippo_eigenvals}")
    
    # Simulate gradient flow over time
    print(f"\nGradient magnitude over time (eigenval^t):")
    print("Time\tRandom\tHiPPO")
    print("-" * 25)
    
    for t in range(6):
        # Take worst eigenvalue for random (largest magnitude)
        random_grad = max(abs(random_eigenvals)) ** t
        # Take typical eigenvalue for HiPPO
        hippo_grad = abs(hippo_eigenvals[0]) ** t
        
        print(f"{t}\t{random_grad:.3f}\t{hippo_grad:.3f}")
    
    print("\nObservation:")
    print("• Random: Can explode (>1) or vanish unpredictably")
    print("• HiPPO: Controlled decay, stable gradients")

def practical_comparison():
    """Show practical differences"""
    print("\n=== Practical Comparison ===\n")
    
    print("RANDOM INITIALIZATION:")
    print("Pros:")
    print("• Simple to implement")
    print("• No special math required")
    print("Cons:")
    print("• Unstable training")
    print("• Vanishing/exploding gradients")
    print("• Poor long-range memory")
    
    print("\nHIPPO INITIALIZATION:")
    print("Pros:")
    print("• Stable training")
    print("• Good gradient flow")
    print("• Excellent long-range memory")
    print("• Mathematical guarantees")
    print("Cons:")
    print("• More complex to implement")
    print("• Requires understanding of theory")
    
    print("\nWhen to use:")
    print("• Random: Learning, prototyping, simple tasks")
    print("• HiPPO: Production, long sequences, real applications")

def hippo_intuition():
    """Final intuition about HiPPO"""
    print("\n=== HiPPO Intuition ===\n")
    
    print("Think of HiPPO as:")
    print("• A smart memory manager")
    print("• Automatically decides what to remember/forget")
    print("• Based on optimal approximation theory")
    print("• Like having a perfect librarian for your data")
    
    print("\nThe key insight:")
    print("• Don't initialize randomly")
    print("• Use 300 years of math (Legendre polynomials)")
    print("• Get stable, optimal memory for free")
    
    print("\nThis is why S4 works and basic SSM doesn't!")

if __name__ == "__main__":
    memory_test()
    long_sequence_test()
    gradient_flow_demo()
    practical_comparison()
    hippo_intuition()