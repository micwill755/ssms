import numpy as np
import sys
import os

# Add paths to import both models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'basic'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 's4'))

from text_utils import SimpleTokenizer
from ssm_language_model import SSMLanguageModel
from s4_language_model import S4LanguageModel

def compare_matrix_structures():
    """Show the key difference in matrix A"""
    print("=== MATRIX A COMPARISON ===\n")
    
    state_dim = 6
    
    # Create both models
    basic_model = SSMLanguageModel(vocab_size=10, state_dim=state_dim, seq_len=10)
    s4_model = S4LanguageModel(vocab_size=10, state_dim=state_dim, seq_len=10)
    
    print("BASIC SSM Matrix A:")
    print(basic_model.A)
    print(f"Structure: Dense matrix")
    print(f"Eigenvalues: {np.linalg.eigvals(basic_model.A)}")
    
    print("\nS4 Matrix A:")
    print(s4_model.A)
    print(f"Structure: Diagonal matrix")
    print(f"Eigenvalues: {np.diag(s4_model.A)}")
    
    print("\nKEY DIFFERENCES:")
    print("• Basic: Random dense matrix, unstable eigenvalues")
    print("• S4: Diagonal matrix, stable negative eigenvalues")

def compare_computation_efficiency():
    """Show computational differences"""
    print("\n=== COMPUTATIONAL EFFICIENCY ===\n")
    
    state_dim = 4
    
    # Basic SSM: Dense matrix power
    A_basic = np.random.randn(state_dim, state_dim) * 0.1
    print("Basic SSM: Computing A^3")
    A_power = np.eye(state_dim)
    for i in range(3):
        A_power = A_power @ A_basic
        print(f"A^{i+1} =\n{A_power}")
    
    # S4: Diagonal matrix power
    A_s4 = np.diag([-0.5, -1.0, -1.5, -2.0])
    print("\nS4: Computing A^3 (diagonal)")
    A_diag = np.diag(A_s4)
    for i in range(3):
        A_power_diag = A_diag ** (i+1)
        print(f"A^{i+1} diagonal = {A_power_diag}")
        print(f"A^{i+1} =\n{np.diag(A_power_diag)}")
    
    print("\nEfficiency:")
    print("• Basic: O(n³) for each A^i computation")
    print("• S4: O(n) for each A^i computation")

def compare_memory_properties():
    """Show memory decay patterns"""
    print("\n=== MEMORY PROPERTIES ===\n")
    
    # Basic SSM eigenvalues (random)
    basic_eigenvals = np.array([0.1, -0.3, 0.8, -0.2])
    
    # S4 eigenvalues (HiPPO-inspired)
    s4_eigenvals = np.array([-0.5, -1.0, -1.5, -2.0])
    
    print("Memory decay over time (eigenvalue^t):")
    print("Time\tBasic SSM\tS4")
    print("-" * 30)
    
    for t in range(5):
        basic_decay = np.abs(basic_eigenvals[0] ** t)
        s4_decay = np.abs(s4_eigenvals[0] ** t)
        print(f"{t}\t{basic_decay:.4f}\t\t{s4_decay:.4f}")
    
    print("\nMemory characteristics:")
    print("• Basic: Unpredictable decay (can explode or vanish)")
    print("• S4: Controlled exponential decay (stable memory)")

def practical_differences():
    """Show practical implications"""
    print("\n=== PRACTICAL IMPLICATIONS ===\n")
    
    print("BASIC SSM:")
    print("✓ Simple to understand")
    print("✓ Good for learning concepts")
    print("✗ Vanishing gradients on long sequences")
    print("✗ Unstable training")
    print("✗ Poor long-range dependencies")
    
    print("\nS4:")
    print("✓ Stable training")
    print("✓ Handles long sequences (16K+ tokens)")
    print("✓ Better gradient flow")
    print("✓ Faster computation (diagonal A)")
    print("✓ Theoretical guarantees")
    print("✗ More complex initialization")
    
    print("\nWhen to use:")
    print("• Basic SSM: Learning, prototyping, understanding concepts")
    print("• S4: Real applications, long sequences, production systems")

if __name__ == "__main__":
    compare_matrix_structures()
    compare_computation_efficiency()
    compare_memory_properties()
    practical_differences()