import numpy as np
import sys
import os

# Add basic folder to path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'basic'))
from text_utils import SimpleTokenizer, create_training_data, tokens_to_embeddings

class S4LanguageModel:
    """
    S4-based language model with HiPPO initialization
    Better than basic SSM for long sequences
    """
    
    def __init__(self, vocab_size, embed_dim=32, state_dim=64, seq_len=50):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # S4: Diagonal A matrix with HiPPO initialization
        self.A = self._hippo_diagonal_init(state_dim)
        self.B = np.random.randn(state_dim, embed_dim) * 0.1
        self.C = np.random.randn(vocab_size, state_dim) * 0.1
        
        # Pre-compute convolution kernel
        self.K = self._compute_convolution_kernel()
        
        print(f"S4 Language Model: vocab={vocab_size}, HiPPO diagonal A")
    
    def _hippo_diagonal_init(self, N):
        """HiPPO-inspired diagonal initialization"""
        # Negative eigenvalues for stability and memory
        eigenvalues = -np.linspace(0.1, 2.0, N)
        return np.diag(eigenvalues)
    
    def _compute_convolution_kernel(self):
        """Compute K[i] = C @ A^i @ B (efficient for diagonal A)"""
        K = []
        
        # For diagonal A, A^i is just diagonal^i
        A_diag = np.diag(self.A)
        
        for i in range(self.seq_len):
            # A^i for diagonal matrix
            A_power_diag = A_diag ** i
            A_power = np.diag(A_power_diag)
            
            K_i = self.C @ A_power @ self.B
            K.append(K_i)
        
        return np.array(K)
    
    def forward_training(self, embeddings):
        """Training mode: convolution"""
        batch_size, seq_len, _ = embeddings.shape
        outputs = []
        
        for b in range(batch_size):
            batch_outputs = []
            
            for t in range(seq_len):
                y = np.zeros(self.vocab_size)
                
                for i in range(min(t+1, len(self.K))):
                    if t-i >= 0:
                        y += self.K[i] @ embeddings[b, t-i]
                
                batch_outputs.append(y)
            
            outputs.append(batch_outputs)
        
        return np.array(outputs)
    
    def forward_inference(self, embeddings):
        """Inference mode: recurrent"""
        seq_len, _ = embeddings.shape
        
        x = np.zeros(self.state_dim)
        outputs = []
        
        for t in range(seq_len):
            # Efficient diagonal matrix multiplication
            x = self.A @ x + self.B @ embeddings[t]
            y = self.C @ x
            outputs.append(y)
        
        return np.array(outputs)
    
    def generate_text(self, tokenizer, prompt="the", max_length=50):
        """Generate text using S4"""
        tokens = tokenizer.encode(prompt)
        generated = tokens.copy()
        
        print(f"S4 generating from: '{prompt}'")
        
        for _ in range(max_length):
            recent_tokens = generated[-self.seq_len:]
            embeddings = tokens_to_embeddings(
                np.array([recent_tokens]), 
                tokenizer.vocab_size, 
                self.embed_dim
            )[0]
            
            logits = self.forward_inference(embeddings)
            next_token = np.argmax(logits[-1])
            generated.append(next_token)
            
            if tokenizer.id_to_char.get(next_token) == '.':
                break
        
        return tokenizer.decode(generated)

def compare_basic_vs_s4_language():
    """Compare basic SSM vs S4 for language modeling"""
    print("=== Basic SSM vs S4 Language Modeling ===\n")
    
    # Sample text
    text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter.
    In the beginning was the Word, and the Word was with God, and the Word was God.
    To be or not to be, that is the question whether tis nobler in the mind to suffer.
    """
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    
    # Create both models
    from ssm_language_model import SSMLanguageModel  # Basic SSM
    
    basic_model = SSMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        state_dim=32,
        seq_len=30
    )
    
    s4_model = S4LanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        state_dim=32,
        seq_len=30
    )
    
    print("\nMatrix A comparison:")
    print(f"Basic A eigenvalues: {np.linalg.eigvals(basic_model.A)[:5]}...")
    print(f"S4 A eigenvalues: {np.diag(s4_model.A)[:5]}...")
    
    print("\nKey S4 advantages:")
    print("• Diagonal A → faster computation")
    print("• HiPPO init → better long-range memory")
    print("• Stable eigenvalues → no vanishing gradients")
    
    # Generate text with both
    print("\nText generation comparison:")
    basic_text = basic_model.generate_text(tokenizer, "the", 20)
    s4_text = s4_model.generate_text(tokenizer, "the", 20)
    
    print(f"Basic SSM: {basic_text}")
    print(f"S4 Model:  {s4_text}")

if __name__ == "__main__":
    compare_basic_vs_s4_language()