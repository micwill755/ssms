import numpy as np
from text_utils import SimpleTokenizer, load_sample_text, create_training_data, tokens_to_embeddings

class SSMLanguageModel:
    """SSM-based language model with dual computation modes"""
    
    def __init__(self, vocab_size, embed_dim=32, state_dim=64, seq_len=50):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # SSM matrices
        self.A = np.random.randn(state_dim, state_dim) * 0.01
        self.B = np.random.randn(state_dim, embed_dim) * 0.1
        self.C = np.random.randn(vocab_size, state_dim) * 0.1
        
        # Make A stable (eigenvalues < 1)
        self.A = self.A * 0.9
        
        # Pre-compute convolution kernel for training
        self.K = self._compute_convolution_kernel()
        
        print(f"SSM Language Model: vocab={vocab_size}, embed={embed_dim}, state={state_dim}")
    
    def _compute_convolution_kernel(self):
        """Compute K[i] = C @ A^i @ B for convolution mode"""
        K = []
        A_power = np.eye(self.state_dim)
        
        for i in range(self.seq_len):
            K_i = self.C @ A_power @ self.B
            K.append(K_i)
            A_power = A_power @ self.A
        
        return np.array(K)
    
    def forward_training(self, embeddings):
        """Training mode: use convolution for parallel processing"""
        batch_size, seq_len, _ = embeddings.shape
        outputs = []
        
        for b in range(batch_size):
            batch_outputs = []
            
            for t in range(seq_len):
                # Convolution: y[t] = sum(K[i] @ u[t-i] for i in range(t+1))
                y = np.zeros(self.vocab_size)
                
                for i in range(min(t+1, len(self.K))):
                    if t-i >= 0:
                        y += self.K[i] @ embeddings[b, t-i]
                
                batch_outputs.append(y)
            
            outputs.append(batch_outputs)
        
        return np.array(outputs)
    
    def forward_inference(self, embeddings):
        """Inference mode: use recurrent for memory efficiency"""
        seq_len, _ = embeddings.shape
        
        x = np.zeros(self.state_dim)  # Hidden state
        outputs = []
        
        for t in range(seq_len):
            # Recurrent: x[t+1] = A @ x[t] + B @ u[t]
            x = self.A @ x + self.B @ embeddings[t]
            
            # Output: y[t] = C @ x[t]
            y = self.C @ x
            outputs.append(y)
        
        return np.array(outputs)
    
    def predict_next_token(self, logits):
        """Convert logits to next token prediction"""
        # Simple: take argmax
        return np.argmax(logits)
    
    def generate_text(self, tokenizer, prompt="the", max_length=50):
        """Generate text using recurrent inference"""
        tokens = tokenizer.encode(prompt)
        generated = tokens.copy()
        
        print(f"Generating from prompt: '{prompt}'")
        
        for _ in range(max_length):
            # Convert recent tokens to embeddings
            recent_tokens = generated[-self.seq_len:]
            embeddings = tokens_to_embeddings(
                np.array([recent_tokens]), 
                tokenizer.vocab_size, 
                self.embed_dim
            )[0]  # Remove batch dimension
            
            # Get predictions using recurrent mode
            logits = self.forward_inference(embeddings)
            
            # Predict next token
            next_token = self.predict_next_token(logits[-1])
            generated.append(next_token)
            
            # Stop if we hit a period
            if tokenizer.id_to_char.get(next_token) == '.':
                break
        
        return tokenizer.decode(generated)

def train_ssm_model():
    """Train the SSM language model"""
    print("=== Training SSM Language Model ===\n")
    
    # Load and prepare data
    text = load_sample_text()
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    
    # Create training data
    seq_len = 30
    inputs, targets = create_training_data(text, tokenizer, seq_len)
    
    print(f"Training on {len(inputs)} sequences of length {seq_len}")
    
    # Create model
    model = SSMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        state_dim=32,
        seq_len=seq_len
    )
    
    # Simple training loop (no real optimization, just demonstration)
    print("\nTraining (simplified - no real gradients)...")
    
    for epoch in range(3):
        total_loss = 0
        
        # Process in small batches
        batch_size = 4
        for i in range(0, min(20, len(inputs)), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Convert to embeddings
            input_embeddings = tokens_to_embeddings(
                batch_inputs, tokenizer.vocab_size, model.embed_dim
            )
            
            # Forward pass using convolution (training mode)
            predictions = model.forward_training(input_embeddings)
            
            # Simple loss (just for demo)
            loss = 0
            for b in range(len(batch_targets)):
                for t in range(len(batch_targets[b])):
                    target_token = batch_targets[b, t]
                    pred_logits = predictions[b, t]
                    
                    # Simple loss: negative log likelihood
                    if target_token < len(pred_logits):
                        loss -= pred_logits[target_token]
            
            total_loss += loss
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.2f}")
    
    return model, tokenizer

def demo_inference(model, tokenizer):
    """Demonstrate text generation"""
    print("\n=== Text Generation (Inference Mode) ===\n")
    
    prompts = ["the", "in", "to be"]
    
    for prompt in prompts:
        generated = model.generate_text(tokenizer, prompt, max_length=30)
        print(f"Prompt: '{prompt}' -> '{generated}'\n")

if __name__ == "__main__":
    # Train model
    model, tokenizer = train_ssm_model()
    
    # Test inference
    demo_inference(model, tokenizer)
    
    print("\n🎯 Key Points:")
    print("• Training used CONVOLUTION mode (parallel)")
    print("• Inference used RECURRENT mode (sequential)")
    print("• Same SSM, different computation!")