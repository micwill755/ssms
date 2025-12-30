import numpy as np
import re

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        """Build vocabulary from text"""
        chars = sorted(set(text.lower()))
        self.vocab_size = len(chars)
        
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for i, char in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.char_to_id.get(char.lower(), 0) for char in text]
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return ''.join([self.id_to_char.get(id, '?') for id in ids])

def load_sample_text():
    """Create sample text data"""
    text = """
    The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.
    In the beginning was the Word, and the Word was with God, and the Word was God.
    To be or not to be, that is the question. Whether tis nobler in the mind to suffer.
    All happy families are alike; each unhappy family is unhappy in its own way.
    It was the best of times, it was the worst of times, it was the age of wisdom.
    Call me Ishmael. Some years ago never mind how long precisely having little or no money.
    In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.
    """
    return text.strip()

def create_training_data(text, tokenizer, seq_len=32):
    """Create input/target pairs for next-token prediction"""
    tokens = tokenizer.encode(text)
    
    inputs = []
    targets = []
    
    # Create overlapping sequences
    for i in range(len(tokens) - seq_len):
        input_seq = tokens[i:i+seq_len]
        target_seq = tokens[i+1:i+seq_len+1]  # Shifted by 1
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)

def tokens_to_embeddings(tokens, vocab_size, embed_dim=16):
    """Convert token IDs to simple one-hot embeddings"""
    batch_size, seq_len = tokens.shape
    embeddings = np.zeros((batch_size, seq_len, embed_dim))
    
    # Simple embedding: use first vocab_size dimensions as one-hot
    for b in range(batch_size):
        for t in range(seq_len):
            token_id = tokens[b, t]
            if token_id < embed_dim:
                embeddings[b, t, token_id] = 1.0
            else:
                # For tokens beyond embed_dim, use random projection
                np.random.seed(token_id)
                embeddings[b, t] = np.random.randn(embed_dim) * 0.1
    
    return embeddings

if __name__ == "__main__":
    # Test the tokenizer
    text = load_sample_text()
    print("Sample text:")
    print(text[:100] + "...")
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    
    # Test encoding/decoding
    sample = "hello world"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest: '{sample}' -> {encoded} -> '{decoded}'")
    
    # Create training data
    inputs, targets = create_training_data(text, tokenizer, seq_len=20)
    print(f"\nTraining data shape: inputs {inputs.shape}, targets {targets.shape}")
    
    # Show first example
    print(f"First input:  '{tokenizer.decode(inputs[0])}'")
    print(f"First target: '{tokenizer.decode(targets[0])}'")
    
    # Convert to embeddings
    input_embeddings = tokens_to_embeddings(inputs[:5], tokenizer.vocab_size)
    print(f"Embeddings shape: {input_embeddings.shape}")