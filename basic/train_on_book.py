import numpy as np
import urllib.request
import os
from text_utils import SimpleTokenizer, create_training_data, tokens_to_embeddings
from ssm_language_model import SSMLanguageModel

def download_book(url, filename):
    """Download a book from Project Gutenberg"""
    if os.path.exists(filename):
        print(f"Using cached {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Downloaded {len(text)} characters")
        return text
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def clean_text(text):
    """Clean and prepare text for training"""
    if text is None:
        return None
    
    # Find start and end of actual content (skip Project Gutenberg headers)
    start_markers = ["*** START OF", "***START OF", "CHAPTER I", "Chapter 1"]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_idx)
        if idx != -1:
            end_idx = idx
            break
    
    # Extract main content
    content = text[start_idx:end_idx]
    
    # Basic cleaning
    content = content.replace('\r\n', '\n')
    content = content.replace('\r', '\n')
    
    # Remove excessive whitespace
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('CHAPTER'):
            cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

def train_on_real_book():
    """Train SSM on a real book"""
    print("=== Training SSM on Real Literature ===\n")
    
    # Try to download Alice in Wonderland (small, good for demo)
    book_url = "https://www.gutenberg.org/files/11/11-0.txt"
    book_file = "alice_in_wonderland.txt"
    
    text = download_book(book_url, book_file)
    
    if text is None:
        print("Using fallback sample text...")
        text = """
        Alice was beginning to get very tired of sitting by her sister on the bank, 
        and of having nothing to do. Once or twice she had peeped into the book her 
        sister was reading, but it had no pictures or conversations in it. And what 
        is the use of a book, thought Alice, without pictures or conversations? So 
        she was considering in her own mind, as well as she could, for the hot day 
        made her feel very sleepy and stupid, whether the pleasure of making a 
        daisy-chain would be worth the trouble of getting up and picking the daisies, 
        when suddenly a White Rabbit with pink eyes ran close by her.
        """
    else:
        text = clean_text(text)
    
    # Limit text size for demo
    text = text[:5000]  # First 5000 characters
    print(f"Training text length: {len(text)} characters")
    print(f"Sample: {text[:200]}...\n")
    
    # Prepare data
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    
    seq_len = 40
    inputs, targets = create_training_data(text, tokenizer, seq_len)
    print(f"Created {len(inputs)} training sequences")
    
    # Create and train model
    model = SSMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=20,
        state_dim=40,
        seq_len=seq_len
    )
    
    # Training with convolution mode
    print("\n🏋️ TRAINING MODE (Convolution - Parallel)")
    print("="*50)
    
    batch_size = 8
    num_batches = min(10, len(inputs) // batch_size)
    
    for epoch in range(5):
        total_loss = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Convert to embeddings
            input_embeddings = tokens_to_embeddings(
                batch_inputs, tokenizer.vocab_size, model.embed_dim
            )
            
            # Forward pass using CONVOLUTION (parallel training)
            predictions = model.forward_training(input_embeddings)
            
            # Compute simple loss
            batch_loss = 0
            for b in range(len(batch_targets)):
                for t in range(len(batch_targets[b])):
                    target_token = batch_targets[b, t]
                    if target_token < model.vocab_size:
                        pred_logits = predictions[b, t]
                        # Simple cross-entropy approximation
                        batch_loss += -pred_logits[target_token] + np.log(np.sum(np.exp(pred_logits)))
            
            total_loss += batch_loss
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.2f}")
    
    return model, tokenizer

def demo_text_generation(model, tokenizer):
    """Generate text using recurrent inference"""
    print("\n📱 INFERENCE MODE (Recurrent - Sequential)")
    print("="*50)
    
    prompts = ["alice", "the", "she", "when"]
    
    for prompt in prompts:
        print(f"\nGenerating from '{prompt}':")
        generated = model.generate_text(tokenizer, prompt, max_length=40)
        print(f"Result: {generated}")

def compare_modes_demo(model, tokenizer):
    """Show the difference between training and inference modes"""
    print("\n🔄 COMPARING COMPUTATION MODES")
    print("="*50)
    
    # Create a test sequence
    test_text = "alice was walking"
    test_tokens = tokenizer.encode(test_text)
    test_embeddings = tokens_to_embeddings(
        np.array([test_tokens]), tokenizer.vocab_size, model.embed_dim
    )
    
    print(f"Test input: '{test_text}'")
    
    # Training mode (convolution)
    print("\nTraining mode (convolution - parallel):")
    train_output = model.forward_training(test_embeddings)
    print(f"Output shape: {train_output.shape}")
    
    # Inference mode (recurrent)
    print("\nInference mode (recurrent - sequential):")
    inf_output = model.forward_inference(test_embeddings[0])
    print(f"Output shape: {inf_output.shape}")
    
    # Check if they're similar (they should be!)
    diff = np.abs(train_output[0] - inf_output).max()
    print(f"Max difference: {diff:.6f}")
    print("✓ Same results!" if diff < 0.1 else "⚠ Different results")

if __name__ == "__main__":
    # Train on real book
    model, tokenizer = train_on_real_book()
    
    # Generate text
    demo_text_generation(model, tokenizer)
    
    # Compare modes
    compare_modes_demo(model, tokenizer)
    
    print("\n🎯 Summary:")
    print("• Trained using CONVOLUTION (fast, parallel)")
    print("• Generated using RECURRENT (memory efficient)")
    print("• Same SSM math, different computation order!")
    print("• This is the secret of modern SSMs like Mamba!")