import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from simple_ssm import SimpleSSM

def create_toy_data(batch_size=32, seq_len=50, input_dim=1):
    """Create a simple sequence prediction task"""
    # Generate sine wave data
    t = torch.linspace(0, 4*3.14159, seq_len)
    x = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, input_dim)
    
    # Target is the next step (shifted by 1)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = 0  # Last target is zero
    
    return x, y

def train_ssm():
    """Train SSM on sequence prediction"""
    # Model setup
    model = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    
    print("Training SSM on sine wave prediction...")
    for epoch in range(100):
        # Generate batch
        x, y = create_toy_data(batch_size=16, seq_len=30)
        
        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Test prediction
    with torch.no_grad():
        test_x, test_y = create_toy_data(batch_size=1, seq_len=50)
        pred = model(test_x)
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(test_x[0, :, 0].numpy(), label='Input', alpha=0.7)
        plt.plot(test_y[0, :, 0].numpy(), label='True Next', alpha=0.7)
        plt.plot(pred[0, :, 0].numpy(), label='SSM Prediction', alpha=0.7)
        plt.title('SSM Sequence Prediction')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/michaelwilliams/Documents/code/deep learning/ssms/ssm_training_results.png')
        plt.show()
        
        print(f"Final test loss: {criterion(pred, test_y).item():.6f}")

if __name__ == "__main__":
    train_ssm()