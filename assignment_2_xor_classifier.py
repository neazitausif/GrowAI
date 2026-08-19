

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class XORNetwork(nn.Module):
    
    def __init__(self):
        super(XORNetwork, self).__init__()
        
        
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        
        
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        
        x = self.hidden(x)
        x = self.relu(x)
        
        
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x


class XORTrainer:
    def __init__(self, learning_rate=0.01, epochs=5000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")
        
        self.model = XORNetwork().to(self.device)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.loss_history = []
        
        
        self.X = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=torch.float32).to(self.device)
        
        self.y = torch.tensor([
            [0],
            [1],
            [1],
            [0]
        ], dtype=torch.float32).to(self.device)
    
    def train(self):
        
        print("\n" + "="*80)
        print(" STARTING XOR NETWORK TRAINING")
        print("="*80)
        print(f"Architecture: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)")
        print(f"Loss Function: Binary Cross-Entropy (BCE)")
        print(f"Optimizer: Adam (lr=0.01)")
        print(f"Training Epochs: {self.epochs}\n")
        
        for epoch in range(self.epochs):
            
            predictions = self.model(self.X)
            
            
            loss = self.criterion(predictions, self.y)
            
            
            self.optimizer.zero_grad()  
            loss.backward() 
            self.optimizer.step()  
            
            
            self.loss_history.append(loss.item())
            
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch [{epoch+1:5d}/{self.epochs}] | Loss: {loss.item():.6f}")
        
        print(f"\n Training Complete! Final Loss: {loss.item():.6f}")
    
    def evaluate(self):
        
        print("\n" + "="*80)
        print(" MODEL EVALUATION ON XOR DATASET")
        print("="*80)
        
        self.model.eval()  
        
        with torch.no_grad():
            predictions = self.model(self.X)
        
        predictions = predictions.cpu().numpy()
        y_true = self.y.cpu().numpy()
        
        print(f"\n{'Input':^15} | {'Expected':^12} | {'Predicted':^12} | {'Rounded':^10} | {'Correct':^10}")
        print("-" * 70)
        
        correct = 0
        for i in range(len(self.X)):
            x_val = self.X[i].cpu().numpy()
            expected = y_true[i][0]
            pred = predictions[i][0]
            rounded = round(pred)
            is_correct = (rounded == expected)
            
            if is_correct:
                correct += 1
            
            print(f"({x_val[0]:.0f}, {x_val[1]:.0f})      | {expected:^12.4f} | {pred:^12.4f} | {rounded:^10.0f} | {'✅' if is_correct else '❌':^10}")
        
        accuracy = (correct / len(self.X)) * 100
        print("-" * 70)
        print(f"\n Accuracy: {correct}/{len(self.X)} ({accuracy:.1f}%)")
        
        return predictions
    
    
    
    

def main():
    print("\n" + " ASSIGNMENT 2: XOR CLASSIFIER WITH NEURAL NETWORKS" + "\n")
    
    
    trainer = XORTrainer(learning_rate=0.01, epochs=5000)
    trainer.train()
    
    
    predictions = trainer.evaluate()
    plt.plot(trainer.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XOR Training Loss")
    plt.show()
    
    
    
    
    
    
   


if __name__ == "__main__":
    main()
