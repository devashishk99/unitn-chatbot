import torch
import torch.nn as nn

# Feed-Forward NN Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # Hidden layer 1
        self.l3 = nn.Linear(hidden_size, num_classes) # Hidden layer 2
        self.relu = nn.ReLU() # Activation function
    
    # Forward Pass definiton
    def forward(self, x):
        out = self.l1(x) # First layer
        out = self.relu(out) # Followed by activation function
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax at the end
        return out
