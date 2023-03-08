import torch  # open source machine learning framework
import torch.nn as nn # module to help in creating and training of the neural network

# Feed-Forward NN Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__() # call to parent function as it is a subclass of nn.Module and is inheriting all methods
        self.l1 = nn.Linear(input_size, hidden_size)  # input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # hidden layer 1
        self.l3 = nn.Linear(hidden_size, num_classes) # hidden layer 2
        self.relu = nn.ReLU() # activation function
    
    # Forward Pass definiton
    def forward(self, x):
        out = self.l1(x) # first layer
        out = self.relu(out) # followed by activation function
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
