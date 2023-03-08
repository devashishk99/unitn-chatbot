import numpy as np
import random
import json # reads data in json format

import torch # open source machine learning framework
import torch.nn as nn # module to help in creating and training of the neural network
from torch.utils.data import Dataset, DataLoader # Dataset stores the samples and their corresponding labels and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

from nltk_utils import bag_of_words, tokenize, stem # to use functions defined in nltk_utils.py file
from model import NeuralNet # to use NeuralNet class module defined in model.py file

# loads the intents needed for the chatbot 
# Each conversational intent contains:
# 1) tag (a unique name)
# 2) patterns (sentence patterns for our neural network text classifier)
# 3) responses (one will be used as a response)
with open('intents.json', 'r') as f:
    intents = json.load(f) # stores all the intents present in json format as dictionary

all_words = [] # stores list of all the words
tags = [] # stores list of all the tags
xy = []  # stores list of pairs of words and tags

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag'] # gets tag associated with sentence
    # add to tag list
    tags.append(tag)
    # loops through list of all patterns for specific intent
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair tuple
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#display data extracted so far
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = [] # stores training features
y_train = [] # stores output label

# iterates over pair of pattern and tags extracted in above step
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) # input features i.e bag of words for each pattern_sentence is appended
    
    label = tags.index(tag)
    y_train.append(label) # output label i.e tag is appended

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8 # samples per batch to load (default: 1)
learning_rate = 0.001 # controls how much to change the model in response to the estimated error each time the model weights are updated
input_size = len(X_train[0])
hidden_size = 8 # size of the hidden layer in neuralnet
output_size = len(tags)
print(input_size, output_size)

# Class definition 
class ChatDataset(Dataset):
    # runs once when instantiating the Dataset object
    def __init__(self):
        self.n_samples = len(X_train) # gets the number of samples based on length of training set
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # return the size of dataset
    def __len__(self):
        return self.n_samples

dataset = ChatDataset() # initializes dataset class object 
train_loader = DataLoader(dataset=dataset, # dataset from which to load the data
                          batch_size=batch_size, # samples per batch to load (default: 1)
                          shuffle=True, # set to True to have the data reshuffled at every epoch (default: False)
                          num_workers=0) #  subprocesses to use for data loading. 0 means that the data will be loaded in the main process (default: 0)

# check to see if torch.cuda is available to be able to train our model on a hardware accelerator like the GPU else continue to use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creates an instance of NeuralNetwork, and moves it to the device defined above
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # computes the cross entropy loss between input logits and target
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # implements Adam algorithm

# Train the model
# loops through number of epochs defined above
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # add tuple of words and label into device
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # call the forward function defined in neuralnet and make predictions for this batch
        outputs = model(words)
        # computes the loss and its gradients
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # zero the parameter gradients
        loss.backward() # backprop the loss
        optimizer.step() # adjust learning weights

    # display the progress after every 100 epochs   
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = { 
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE) # saves trained model based on the data dictionary and file path defined above 

print(f'training complete. file saved to {FILE}')
