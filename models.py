import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class FCNet(nn.Module):

    def __init__(self, activation_function_name):
        super(FCNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid

        self.linear1 = nn.Linear(32 * 32 * 3, 500)  # For each image, 32x32 with 3 colour channels
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # Apply the activation functions to each of the layers
        # "self.linear1(x)" is the output later of the layer 1, and passed into the activation function
        x = self.activation_function(self.linear1(x))     
        x = self.activation_function(self.linear2(x)) 
        x = self.linear3(x)  # In the spec: do not apply the activation function to 3rd layer
        return x


class ConvNet(nn.Module):

    def __init__(self, activation_function_name):
        super(ConvNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid
        # TODO: initialize the layers for the convolutional neural network (please do not change layer names!)
        self.conv1 = None
        self.conv2 = None
        self.maxpool2d = None
        self.flatten = None
        self.linear1 = None

    def forward(self, x):
        # TODO: complete the forward pass (use self.activation_function)
        
        return x