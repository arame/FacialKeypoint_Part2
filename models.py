## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224 - 5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (10, 220, 220)
        # after one pool layer, this becomes (10, 110, 110)
        self.conv1 = nn.Conv2d(1, 16, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)

        # 16 outputs * the 110*110 filtered/pooled map size
        # 136 output channels (for the 136 classes)
        self.fc1 = nn.Linear(110 * 110 * 16, 136)
        print("model init")
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))

        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224 - 5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (10, 220, 220)
        # after one pool layer, this becomes (10, 110, 110)
        self.conv1 = nn.Conv2d(1, 16, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        ## output size = (W-F)/S +1 = (110 - 5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (10, 106, 106)
        # after one pool layer, this becomes (10, 53, 53)
        self.conv2 = nn.Conv2d(16, 16, 5)

        # 16 outputs * the 53*53 filtered/pooled map size
        # 136 output channels (for the 136 classes)
        self.fc1 = nn.Linear(16*53*53, 256)
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

