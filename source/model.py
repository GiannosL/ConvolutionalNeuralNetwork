import torch.nn as nn
import torch.nn.functional as F

class Convolutional_Model(nn.Module):
    def __init__(self, output_features:int) -> None:
        super(Convolutional_Model, self).__init__()
        
        #
        channels_1 = 6
        channels_2 = 16

        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels_1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=channels_2, kernel_size=3, stride=1)

        # final dimensions
        # convolution with no padding is -2
        # pooling is the division by 2
        reduced_image_dims = int((((28 - 2)/2) - 2)/2)
        convolutional_output = channels_2 * reduced_image_dims * reduced_image_dims

        # normal fully connected layers
        self.fc1 = nn.Linear(convolutional_output, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_features)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten images
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x