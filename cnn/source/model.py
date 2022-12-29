import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolutional_Model(nn.Module):
    def __init__(self, image_dims:int, output_features:int, colored_image:bool=True) -> None:
        super(Convolutional_Model, self).__init__()
        #
        color_channels = 3 if colored_image else 1
        #
        channels_1 = 6
        channels_2 = 16

        # convolutional layer, input_channels refers to the image colours (1 - gray, 3 - colour)
        self.conv1 = nn.Conv2d(in_channels=color_channels, out_channels=channels_1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=channels_2, kernel_size=3, stride=1)

        # final dimensions
        # convolution with no padding is -2
        # pooling is the division by 2
        reduced_image_dims = int((((image_dims - 2)/2) - 2)/2)
        convolutional_output = channels_2 * reduced_image_dims * reduced_image_dims

        # normal fully connected layers
        self.fc1 = nn.Linear(convolutional_output, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_features)
    
    def forward(self, x) -> torch.Tensor:
        """
        pass the input image batch through the model
        """
        # pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # flatten images
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # softmax -> assign decimal probabilities to images
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
