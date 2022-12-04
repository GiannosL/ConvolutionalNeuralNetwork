#!/usr/bin/env python3
import torch.utils
import torchvision

from source.cnn import CNN
from source.visuals import *


# load dataset
train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=True)


# convolutional model
cnn_model = CNN(image_dimensions=28, output_classes=10)
cnn_model.train(x=train_loader, epochs=15)
cnn_model.predict(x=test_loader)

# save model
cnn_model.save()

# plot stuff
cnn_model.plot_training_loss()
cnn_model.plot_pred_loss()
