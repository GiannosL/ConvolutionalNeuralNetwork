import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from source.model import Convolutional_Model


class CNN:
    def __init__(self, output_classes:int, learning_rate:float=0.001) -> None:
        self.model = Convolutional_Model(output_features=output_classes)

        # loss measure
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # training statistics
        self.training_losses = []
        self.train_correct = []

        # has the model been trained?
        self.trained = False

    def train(self, x, epochs:int=5, verbose:bool=True):
        start_time = time.time()
        self.epochs = epochs

        for epoch in range(epochs):
            # show which epoch we are on
            print(f"Epoch {epoch}")

            # number of correct predictions
            train_correct = 0

            for x_train, y_train in x:
                
                y_pred = self.model(x_train)
                
                batch_loss = self.loss_criterion(y_pred, y_train)
                
                predicted = torch.max(y_pred.data, 1)[1]
                batch_correct = (predicted == y_train).sum()
                train_correct += batch_correct

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            
            self.training_losses.append(batch_loss.item())
            self.train_correct.append(train_correct)
        
        # model has been trained
        self.trained = True

        # calculate training time, in minutes
        self.training_time = (time.time() - start_time)/60
        #
        if verbose:
            print(f"Training time: {self.training_time: 0.2f} mins")
    
    def plot_training(self):
        """
        Plot loss over the range of training epochs.
        """
        if not self.trained:
            print("Model has not been trained yet!")
            return 1
        
        plt.figure(figsize=(10, 7))
        plt.plot(range(self.epochs), self.training_losses, color="maroon",)
        plt.title("Training loss over epochs", fontsize=16, weight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.show()

    def predict(self, x):
        test_correct = []

        with torch.no_grad():
            for x_test, y_test in x:
                y_val = self.model(x_test)

                predicted = torch.max(y_val.data, 1)[1]
                test_correct += (predicted == y_test).sum()
        
        loss = self.loss_criterion(y_val, y_test)
    
        return predicted