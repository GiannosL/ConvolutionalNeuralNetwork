import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from source.model import Convolutional_Model
from source.image_loader import Image_Loader


class CNN:
    def __init__(self, image_dimensions:int, output_classes:int, learning_rate:float=0.001) -> None:
        # initialized model
        self.model = Convolutional_Model(image_dims=image_dimensions, output_features=output_classes)

        # training_parameters
        self.learning_rate = learning_rate

        # loss measure
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # training statistics
        self.training_losses = []
        self.train_correct = []

        # latest prediction statistics
        self.prediction_loss = None
        self.prediction_correct = None

        # has the model been trained?
        self.trained = False
        self.prediction_flag = False
    
    def set_epochs(self, n:int) -> None:
        """
        sets number of epochs
        """
        self.n_epochs = n
    
    def set_training_data(self, data:Image_Loader) -> None:
        """
        set dataset for training
        """
        self.data = data

    def train(self, verbose:bool=True) -> None:
        """
        train the model on the training dataset
        """
        # start timing
        start_time = time.time()

        x = self.data.train_data
        for epoch in range(self.n_epochs):
            # show which epoch we are on
            print(f"Epoch {epoch+1}")

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
            
            # save for plotting
            self.training_losses.append(batch_loss.item())
            self.train_correct.append(train_correct)
        
        # model has been trained
        self.trained = True

        # calculate training time, in minutes
        self.training_time = (time.time() - start_time)/60
        # show training time if verbose
        if verbose:
            print(f"Training time: {self.training_time: 0.2f} mins")
    
    def plot_training_loss(self) -> None:
        """
        Plot loss over the range of training epochs.
        """
        # make sure the model has been trained
        if not self.trained:
            print("Model has not been trained yet!")
            return 1
        
        plt.figure(figsize=(10, 7))
        plt.plot(range(self.n_epochs), self.training_losses, color="maroon")
        plt.title("Training loss over epochs", fontsize=16, weight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.show()

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        """
        make predictions through batches, this is not
        used to train the model
        """
        test_correct = 0

        with torch.no_grad():
            for x_test, y_test in x:
                y_val = self.model(x_test)

                predicted = torch.max(y_val.data, 1)[1]
                test_correct += (predicted == y_test).sum()
        
        self.prediction_loss = self.loss_criterion(y_val, y_test).item()
        self.prediction_correct = test_correct
        self.prediction_flag = True
    
        return predicted
    
    def plot_pred_loss(self):
        """
        Plot prediction loss as a horizontal line
        in the sample plot with the training loss over
        epochs.
        """
        # make sure the model has been trained and made predictions
        if not self.trained:
            print("Model has not been trained yet!")
            return 1
        elif not self.prediction_flag:
            print("Model has not made a prediction yet!")
            return 1
        
        plt.figure(figsize=(10, 7))
        plt.axhline(y=self.prediction_loss, color="orange", label="test set")
        plt.plot(range(self.n_epochs), self.training_losses, color="black", label="train set")
        plt.title("Validation vs Training loss", fontsize=16, weight="bold")
        plt.ylabel("Loss", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend()
        plt.show()

    def save(self, save_path:str="my_model.pt") -> None:
        """
        save model
        """
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, model_path:str):
        self.model.load_state_dict(torch.load(model_path))
