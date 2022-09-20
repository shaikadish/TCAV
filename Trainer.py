import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    """
    Class for training and evaluating model

    Methods
    -------
    train:
        Training loop for input model
    validation:
        Validation loop for input model
    fit:
        Train and validate model
    show_plots:
        Show training and validation over epochs

    """

    def __init__(self):
        ...

    def train(self, model, trainloader, optimizer, criterion):

        model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0

        for i, data in enumerate(trainloader):
            counter += 1

            # Load data
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(image)

            # Calculate Loss
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()

            # Calculate Accuracy
            preds = outputs.argmax(axis=1)
            train_running_correct += (preds.squeeze() == labels).sum().item()

            # Backprop and optimize
            loss.backward()
            optimizer.step()

        # Epoch Loss and Accuracy
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.sampler))

        return epoch_loss, epoch_acc

    def validate(self, model, testloader, criterion):

        model.eval()
        print('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                counter += 1

                # Load data
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                # Forward Pass
                outputs = model(image)

                # Calculate loss
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()

                # Calculate accuracy
                preds = outputs.argmax(axis=1)
                valid_running_correct += (preds.squeeze()
                                          == labels).sum().item()

        # Epoch Loss and Accuracy
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.sampler))
        return epoch_loss, epoch_acc

    def fit(self, model, train_loader, valid_loader, epochs=4, lr=1e-4, save=False, show=False):

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Track accuracy and loss
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        # Loop through epochs
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")

            # Train model
            train_epoch_loss, train_epoch_acc = self.train(model, train_loader,
                                                           optimizer, criterion)

            # Validate model
            valid_epoch_loss, valid_epoch_acc = self.validate(model, valid_loader,
                                                              criterion)

            # Track training and validation
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(
                f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(
                f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)

        # Save trained model
        if save:
            torch.save(model.state_dict(), f'models/model_e{epochs}.pth')

        # Visualize training progress
        if show:
            self.show_plots(train_acc, valid_acc, train_loss, valid_loss)

        print('TRAINING COMPLETE')

    def show_plots(self, train_acc, valid_acc, train_loss, valid_loss):

        # Accuacy
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-',
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-',
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-',
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-',
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
