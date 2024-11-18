from time import time

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Pan2425Dataset(Dataset):
    def __init__(self, features: np.array, labels: np.array, removed_feature=None):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.flatten())

        self.removed_feature = removed_feature  # To store removed feature and its index

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    def remove_feature(self, idx: int):
        """
        Removes a feature from the dataset. The last call to this method can be undone by restore_feature()
        :param idx: the index of the feature column to remove
        :return: A new dataset with the same data as this one, but without the removed feature.
        """
        features = self.features.numpy()

        # Save removed feature for later restoration
        removed_feature = (idx, features[:, idx])

        new_features = np.delete(features, idx, axis=1)
        return Pan2425Dataset(new_features, self.labels.numpy(), removed_feature=removed_feature)

    def restore_feature(self):
        """
        Adds back the last feature removed by remove_feature() to the dataset.
        :return: A new dataset with the same data as this one, but with the last removed feature re-added.
        """
        if self.removed_feature is None:
            raise ValueError("No feature has been removed to restore.")

        idx, removed_feature = self.removed_feature

        features = self.features.numpy()
        restored_features = np.insert(features, idx, removed_feature, axis=1)
        restored_dataset = Pan2425Dataset(restored_features, self.labels.numpy())
        self.removed_feature = None

        return restored_dataset


class AuthorIdentificationNetwork(torch.nn.Module):
    def __init__(self, input_features: int = 175, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_features, 300),
            torch.nn.BatchNorm1d(300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 20),
        )

        self.lr = learning_rate
        self.wd = weight_decay
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self, x):
        return self.net.forward(x)

    def loss(self, output, y):
        return self.loss_function(output, y)


def train(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
    start = time()

    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    # repeat for multiple epochs
    for epoch in range(epochs):
        print(f"Epoch [{epoch}/{epochs}]", end='\r')
        # First train:
        # compute the mean loss and accuracy for this epoch
        loss_sum = 0.0
        accuracy_sum = 0.0
        steps = 0

        # loop over all minibatches in the training set
        for x, y in train_loader:
            # compute the prediction given the input x
            output = net.forward(x)

            # compute the loss by comparing with the target output y
            loss = net.loss(output, y)

            # for a one-hot encoding, the output is a score for each class
            # we assign each sample to the class with the highest score
            pred_class = torch.argmax(output, dim=1)
            # compute the mean accuracy
            accuracy = torch.mean((pred_class == y).to(float))

            # reset all gradients to zero before backpropagation
            net.optimizer.zero_grad()
            # compute the gradient
            loss.backward()
            # use the optimizer to update the parameters
            net.optimizer.step()

            accuracy_sum += accuracy.detach().cpu().item()
            loss_sum += loss.detach().cpu().item()
            steps += 1

        train_loss.append(loss_sum / steps)
        train_acc.append(accuracy_sum / steps)

        # Then validate:
        with torch.no_grad():
            loss_sum = 0.0
            accuracy_sum = 0.0
            steps = 0
            for x, y in val_loader:
                # compute the prediction given the input x
                output = net.forward(x)

                # compute the loss by comparing with the target output y
                loss = net.loss(output, y)

                # for a one-hot encoding, the output is a score for each class
                # we assign each sample to the class with the highest score
                pred_class = torch.argmax(output, dim=1)
                # compute the mean accuracy
                accuracy = torch.mean((pred_class == y).to(float))
                accuracy_sum += accuracy.detach().cpu().item()
                loss_sum += loss.detach().cpu().item()
                steps += 1
            val_loss.append(loss_sum / steps)
            val_acc.append(accuracy_sum / steps)

    print(f"{epochs} epochs completed in {(time() - start):.2f} seconds", end='\r')
    return train_loss, val_loss, train_acc, val_acc


def plot_training(train_loss, val_loss, train_acc, val_acc, epochs: int):
    # Plot a couple of sample trajectories for the current ensemble size
    plt.figure(figsize=(8, 8))
    x_axis = list(range(len(train_acc)))
    plt.plot(x_axis, train_acc, label="Train accuracy", color="darkred")
    plt.plot(x_axis, val_acc, label="Validation accuracy", color="darkgreen")
    plt.plot(x_axis, train_loss, label="Train loss", color="red", linestyle='dashed')
    plt.plot(x_axis, val_loss, label="Validation loss", color="green", linestyle='dashed')
    plt.title(f'FCNN performance')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
