
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy
import lightning as L


class Pan2425Dataset(Dataset):
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FullyConnectedNeuralNetwork(L.LightningModule):
    def __init__(self, n_features=175, n_authors=20, learning_rate=0.0001):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_authors),
            # torch.nn.Softmax(dim=1) # Softmax is applied in loss function.
        )

        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        features, labels = batch

        output = self.forward(features)
        loss = cross_entropy(output, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        output = self.forward(features)
        loss = cross_entropy(output, labels)

        prediction = torch.argmax(output, dim=1)
        accuracy = torch.mean((prediction == labels).float())

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
