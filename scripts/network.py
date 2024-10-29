import torch
import numpy as np
import lightning as L


class Pan2425Dataset(torch.utils.data.Dataset):
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FullyConnectedNeuralNetwork(L.LightningModule):
    def __init__(self, learning_rate=0.0001):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(175, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 20),
            # torch.nn.Softmax(dim=1) # Softmax is applied in loss function.
        )

        self.learning_rate = learning_rate
        self.loss_function = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        features, labels = batch

        output = self.forward(features)
        loss = self.loss_function(output, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        output = self.forward(features)
        loss = self.loss_function(output, labels)

        prediction = torch.argmax(output, dim=1)
        accuracy = torch.mean((prediction == labels).float())

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
