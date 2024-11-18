
import torch
from torch.utils.data import Dataset

class Pan2425Dataset(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AuthorIdentificationNetwork(torch.nn.Module):
    def __init__(self, input_features:int=175, learning_rate:float=1e-3, weight_decay:float=1e-5):
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
    


