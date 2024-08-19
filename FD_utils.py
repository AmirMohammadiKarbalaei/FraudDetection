import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the FraudDataset object with the given data.

        Args:
            data: The data to be stored in the FraudDataset object.

        Returns:
            None
        """
        self.data = data

    def __len__(self):
        """
        Returns the length of the data stored in the FraudDataset object.

        :return: An integer representing the length of the data.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index from the dataset.

        Args:
            idx (int): The index of the item to be retrieved.

        Returns:
            torch.tensor: The item at the specified index, converted to a torch tensor with float32 data type.
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)


class FraudDetectionModel(nn.Module):
    def __init__(self, num_numerical_features, num_types, embedding_dim=32):
        """
        Initializes the FraudDetectionModel object with the given parameters.

        Args:
            num_numerical_features (int): The number of numerical features in the input data.
            num_types (int): The number of types in the input data.
            embedding_dim (int, optional): The dimension of the embedding layer. Defaults to 32.

        Returns:
            None
        """
        super(FraudDetectionModel, self).__init__()
        self.type_embedding = nn.Embedding(num_types, embedding_dim)

        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Defines the forward pass of the FraudDetectionModel.

        Args:
            x (torch.tensor): The input tensor to be processed.

        Returns:
            torch.tensor: The output tensor after passing through the network.
        """
        type_embedded = self.type_embedding(x[:, 0].long())

        x_numerical = x[:, [1, 2, 3, 4, 5, 6, 7]]

        x = torch.cat([x_numerical, type_embedded], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
