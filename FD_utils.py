import torch
import torch.nn as nn
from torch.utils.data import  Dataset

class FraudDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

class FraudDetectionModel(nn.Module):
    def __init__(self, num_numerical_features, num_types, embedding_dim=32):
        super(FraudDetectionModel, self).__init__()
        self.type_embedding = nn.Embedding(num_types, embedding_dim)
        
        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        type_embedded = self.type_embedding(x[:, 0].long())

        
        x_numerical = x[:, [ 1,2, 3,4, 5, 6,7]]
        
        x = torch.cat([x_numerical, type_embedded], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
