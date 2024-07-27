import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from FD_utils import DataLoader,FraudDataset,FraudDetectionModel




num_numerical_features=7 
embedding_dim=10
model_path = 'Models/fraud_detection_deep_model_0.94ac.pth'

# Load dataset
data = pd.read_csv('Fraud_data/Fraud.csv')
data = data[:1000]


# Encode categorical variables
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Fill missing values
data['oldbalanceDest'].fillna(0, inplace=True)
data['newbalanceDest'].fillna(0, inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest','nameOrig','nameDest']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and target
features = data.drop(columns=['isFlaggedFraud',"step"])

# Split data into training and testing sets
X = features.values


# Number of unique values for embeddings
num_types = data['type'].nunique()


# Prepare datasets and dataloaders
dataset = FraudDataset(X)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Ensure CUDA is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


model = FraudDetectionModel(
    num_numerical_features=num_numerical_features, 
    num_types=num_types, 
    embedding_dim=embedding_dim
)
model.load_state_dict(torch.load(model_path))
print("Model loaded successfully")
if torch.cuda.is_available():
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

else:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            probs = outputs.squeeze()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

np.save('predictions.npy', all_preds)

print("Predictions saved using numpy.")


