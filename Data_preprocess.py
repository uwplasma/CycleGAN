import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(file_path, batch_size=32,nrows=500):
    # Load the data
    X = pd.read_csv(file_path, nrows=nrows)
    
    # Only use the first 10 columns
    X = X.iloc[:, :10]
    
    # Split the data into training, validation, and testing sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Convert the data into torch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_val = torch.tensor(X_val.values, dtype=torch.float)
    X_test = torch.tensor(X_test.values, dtype=torch.float)

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_and_preprocess_data('pass_desc.csv')