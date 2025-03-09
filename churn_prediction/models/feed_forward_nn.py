"""Feed forward neural network model definition."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.001):
        self.model = FeedForwardNN(input_size, hidden_size1, hidden_size2, output_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, val_loader, num_epochs=20):
        train_losses, val_losses = [], []
        train_auc_scores, val_auc_scores = [], []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            train_preds, train_labels = [], []

            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                train_preds.extend(outputs.detach().numpy())
                train_labels.extend(batch_y.detach().numpy())

            train_losses.append(epoch_train_loss / len(train_loader))
            train_auc = roc_auc_score(train_labels, train_preds)
            train_auc_scores.append(train_auc)

            self.model.eval()
            epoch_val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
                    val_preds.extend(outputs.numpy())
                    val_labels.extend(batch_y.numpy())

            val_losses.append(epoch_val_loss / len(val_loader))
            val_auc = roc_auc_score(val_labels, val_preds)
            val_auc_scores.append(val_auc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        return train_losses, val_losses, train_auc_scores, val_auc_scores

    def predict_proba(self, X_test):
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(torch.tensor(X_test.values, dtype=torch.float32))
            return test_outputs.numpy()