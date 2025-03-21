import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from churn_prediction.models.feed_forward_nn import FeedForwardNN, NeuralNetworkModel

class TestFeedForwardNN(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size1 = 5
        self.hidden_size2 = 3
        self.output_size = 1
        self.model = FeedForwardNN(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
    def test_model_initialization(self):
        self.assertIsInstance(self.model, FeedForwardNN)
        self.assertEqual(self.model.fc1.in_features, self.input_size)
        self.assertEqual(self.model.fc3.out_features, self.output_size)
    def test_forward_pass(self):
        x = torch.randn(1, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, (1, self.output_size))
        
class TestNeuralNetworkModel(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size1 = 5
        self.hidden_size2 = 3
        self.output_size = 1
        self.model = NeuralNetworkModel(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
        
        # Create dummy data
        X_train = np.random.rand(100, self.input_size).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)
        X_val = np.random.rand(20, self.input_size).astype(np.float32)
        y_val = np.random.randint(0, 2, size=(20, 1)).astype(np.float32)
        # Create DataLoader
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=10)
        self.val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=5)
    def test_training(self):
        train_losses, val_losses, train_auc_scores, val_auc_scores = self.model.train(self.train_loader, self.val_loader, num_epochs=1)
        self.assertIsInstance(train_losses, list)
        self.assertIsInstance(val_losses, list)
        self.assertGreater(len(train_losses), 0)
        self.assertGreater(len(val_losses), 0)
    def test_predict_proba(self):
        X_test = np.random.rand(5, self.input_size).astype(np.float32)
        predictions = self.model.predict_proba(pd.DataFrame(X_test))
        self.assertEqual(predictions.shape, (5, self.output_size))