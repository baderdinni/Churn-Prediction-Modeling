import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from churn_prediction.models.random_forest import RandomForestModel

class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        # Create a synthetic binary classification dataset
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestModel(n_estimators=10, max_depth=3)
    def test_model_initialization(self):
        self.assertIsInstance(self.model, RandomForestModel)
        self.assertEqual(self.model.model.n_estimators, 10)
        self.assertEqual(self.model.model.max_depth, 3)
    def test_cross_validate(self):
        cv_results = self.model.cross_validate(self.X_train, self.y_train, cv=5)
        self.assertIn('train_score', cv_results)
        self.assertIn('test_score', cv_results)
        self.assertEqual(len(cv_results['train_score']), 5)
        self.assertEqual(len(cv_results['test_score']), 5)
    def test_fit(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.model, 'estimators_'))  # Check if the model has been fitted
    def test_predict_proba(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict_proba(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])  # Check if predictions match the number of test samples
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1) )  # Check if predictions are probabilities