import pandas as pd
import numpy as np
from models.random_forest import RandomForestModel
from models.feed_forward_nn import NeuralNetworkModel
from utils.data_preprocessing import preprocess_data, split_data
from utils.visualization import plot_cv_results, plot_loss_and_auc, plot_monthly_churn
from torch.utils.data import DataLoader, TensorDataset
import torch
from utils.backtesting_analysis import generate_metrics_table
from utils.feature_importance import extract_and_plot_feature_importance

# Load data
data = pd.read_csv('data/churn_data.csv')

# Preprocess data
X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(data)

# Train Random Forest model
rf_model = RandomForestModel()
cv_results = rf_model.cross_validate(X_train, y_train)
plot_cv_results(cv_results['train_score'], cv_results['test_score'])

rf_model.fit(X_train, y_train)
rf_pred_proba = rf_model.predict_proba(X_test)

# Prepare data for Neural Network
X_train_split, X_val_split, y_train_split, y_val_split = split_data(X_train, y_train)

X_train_tensor = torch.tensor(X_train_split.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.float32).reshape(-1, 1)
X_val_tensor = torch.tensor(X_val_split.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train Neural Network
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1

nn_model = NeuralNetworkModel(input_size, hidden_size1, hidden_size2, output_size)
train_losses, val_losses, train_auc_scores, val_auc_scores = nn_model.train(train_loader, val_loader)

plot_loss_and_auc(train_losses, val_losses, train_auc_scores, val_auc_scores, num_epochs=20)

# Evaluate on test data
nn_pred_proba = nn_model.predict_proba(X_test)

# Add predictions to test data
test_data = data.iloc[int(0.8 * len(data)):]
test_data['rf_predicted_churn_prob'] = rf_pred_proba
test_data['nn_predicted_churn_prob'] = nn_pred_proba
test_data['actual_churn'] = y_test

# Aggregate monthly data
test_data['month_start'] = test_data['policy_date'].dt.to_period('M').dt.start_time
monthly_data = test_data.groupby('month_start').agg({
    'rf_predicted_churn_prob': 'mean',
    'nn_predicted_churn_prob': 'mean',
    'actual_churn': 'mean',
    'surrender': 'count'
}).reset_index()

monthly_data.columns = ['month_start', 'rf_predicted_mean', 'nn_predicted_mean', 'actual_mean', 'n_obs']

monthly_data['rf_predicted_se'] = np.sqrt(
    (monthly_data['rf_predicted_mean'] * (1 - monthly_data['rf_predicted_mean'])) / monthly_data['n_obs']
)
monthly_data['nn_predicted_se'] = np.sqrt(
    (monthly_data['nn_predicted_mean'] * (1 - monthly_data['nn_predicted_mean'])) / monthly_data['n_obs']
)

monthly_data['rf_predicted_lower'] = monthly_data['rf_predicted_mean'] - 1.96 * monthly_data['rf_predicted_se']
monthly_data['rf_predicted_upper'] = monthly_data['rf_predicted_mean'] + 1.96 * monthly_data['rf_predicted_se']
monthly_data['nn_predicted_lower'] = monthly_data['nn_predicted_mean'] - 1.96 * monthly_data['nn_predicted_se']
monthly_data['nn_predicted_upper'] = monthly_data['nn_predicted_mean'] + 1.96 * monthly_data['nn_predicted_se']

# Plot monthly churn
plot_monthly_churn(monthly_data)

# Back testing results
metrics_table = generate_metrics_table(test_data)
print(metrics_table)

# Feature importance
rf_feature_importance_df = extract_and_plot_feature_importance(rf_model, X_train.columns)