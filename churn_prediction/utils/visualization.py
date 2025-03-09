import matplotlib.pyplot as plt

def plot_cv_results(train_scores, val_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_scores) + 1), train_scores, label='Train ROC-AUC', marker='o', color='blue')
    plt.plot(range(1, len(val_scores) + 1), val_scores, label='Validation ROC-AUC', marker='o', color='red')
    plt.title('Random Forest Cross-Validation ROC-AUC Scores', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('ROC-AUC Score', fontsize=14)
    plt.xticks(range(1, len(train_scores) + 1))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss_and_auc(train_losses, val_losses, train_auc_scores, val_auc_scores, num_epochs):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_auc_scores, label='Training AUC', marker='o')
    plt.plot(range(1, num_epochs + 1), val_auc_scores, label='Validation AUC', marker='o')
    plt.title('Training and Validation AUC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_monthly_churn(monthly_data):
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_data['month_start'], monthly_data['rf_predicted_mean'], label='Random Forest Predicted Churn Probability', marker='o', color='blue')
    plt.fill_between(
        monthly_data['month_start'],
        monthly_data['rf_predicted_lower'],
        monthly_data['rf_predicted_upper'],
        color='blue', alpha=0.2, label='Random Forest 95% Confidence Interval'
    )
    plt.plot(monthly_data['month_start'], monthly_data['nn_predicted_mean'], label='Neural Network Predicted Churn Probability', marker='o', color='red')
    plt.fill_between(
        monthly_data['month_start'],
        monthly_data['nn_predicted_lower'],
        monthly_data['nn_predicted_upper'],
        color='red', alpha=0.2, label='Neural Network 95% Confidence Interval'
    )
    plt.plot(monthly_data['month_start'], monthly_data['actual_mean'], label='Actual Churn Rate', marker='o', color='orange')
    plt.title('Monthly Churn Probability vs Actual Churn Rate (Random Forest vs Neural Network)', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Churn Probability / Churn Rate', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()