import pandas as pd
import matplotlib.pyplot as plt

def extract_and_plot_feature_importance(rf_model, feature_names):
    """
    Extracts feature importance from a trained Random Forest model and plots it.

    Args:
        rf_model: Trained Random Forest model (e.g., from sklearn.ensemble.RandomForestClassifier).
        feature_names (list or pd.Index): List of feature names corresponding to the model's input.

    Returns:
        pd.DataFrame: A DataFrame containing feature names and their importance scores, sorted by importance.
    """
    # Extract feature importance from the trained Random Forest model
    rf_feature_importance = rf_model.feature_importances_

    # Create a DataFrame to store feature importance
    rf_feature_importance_df = pd.DataFrame({
        'feature': feature_names,  # Feature names
        'importance': rf_feature_importance  # Importance scores
    })

    # Sort the DataFrame by importance in descending order
    rf_feature_importance_df = rf_feature_importance_df.sort_values(by='importance', ascending=False)

    # Display the feature importance
    print(rf_feature_importance_df)

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(rf_feature_importance_df['feature'], rf_feature_importance_df['importance'], color='skyblue')
    plt.title('Random Forest Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

    return rf_feature_importance_df