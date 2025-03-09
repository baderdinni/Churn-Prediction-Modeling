import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_metrics(data):
    """
    Calculates key metrics for churn analysis, including actual surrender rate, predicted surrender rates,
    confidence intervals, and A/E ratios.

    Args:
        data (pd.DataFrame): DataFrame containing monthly churn data with columns:
            - 'actual_churn': Actual churn (0 or 1).
            - 'rf_predicted_churn_prob': Random Forest predicted churn probability.
            - 'nn_predicted_churn_prob': Neural Network predicted churn probability.
            - 'n_obs': Number of observations for each month.

    Returns:
        dict: A dictionary containing the following metrics:
            - actual_surrender_rate: Weighted mean of actual churn.
            - rf_predicted_surrender_rate: Weighted mean of Random Forest predictions.
            - nn_predicted_surrender_rate: Weighted mean of Neural Network predictions.
            - rf_lower_ci, rf_upper_ci: Confidence interval for Random Forest predictions.
            - nn_lower_ci, nn_upper_ci: Confidence interval for Neural Network predictions.
            - rf_ae_ratio, nn_ae_ratio: A/E ratios for Random Forest and Neural Network.
    """
    results = {}
    
    # Calculate weighted means
    total_obs = data['n_obs'].sum()
    results['actual_surrender_rate'] = (data['actual_churn'] * data['n_obs']).sum() / total_obs
    results['rf_predicted_surrender_rate'] = (data['rf_predicted_churn_prob'] * data['n_obs']).sum() / total_obs
    results['nn_predicted_surrender_rate'] = (data['nn_predicted_churn_prob'] * data['n_obs']).sum() / total_obs
    
    # Calculate confidence intervals for Random Forest
    rf_se = np.sqrt((results['rf_predicted_surrender_rate'] * (1 - results['rf_predicted_surrender_rate'])) / total_obs)
    results['rf_lower_ci'] = results['rf_predicted_surrender_rate'] - 1.96 * rf_se
    results['rf_upper_ci'] = results['rf_predicted_surrender_rate'] + 1.96 * rf_se
    
    # Calculate confidence intervals for Neural Network
    nn_se = np.sqrt((results['nn_predicted_surrender_rate'] * (1 - results['nn_predicted_surrender_rate'])) / total_obs)
    results['nn_lower_ci'] = results['nn_predicted_surrender_rate'] - 1.96 * nn_se
    results['nn_upper_ci'] = results['nn_predicted_surrender_rate'] + 1.96 * nn_se
    
    # Calculate A/E ratios
    results['rf_ae_ratio'] = results['actual_surrender_rate'] / results['rf_predicted_surrender_rate']
    results['nn_ae_ratio'] = results['actual_surrender_rate'] / results['nn_predicted_surrender_rate']
    
    return results


def generate_metrics_table(test_data):
    """
    Generates a table of churn metrics for inception-to-date (ITD) and last twelve months (LTM).

    Args:
        test_data (pd.DataFrame): DataFrame containing test data with columns:
            - 'month_start': Start date of the month.
            - 'actual_churn': Actual churn (0 or 1).
            - 'rf_predicted_churn_prob': Random Forest predicted churn probability.
            - 'nn_predicted_churn_prob': Neural Network predicted churn probability.

    Returns:
        pd.DataFrame: A table summarizing churn metrics for ITD and LTM periods.
    """
    # Define the latest date in the test data
    latest_date = test_data['month_start'].max()

    # Calculate inception-to-date (ITD) metrics
    itd_data = test_data.groupby('month_start').agg({
        'actual_churn': 'mean',
        'rf_predicted_churn_prob': 'mean',
        'nn_predicted_churn_prob': 'mean',
        'policy_date': 'count'
    }).reset_index()
    itd_data = itd_data.rename(columns={'policy_date': 'n_obs'})

    # Calculate last twelve months (LTM) metrics
    ltm_start_date = latest_date - timedelta(days=365)
    ltm_data = test_data[test_data['month_start'] >= ltm_start_date].groupby('month_start').agg({
        'actual_churn': 'mean',
        'rf_predicted_churn_prob': 'mean',
        'nn_predicted_churn_prob': 'mean',
        'policy_date': 'count'
    }).reset_index()
    ltm_data = ltm_data.rename(columns={'policy_date': 'n_obs'})

    # Calculate metrics for each period
    itd_metrics = calculate_metrics(itd_data)
    ltm_metrics = calculate_metrics(ltm_data)

    # Create a table to display the results
    metrics_table = pd.DataFrame({
        'Period': ['Inception-to-Date', 'Last Twelve Months'],
        'Actual Surrender Rate': [itd_metrics['actual_surrender_rate'], ltm_metrics['actual_surrender_rate']],
        'RF Predicted Surrender Rate': [itd_metrics['rf_predicted_surrender_rate'], ltm_metrics['rf_predicted_surrender_rate']],
        'RF Confidence Interval': [
            f"({itd_metrics['rf_lower_ci']:.4f}, {itd_metrics['rf_upper_ci']:.4f})",
            f"({ltm_metrics['rf_lower_ci']:.4f}, {ltm_metrics['rf_upper_ci']:.4f})"
        ],
        'RF A/E Ratio': [itd_metrics['rf_ae_ratio'], ltm_metrics['rf_ae_ratio']],
        'NN Predicted Surrender Rate': [itd_metrics['nn_predicted_surrender_rate'], ltm_metrics['nn_predicted_surrender_rate']],
        'NN Confidence Interval': [
            f"({itd_metrics['nn_lower_ci']:.4f}, {itd_metrics['nn_upper_ci']:.4f})",
            f"({ltm_metrics['nn_lower_ci']:.4f}, {ltm_metrics['nn_upper_ci']:.4f})"
        ],
        'NN A/E Ratio': [itd_metrics['nn_ae_ratio'], ltm_metrics['nn_ae_ratio']]
    })

    return metrics_table