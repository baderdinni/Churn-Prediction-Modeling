import pandas as pd
from data_processing import filter_churn_data, prepare_churn_renewal_data
from visualization import plot_surrender_by_years_since_renewal, plot_categorical_surrender_rates, plot_continuous_surrender_rates, plot_cumulative_surrender_rate

# Load data (Replace with actual data loading code)
# churn_data = pd.read_csv("your_data.csv")

# Process data
churn_negative, churn_non_negative = filter_churn_data(churn_data)
churn_renewal_year_0 = prepare_churn_renewal_data(churn_data)

# Define categorical and continuous variables
categorical_vars = ['tax_status', 'policy_month_in_year', 'distributor_group', 'account_value_category', 'product_type', 'age_category']
continuous_vars = ['treasury_rate', 'new_money_rate']

# Generate plots
plot_surrender_by_years_since_renewal(churn_negative, churn_non_negative)
plot_categorical_surrender_rates(churn_renewal_year_0, categorical_vars)
plot_continuous_surrender_rates(churn_renewal_year_0, continuous_vars)
plot_cumulative_surrender_rate(churn_renewal_year_0)
