import pandas as pd

def filter_churn_data(churn_data):
    """
    Splits churn data into two groups based on 'years_since_renewal'.
    """
    churn_negative = churn_data[churn_data['years_since_renewal'] < 0]
    churn_non_negative = churn_data[(churn_data['years_since_renewal'] >= 0) & (churn_data['years_since_renewal'] <= 2)]
    return churn_negative, churn_non_negative

def prepare_churn_renewal_data(churn_data):
    """
    Prepares churn data for renewal year = 0 with ordered categorical variables.
    """
    age_category_order = ['0-45', '45-55', '55-65', '70-80', '80+']
    account_value_category_order = ['0-25K', '25K-50K', '50K-100K', '100K-200K', '200K-300K', '300K+']
    
    churn_renewal_year_0 = churn_data[churn_data['years_since_renewal'] == 0].copy()
    churn_renewal_year_0['age_category'] = pd.Categorical(
        churn_renewal_year_0['age_category'], categories=age_category_order, ordered=True
    )
    churn_renewal_year_0['account_value_category'] = pd.Categorical(
        churn_renewal_year_0['account_value_category'], categories=account_value_category_order, ordered=True
    )
    
    return churn_renewal_year_0