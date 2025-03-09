import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy.special import expit  # Logistic function

# Set random seed for reproducibility
np.random.seed(42)

def generate_policy_data(num_policies):
    """
    Generate base policy data for a given number of policies.

    Parameters:
    num_policies (int): Number of policies to generate.

    Returns:
    pd.DataFrame: A DataFrame containing the base policy data with columns:
                 - policy_id: Unique policy identifier.
                 - issue_date: Date the policy was issued.
                 - product_type: Type of product (e.g., 'Annuity', 'Indexed Annuity').
                 - product_length: Length of the product in years.
                 - line_of_business: Line of business (e.g., 'Retail', 'Reinsured').
                 - tax_status: Tax status of the policy (e.g., 'Qualified', 'Non-Qualified').
                 - distributor_group: Distributor group (e.g., 'Bank', 'Broker Dealer').
                 - initial_account_value: Initial account value of the policy.
                 - age_at_issue: Age of the policyholder at the time of issue.
                 - initial_crediting_rate: Initial crediting rate of the policy.

    Examples:
    >>> policies = generate_policy_data(10)
    >>> isinstance(policies, pd.DataFrame)
    True
    >>> len(policies) == 10
    True
    >>> set(policies.columns) == {'policy_id', 'issue_date', 'product_type', 'product_length', 'line_of_business', 'tax_status', 'distributor_group', 'initial_account_value', 'age_at_issue', 'initial_crediting_rate'}
    True
    """
    # Issue dates (randomly distributed over past 10 years)
    start_date = datetime(2007, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_between = (end_date - start_date).days
    
    issue_dates = [start_date + timedelta(days=random.randint(0, days_between)) 
                   for _ in range(num_policies)]
    
    # Product types and lengths
    product_types = np.random.choice(['Annuity', 'Indexed Annuity'], num_policies, p=[0.7, 0.3])
    
    # Product lengths (in years)
    product_lengths = np.random.choice([3, 5, 7], num_policies, p=[0.4, 0.3, 0.3])
    
    # Line of business
    line_of_business = np.random.choice(['Retail', 'Reinsured'], num_policies, p=[0.6, 0.4])
    
    # Tax status
    tax_status = np.random.choice(['Qualified', 'Non-Qualified'], num_policies, p=[0.65, 0.35])
    
    # Distributor groups
    distributor_groups = np.random.choice(['Bank', 'Broker Dealer', 'Independent Agent', 'Financial Advisor', 'Direct'], 
                                         num_policies, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Initial account values
    initial_values = np.random.choice([
        np.random.uniform(10000, 25000),
        np.random.uniform(25000, 50000),
        np.random.uniform(50000, 100000),
        np.random.uniform(100000, 200000),
        np.random.uniform(200000, 300000),
        np.random.uniform(300000, 500000)
    ], num_policies, p=[0.15, 0.25, 0.3, 0.2, 0.07, 0.03])
    
    # Policyholder age at issue
    age_at_issue = np.random.choice([
        np.random.randint(30, 45),
        np.random.randint(45, 55),
        np.random.randint(55, 65),
        np.random.randint(65, 70),
        np.random.randint(70, 80),
        np.random.randint(80, 90)
    ], num_policies, p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.1])
    
    # Initial crediting rates (based on product type and length)
    crediting_rates = []
    for i in range(num_policies):
        if product_types[i] == 'Annuity':
            base_rate = 2.0 + (product_lengths[i] * 0.2)
        else:  # Indexed annuity
            base_rate = 1.5 + (product_lengths[i] * 0.15)
            
        # Add some random variation
        rate = base_rate + np.random.uniform(-0.5, 0.5)
        crediting_rates.append(round(rate, 2))
    
    # Policy IDs
    policy_ids = [f'POL{str(i+1).zfill(6)}' for i in range(num_policies)]
    
    # Create base policy DataFrame
    base_policies = pd.DataFrame({
        'policy_id': policy_ids,
        'issue_date': issue_dates,
        'product_type': product_types,
        'product_length': product_lengths,
        'line_of_business': line_of_business,
        'tax_status': tax_status,
        'distributor_group': distributor_groups,
        'initial_account_value': initial_values,
        'age_at_issue': age_at_issue,
        'initial_crediting_rate': crediting_rates
    })
    
    return base_policies

def generate_policy_history(base_policies, observation_date=datetime(2024, 1, 1)):
    """
    Generate monthly policy history for each policy up to the observation date.

    Parameters:
    base_policies (pd.DataFrame): DataFrame containing base policy data.
    observation_date (datetime): The date up to which the history is generated.

    Returns:
    pd.DataFrame: A DataFrame containing the monthly policy history with columns:
                 - policy_id: Unique policy identifier.
                 - policy_date: Date of the policy month.
                 - policy_year: Policy year.
                 - policy_month_in_year: Month within the policy year.
                 - product_type: Type of product.
                 - product_length: Length of the product in years.
                 - line_of_business: Line of business.
                 - tax_status: Tax status of the policy.
                 - distributor_group: Distributor group.
                 - account_value: Account value at the end of the month.
                 - account_value_category: Category of account value.
                 - attained_age: Age of the policyholder at the end of the month.
                 - age_category: Category of attained age.
                 - crediting_rate: Crediting rate for the month.
                 - treasury_rate: Treasury rate for the month.
                 - new_money_rate: New money rate for the month.
                 - years_since_renewal: Years since the last renewal.
                 - in_first_renewal_year: Whether the policy is in the first renewal year.

    Examples:
    >>> policies = generate_policy_data(10)
    >>> history = generate_policy_history(policies)
    >>> isinstance(history, pd.DataFrame)
    True
    >>> set(history.columns) == {'policy_id', 'policy_date', 'policy_year', 'policy_month_in_year', 'product_type', 'product_length', 'line_of_business', 'tax_status', 'distributor_group', 'account_value', 'account_value_category', 'attained_age', 'age_category', 'crediting_rate', 'treasury_rate', 'new_money_rate', 'years_since_renewal', 'in_first_renewal_year'}
    True
    """
    policy_months = []
    
    # Economic data (5-year treasury rates) - monthly values for past 10 years
    # Simulating a realistic pattern with some trends
    months = 240  # 10 years of monthly data
    base_treasury_rate = 2.0
    treasury_trend = np.cumsum(np.random.normal(0, 0.1, months)) + base_treasury_rate
    treasury_trend = np.clip(treasury_trend, 0.5, 5.0)  # Keep within reasonable bounds
    
    # Company's new money rates (usually treasury + some spread)
    new_money_spread = 0.8
    new_money_rates = treasury_trend + new_money_spread + np.random.normal(0, 0.1, months)
    new_money_rates = np.clip(new_money_rates, 1.0, 6.0)
    
    # Map rates to dates
    rate_start_date = datetime(2007, 1, 1)
    rate_dates = [rate_start_date + timedelta(days=30*i) for i in range(months)]
    
    # Create rates dataframes for easy lookup
    treasury_df = pd.DataFrame({
        'date': rate_dates,
        'treasury_rate': treasury_trend
    })
    
    new_money_df = pd.DataFrame({
        'date': rate_dates,
        'new_money_rate': new_money_rates
    })
    
    # Process each policy
    for _, policy in base_policies.iterrows():
        # Calculate policy duration up to observation date
        policy_start = policy['issue_date']
        if policy_start >= observation_date:
            continue  # Skip policies issued after observation date
            
        max_months = (observation_date.year - policy_start.year) * 12 + (observation_date.month - policy_start.month)
        
        # Only include policies with at least 1 month of history
        if max_months <= 0:
            continue
            
        # Generate monthly records
        for month in range(1, max_months + 1):
            # Calculate policy date for this month
            policy_date = policy_start + timedelta(days=30*month)
            
            # Skip if beyond observation date
            if policy_date > observation_date:
                continue
                
            # Calculate policy year and policy month within year
            policy_year = (month - 1) // 12 + 1
            policy_month_in_year = (month - 1) % 12 + 1
            
            # Determine if in renewal period
            product_length = policy['product_length']
            years_since_renewal = policy_year - product_length
            in_renewal_period = (years_since_renewal == 1)
            
            # Calculate attained age
            attained_age = policy['age_at_issue'] + (month // 12)
            
            # Simulate account growth (simple modeling)
            # Start with initial value
            if month == 1:
                prev_account_value = policy['initial_account_value']
            else:
                prev_account_value = policy_months[-1]['account_value']
            
            # Growth based on crediting rate (simplistic model)
            crediting_rate = policy['initial_crediting_rate']
            monthly_growth = prev_account_value * (crediting_rate / 100 / 12)
            account_value = prev_account_value + monthly_growth
            
            # Get closest treasury and new money rates
            closest_date_idx = np.argmin([abs((d - policy_date).days) for d in rate_dates])
            treasury_rate = treasury_trend[closest_date_idx]
            new_money_rate = new_money_rates[closest_date_idx]
            
            # Create account value category
            if account_value < 25000:
                account_value_category = '0-25K'
            elif account_value < 50000:
                account_value_category = '25K-50K'
            elif account_value < 100000:
                account_value_category = '50K-100K'
            elif account_value < 200000:
                account_value_category = '100K-200K'
            elif account_value < 300000:
                account_value_category = '200K-300K'
            else:
                account_value_category = '300K+'
                
            # Create age category
            if attained_age < 45:
                age_category = '0-45'
            elif attained_age < 55:
                age_category = '45-55'
            elif attained_age < 65:
                age_category = '55-65'
            elif attained_age < 70:
                age_category = '65-70'
            elif attained_age < 80:
                age_category = '70-80'
            else:
                age_category = '80+'
            
            # Add to policy months
            policy_months.append({
                'policy_id': policy['policy_id'],
                'policy_date': policy_date,
                'policy_year': policy_year,
                'policy_month_in_year': policy_month_in_year,
                'product_type': policy['product_type'],
                'product_length': product_length,
                'line_of_business': policy['line_of_business'],
                'tax_status': policy['tax_status'],
                'distributor_group': policy['distributor_group'],
                'account_value': account_value,
                'account_value_category': account_value_category,
                'attained_age': attained_age,
                'age_category': age_category,
                'crediting_rate': crediting_rate,
                'treasury_rate': treasury_rate,
                'new_money_rate': new_money_rate,
                'years_since_renewal': years_since_renewal,
                'in_first_renewal_year': in_renewal_period
            })
    
    # Convert to DataFrame
    return pd.DataFrame(policy_months)

def calculate_surrender_probability(row):
    """
    Calculate the surrender probability for a given policy month.

    Parameters:
    row (pd.Series): A row from the policy history DataFrame containing relevant features.

    Returns:
    float: The probability of surrender for the given policy month.

    Examples:
    >>> policy = pd.Series({
    ...     'years_since_renewal': 0,
    ...     'crediting_rate': 2.5,
    ...     'treasury_rate': 3.0,
    ...     'new_money_rate': 3.5,
    ...     'tax_status': 'Qualified',
    ...     'product_type': 'Annuity',
    ...     'policy_month_in_year': 1,
    ...     'product_length': 5,
    ...     'age_category': '55-65',
    ...     'account_value_category': '50K-100K'
    ... })
    >>> 0 <= calculate_surrender_probability(policy) <= 1
    True
    """
    # Base log-odds
    log_odds = -6
    
    # Main effects
    # Years since renewal/maturity (from -0.9 to 0.33 at renewal, then decreasing to 0.18)
    ysr = row['years_since_renewal']
    if ysr < 0:
        # Before renewal, starts at -0.9 and increases toward renewal
        years_since_renewal_effect = -0.9 + (0.9 + 0.33) * (ysr + 10) / 10
    elif ysr == 0:
        # At renewal
        years_since_renewal_effect = 0.33
    else:
        # After renewal, starts at 0.33 and decreases to 0.18
        years_since_renewal_effect = max(0.33 - (0.33 - 0.18) * min(ysr, 4) / 4, 0.18)
    
    log_odds += years_since_renewal_effect
    
    # Crediting rate (negative effect)
    log_odds += -25 * (row['crediting_rate'] / 100)  # Convert percentage to decimal
    
    # Five-year gov rate (positive effect)
    log_odds += 16 * row['treasury_rate'] / 100  # Convert percentage to decimal
    
    # New money rate (positive effect)
    log_odds += 25 * row['new_money_rate'] / 100  # Convert percentage to decimal
    
    # Tax status
    if row['tax_status'] == 'Non-Qualified':
        log_odds += 0.08
    
    # Product type
    if row['product_type'] == 'Annuity':  # Non-Indexed
        log_odds += 1.1
    
    # --- Interaction terms ---
    
    # Policy month by renewal period
    if row['years_since_renewal'] < 0:  # Before renewal
        if row['policy_month_in_year'] == 1:
            log_odds += 0.5
    elif row['years_since_renewal'] == 0:  # First renewal year
        if row['policy_month_in_year'] == 1:
            log_odds += 3.2
        elif row['policy_month_in_year'] == 2:
            log_odds += 2.2
        elif row['policy_month_in_year'] == 3:
            log_odds += 1.2
        elif row['policy_month_in_year'] == 4:
            log_odds += 0.5
        elif row['policy_month_in_year'] == 5:
            log_odds += 0.2
    else:  # After first renewal
        if row['policy_month_in_year'] == 1:
            log_odds += 1.2
        elif row['policy_month_in_year'] == 2:
            log_odds += 1.15
        elif row['policy_month_in_year'] == 3:
            log_odds += 1.1
        elif row['policy_month_in_year'] == 4:
            log_odds += 1.07
        elif row['policy_month_in_year'] == 5:
            log_odds += 1.02
    
    # Product length by renewal period
    if row['product_length'] == 3:
        log_odds += 0.27*10
    elif row['product_length'] == 4:
        log_odds += 0.24*10
    elif row['product_length'] == 5:
        log_odds += 0.21*10
    elif row['product_length'] == 7:
        log_odds += 0.18*10
    
    # Age effects by renewal period
    age_cat = row['age_category']
    if row['years_since_renewal'] < 0:  # Before renewal
        if age_cat == '0-45':
            log_odds += -4.01
        elif age_cat == '45-55':
            log_odds += -4.02
        elif age_cat == '55-65':
            log_odds += -4.03
        elif age_cat == '65-70':
            log_odds += -4.04
        elif age_cat == '70-80':
            log_odds += -4.05
        else:  # 80+
            log_odds += -4.06
    else:  # After renewal (first or later)
        if age_cat == '0-45':
            log_odds += 0.1
        elif age_cat == '45-55':
            log_odds += 0.2
        elif age_cat == '55-65':
            log_odds += 0.3
        elif age_cat == '65-70':
            log_odds += 0.4
        elif age_cat == '70-80':
            log_odds += 0.5
        else:  # 80+
            log_odds += 0.3
    
    # Account value effects by renewal period
    av_cat = row['account_value_category']
    if row['years_since_renewal'] < 0:  # Before renewal
        if av_cat == '0-25K':
            log_odds += 1.6
        elif av_cat == '25K-50K':
            log_odds += 0.6
        elif av_cat == '50K-100K':
            log_odds += 0.2
        elif av_cat == '200K-300K':
            log_odds += -0.2
        elif av_cat == '300K+':
            log_odds += -0.6
    else:  # After renewal (first or later)
        if av_cat == '50K-100K':
            log_odds += 0.6
        elif av_cat == '100K-200K':
            log_odds += 0.7
        elif av_cat == '200K-300K':
            log_odds += 0.8
        elif av_cat == '300K+':
            log_odds += 0.9
    # Final simulation adjustment       
    log_odds -= 0.4
    # Convert log-odds to probability using logistic function
    probability = expit(log_odds)
    
    return probability

def generate_surrender_events(policy_history):
    """
    Generate surrender events based on the calculated surrender probabilities.

    Parameters:
    policy_history (pd.DataFrame): DataFrame containing the policy history.

    Returns:
    pd.DataFrame: A DataFrame with surrender events added.

    Examples:
    >>> policies = generate_policy_data(10)
    >>> history = generate_policy_history(policies)
    >>> history_with_surrender = generate_surrender_events(history)
    >>> isinstance(history_with_surrender, pd.DataFrame)
    True
    >>> 'surrender' in history_with_surrender.columns
    True
    """
    # Add surrender probability
    policy_history['surrender_probability'] = policy_history.apply(calculate_surrender_probability, axis=1)
    
    # Add random number for comparison
    policy_history['random_value'] = np.random.random(len(policy_history))
    
    # Determine surrender event (1 if random value < surrender probability)
    policy_history['surrender'] = (policy_history['random_value'] < policy_history['surrender_probability']).astype(int)
    
    # Clean up temporary columns
    policy_history = policy_history.drop(columns=['random_value'])
    
    # Identify first surrender for each policy
    policy_history['first_surrender'] = policy_history.groupby('policy_id')['surrender'].cumsum()
    policy_history['valid_record'] = (policy_history['first_surrender'] <= 1)
    
    # Filter out records after first surrender
    final_history = policy_history[policy_history['valid_record']].copy()
    final_history = final_history.drop(columns=['first_surrender', 'valid_record'])
    
    return final_history

def generate_annuity_churn_dataset(num_policies=1000):
    """
    Generate a complete dataset of annuity policies with churn (surrender) events.

    Parameters:
    num_policies (int): Number of policies to generate.

    Returns:
    tuple: A tuple containing two DataFrames:
           - final_dataset: The final dataset with surrender events.
           - policy_history: The policy history before adding surrender events.

    Examples:
    >>> final_data, history = generate_annuity_churn_dataset(10)
    >>> isinstance(final_data, pd.DataFrame) and isinstance(history, pd.DataFrame)
    True
    >>> len(final_data) <= len(history)
    True
    """
    print("Generating base policy data...")
    base_policies = generate_policy_data(num_policies)
    
    print("Generating policy history...")
    policy_history = generate_policy_history(base_policies)
    
    print("Generating surrender events...")
    final_dataset = generate_surrender_events(policy_history)
    
    print(f"Dataset generated with {len(final_dataset)} policy-month observations")
    print(f"Total policies: {final_dataset['policy_id'].nunique()}")
    print(f"Total surrenders: {final_dataset['surrender'].sum()}")
    
    return final_dataset, policy_history

# Generate the dataset
churn_data, policy_history = generate_annuity_churn_dataset(num_policies=9000)
churn_data.groupby(['product_length','years_since_renewal','product_length']).surrender.agg(['mean','count','sum'])

if __name__ == "__main__":
    import doctest
    doctest.testmod()