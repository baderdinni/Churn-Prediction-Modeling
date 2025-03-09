import matplotlib.pyplot as plt
import seaborn as sns

def plot_surrender_by_years_since_renewal(churn_negative, churn_non_negative):
    """
    Plots surrender rates based on years_since_renewal.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    sns.barplot(data=churn_negative, x='years_since_renewal', y='surrender', hue='product_length', palette='viridis', ax=axes[0])
    axes[0].set_title('Surrender Rate for Years Since Renewal < 0')
    axes[0].set_xlabel('years_since_renewal')
    axes[0].set_ylabel('Surrender Rate')
    axes[0].legend(title='product_length')
    axes[0].grid(True)
    
    sns.barplot(data=churn_non_negative, x='years_since_renewal', y='surrender', hue='product_length', palette='plasma', ax=axes[1])
    axes[1].set_title('Surrender Rate for Years Since Renewal >= 0')
    axes[1].set_xlabel('years_since_renewal')
    axes[1].set_ylabel('Surrender Rate')
    axes[1].legend(title='product_length')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_surrender_rates(churn_renewal_year_0, categorical_vars):
    """
    Plots surrender rates for various categorical variables.
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, var in enumerate(categorical_vars):
        sns.barplot(data=churn_renewal_year_0, x=var, y='surrender', palette='viridis', ax=axes[i])
        axes[i].set_title(f'Surrender Rate by {var} (Renewal Year = 0)')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Surrender Rate')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_continuous_surrender_rates(churn_renewal_year_0, continuous_vars):
    """
    Plots surrender rates for continuous variables.
    """
    fig, axes = plt.subplots(1, len(continuous_vars), figsize=(10, 5))
    if len(continuous_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(continuous_vars):
        sns.lineplot(data=churn_renewal_year_0, x=var, y='surrender', marker='o', ax=axes[i])
        axes[i].set_title(f'{var} (Renewal Year = 0)')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Surrender Rate')
    
    plt.tight_layout()
    plt.show()


def plot_surrender_rate_by_policy_month(churn_renewal_year_0):
    """
    Plots surrender rates by policy month in renewal year 0.
    """
    grouped = churn_renewal_year_0.groupby(['policy_month_in_year', 'product_length']).apply(
        lambda x: x['surrender'].sum() / x['surrender'].count()
    ).reset_index(name='surrender_rate')

    pivot_table = grouped.pivot(index='policy_month_in_year', columns='product_length', values='surrender_rate')
    pivot_table.plot(kind='line', marker='o')
    plt.title('Surrender Rate by Policy Month and Product Length (Renewal Year 0)')
    plt.xlabel('Policy Month in Year')
    plt.ylabel('Surrender Rate')
    plt.grid(True)
    plt.show()

def plot_cumulative_surrender_rate(churn_renewal_year_0):
    """
    Plots cumulative surrender rates by policy month in renewal year 0.
    """
    churn_renewal_year_0 = churn_renewal_year_0.sort_values(by='policy_month_in_year')
    grouped_survival = churn_renewal_year_0.groupby('product_length')
    survival_results = {}

    for product, product_data in grouped_survival:
        S_t = 1.0
        cumulative_surrender_rate = []
        
        for month in range(1, 13):
            month_data = product_data[product_data['policy_month_in_year'] == month]
            if len(month_data) > 0:
                hazard_rate = month_data['surrender'].sum() / len(month_data)
                S_t *= (1 - hazard_rate)
                cumulative_surrender_rate.append(1 - S_t)
            else:
                cumulative_surrender_rate.append(cumulative_surrender_rate[-1] if cumulative_surrender_rate else 0)
        
        survival_results[product] = cumulative_surrender_rate

    cumulative_surrender_df = pd.DataFrame(survival_results, index=range(1, 13))
    cumulative_surrender_df.plot(kind='line', marker='o')
    plt.title('Cumulative Surrender Rate by Policy Month and Product Length (Renewal Year 0)')
    plt.xlabel('Policy Month in Year')
    plt.ylabel('Cumulative Surrender Rate')
    plt.grid(True)
    plt.show()