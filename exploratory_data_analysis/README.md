# README: Churn Data Analysis and Visualization

## Overview
This project provides a structured pipeline for analyzing and visualizing churn data. It includes data processing functions, visualization utilities, and a main script to execute exploratory data analysis (EDA) workflows.

## Project Structure
```
├── eda_main.py          # Main script to run the EDA pipeline
├── data_processing.py   # Functions for filtering and preparing data
├── visualization.py     # Functions for visualizing churn trends
├── README.md            # Documentation for the project
```

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas matplotlib seaborn
```

## Usage
1. **Load and Process Data:**
   - The `eda_main.py` script expects a churn dataset (`churn_data`) in a Pandas DataFrame format.
   - It processes data using functions from `data_processing.py`.

2. **Data Processing Functions:**
   - `filter_churn_data(churn_data)`: Splits churn data into `churn_negative` and `churn_non_negative` based on `years_since_renewal`.
   - `prepare_churn_renewal_data(churn_data)`: Prepares churn data for `years_since_renewal = 0` with categorical ordering.

3. **Visualization Functions:**
   - `plot_surrender_by_years_since_renewal(churn_negative, churn_non_negative)`: Visualizes surrender rates based on years since renewal.
   - `plot_categorical_surrender_rates(churn_renewal_year_0, categorical_vars)`: Displays categorical variable surrender rates.
   - `plot_continuous_surrender_rates(churn_renewal_year_0, continuous_vars)`: Shows trends for continuous variables.
   - `plot_cumulative_surrender_rate(churn_renewal_year_0)`: Plots cumulative surrender rates over policy months.

4. **Run the Analysis:**
   - Modify `eda_main.py` to load your dataset.
   - Execute the script:
   ```bash
   python eda_main.py
   ```

## Data Requirements
The dataset should include the following columns:
- `years_since_renewal`
- `surrender`
- `product_length`
- `policy_month_in_year`
- `age_category`
- `account_value_category`
- `tax_status`
- `distributor_group`
- `product_type`
- `treasury_rate`
- `new_money_rate`