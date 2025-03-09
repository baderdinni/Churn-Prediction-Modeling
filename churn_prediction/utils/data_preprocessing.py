import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    """
    Pre process training, test data for standardization, one hot encoding etc.
    """
    data['policy_date'] = pd.to_datetime(data['policy_date'])
    data = data.sort_values(by='policy_date')

    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    categorical_cols = ['product_type', 'line_of_business', 'tax_status', 'distributor_group']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
        label_encoders[col] = le

    X_train = train_data.drop(columns=['policy_date', 'surrender'])
    y_train = train_data['surrender']
    X_test = test_data.drop(columns=['policy_date', 'surrender'])
    y_test = test_data['surrender']

    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_normalized, columns=X_test.columns)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

def split_data(X_train, y_train):
    """
    Train and val split data definition.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    return X_train_split, X_val_split, y_train_split, y_val_split