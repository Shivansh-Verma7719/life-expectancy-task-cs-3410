import pandas as pd

# Standardizes the features using Z-score normalization.
def scaler(X):
    X_scaled = X.copy()
    
    # Iterating over each col
    for column in X_scaled.columns:
        mean = X_scaled[column].mean()
        std = X_scaled[column].std()
        
        # Avoiding division by 0 
        if std != 0:
            X_scaled[column] = (X_scaled[column] - mean) / std
        else:
            # If std is 0 all values in the column are the same. Scaling results in 0.
            X_scaled[column] = 0
            
    return X_scaled

# Loads and preprocesses the life expectancy data.
def preprocess_data(data_path):    
    # Load data
    df = pd.read_csv(data_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Dropping the 'Country' column as it's an identifier and a string.
    df = df.drop('Country', axis=1)

    # Handling missing values by filling with the median
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    # Turning the 'Status' variable into int
    df['Status'] = df['Status'].apply(lambda x: 1 if x == 'Developed' else 0)

    # Separate features (X) and target (y)
    X = df.drop('Life expectancy', axis=1)
    y = df['Life expectancy']


    # Scaling numerical features
    X_scaled = scaler(X)

    return X_scaled, y

# For testing only
if __name__ == '__main__':
    # This is relative from base dir (I ran it from life_expectancy_task dir)
    X, y = preprocess_data('data/train_data.csv')
    print("Data preprocessed successfully")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape, "\n")
    print("First 5 rows of features (X):")
    print(X.head())