import pandas as pd
from sklearn.model_selection import train_test_split

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df['PRCP_lag1'] = df['PRCP'].shift(1)
    df['TMAX_lag1'] = df['TMAX'].shift(1)
    df['TMIN_lag1'] = df['TMIN'].shift(1)
    df['RAIN_lag1'] = df['RAIN'].shift(1)
    df.dropna(inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)

if __name__ == "__main__":
    create_features("data/prepared.csv")
