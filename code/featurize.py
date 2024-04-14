import pandas as pd
import sys

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df['PRCP_lag1'] = df['PRCP'].shift(1)
    df['TMAX_lag1'] = df['TMAX'].shift(1)
    df['TMIN_lag1'] = df['TMIN'].shift(1)
    df['RAIN_lag1'] = df['RAIN'].shift(1)
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    create_features(sys.argv[1], sys.argv[2])
