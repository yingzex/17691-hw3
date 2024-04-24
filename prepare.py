# Example content for prepare.py
import pandas as pd

def prepare_data(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['DATE'])
    df = df[df['DATE'].dt.isocalendar().week.isin(range(35, 41))]  # Filter weeks 35 to 40
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    prepare_data('data/vineyard_weather_1948-2017.csv', 'data/prepared.csv')
