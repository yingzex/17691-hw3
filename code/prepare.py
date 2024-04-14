import pandas as pd
import sys

def aggregate_data(file_path, output_path):
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df.set_index('DATE', inplace=True)
    df = df[df.index.isocalendar().week.between(35, 40)]
    df.to_csv(output_path)

if __name__ == "__main__":
    aggregate_data(sys.argv[1], sys.argv[2])

