import pandas as pd
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from dvclive import Live

def aggregate_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df.set_index('DATE', inplace=True)
    df = df[df.index.isocalendar().week.between(35, 40)]
    weekly_aggregated = df.resample('W-MON').agg({
        'PRCP': 'sum', 
        'TMAX': 'mean', 
        'TMIN': 'mean', 
        'RAIN': 'max'
    })
    weekly_aggregated['PRCP_lag1'] = weekly_aggregated['PRCP'].shift(1)
    weekly_aggregated['TMAX_lag1'] = weekly_aggregated['TMAX'].shift(1)
    weekly_aggregated['TMIN_lag1'] = weekly_aggregated['TMIN'].shift(1)
    weekly_aggregated['RAIN_lag1'] = weekly_aggregated['RAIN'].shift(1)
    weekly_aggregated.dropna(inplace=True)
    print(weekly_aggregated.head())
    return weekly_aggregated

def prediction_model(weekly_aggregated):
    weekly_aggregated['RAIN'] = weekly_aggregated['RAIN'].astype(int)
    weekly_aggregated['RAIN_lag1'] = weekly_aggregated['RAIN_lag1'].astype(int)
    X = weekly_aggregated[['PRCP', 'TMAX', 'TMIN', 'PRCP_lag1', 'TMAX_lag1', 'TMIN_lag1', 'RAIN_lag1']]
    y = weekly_aggregated['RAIN']
    tscv = TimeSeriesSplit(n_splits=5)
    logistic_model = LogisticRegression()

    with Live() as live:
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logistic_model.fit(X_train_scaled, y_train)
            y_pred = logistic_model.predict(X_test_scaled)

            # Log metrics
            live.log_metric("accuracy", accuracy_score(y_test, y_pred))
            live.log_metric("precision", precision_score(y_test, y_pred))
            live.log_metric("recall", recall_score(y_test, y_pred))

if __name__ == "__main__":
    file_path = "../data/vineyard_weather_1948-2017.csv"
    weekly_aggregated = aggregate_data(file_path)
    prediction_model(weekly_aggregated)
