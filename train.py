import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from dvclive import Live
import pickle
import os

import yaml
import sys

def load_params(params_path):
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def get_model(params_path):
    params = load_params(params_path)
    model_config = params['model']
    model_type = model_config['type']
    params = model_config['params']

    if model_type == 'logistic_regression':
        model = LogisticRegression(**params['logistic_regression'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params['random_forest'])
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**params['gradient_boosting'])
    else:
        raise ValueError("Unsupported model type")

    return model


def train_model(input_path, model_output_path):
    df = pd.read_csv(input_path)
    X = df[['PRCP', 'TMAX', 'TMIN', 'PRCP_lag1', 'TMAX_lag1', 'TMIN_lag1', 'RAIN_lag1']]
    y = df['RAIN'].astype(int)
    tscv = TimeSeriesSplit(n_splits=5)
    model = get_model('params.yaml')
    
    with Live() as live:
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            live.log_metric("accuracy", accuracy_score(y_test, y_pred))
            live.log_metric("precision", precision_score(y_test, y_pred))
            live.log_metric("recall", recall_score(y_test, y_pred))

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    train_model("data/train_data.csv", "model/model.pkl")
