import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

def evaluate_model(model_path, test_data_path, output_path):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['PRCP', 'TMAX', 'TMIN', 'PRCP_lag1', 'TMAX_lag1', 'TMIN_lag1', 'RAIN_lag1']]
    y_test = test_data['RAIN'].astype(int)

    # Scale features as was done in training
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)

    # Generate a classification report
    report = classification_report(y_test, y_pred)

    with open(output_path, 'w') as output_file:
        output_file.write(f"Accuracy: {accuracy}\n")
        output_file.write(f"Precision: {precision}\n")
        output_file.write(f"Recall: {recall}\n")
        output_file.write(report)

    # Print the report for quick debugging
    print(report)

if __name__ == "__main__":
    evaluate_model("model/model.pkl", "data/test_data.csv", "results/evaluation_report.txt")
