import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Data Preparation
def aggregate_data(file_path):
	# 1.Aggregate the daily data into weekly data. Since you are focusing on weeks 35 to 40, ensure that you compute weekly totals (e.g., precipitation) and averages or maximums (e.g., temperature).
	file_path = '/home/kali/Desktop/17691/hw2/vineyard_weather_1948-2017.csv'

	# Read the data from the CSV file
	df = pd.read_csv(file_path, parse_dates=['DATE'])

	# Set 'DATE' as the index
	df.set_index('DATE', inplace=True)

	# Filter the data to include only weeks 35 to 40 of each year
	df = df[df.index.isocalendar().week.between(35, 40)]

	# Aggregate the data into weekly format, summing up the precipitation and averaging temperatures
	weekly_aggregated = df.resample('W-MON').agg({
		'PRCP': 'sum', 
		'TMAX': 'mean', 
		'TMIN': 'mean', 
		'RAIN': 'max'
	})

	# 2.Create lagged features (e.g., precipitation and temperature from previous weeks) to help the model understand trends and patterns over time.
	# Create lagged features for the previous week
	weekly_aggregated['PRCP_lag1'] = weekly_aggregated['PRCP'].shift(1)
	weekly_aggregated['TMAX_lag1'] = weekly_aggregated['TMAX'].shift(1)
	weekly_aggregated['TMIN_lag1'] = weekly_aggregated['TMIN'].shift(1)
	weekly_aggregated['RAIN_lag1'] = weekly_aggregated['RAIN'].shift(1)

	# Drop rows with NaN values that result from lagging
	weekly_aggregated.dropna(inplace=True)

	# # Display the first few rows to verify the result
	# print(weekly_aggregated.head())
	return weekly_aggregated

def prediction_model(weekly_aggregated):
	# Convert the boolean 'RAIN' column to integers for modeling
	weekly_aggregated['RAIN'] = weekly_aggregated['RAIN'].astype(int)
	weekly_aggregated['RAIN_lag1'] = weekly_aggregated['RAIN_lag1'].astype(int)

	# Split the dataset into features and target variable
	X = weekly_aggregated[['PRCP', 'TMAX', 'TMIN', 'PRCP_lag1', 'TMAX_lag1', 'TMIN_lag1', 'RAIN_lag1']]
	y = weekly_aggregated['RAIN']

	# TimeSeriesSplit provides train/test indices to split time series data samples in a sequential manner
	tscv = TimeSeriesSplit(n_splits=5)

	# Initialize the logistic regression model
	logistic_model = LogisticRegression()

	# Prepare to collect the evaluation metrics
	metrics = {
		'accuracy': [],
		'precision': [],
		'recall': []
	}

	# Iterate over the folds, preserving the time order
	for train_index, test_index in tscv.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]

		# Scale the features
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		# Fit the logistic regression model
		logistic_model.fit(X_train_scaled, y_train)

		# Predict on the test set
		y_pred = logistic_model.predict(X_test_scaled)

		# Calculate and store the metrics
		metrics['accuracy'].append(accuracy_score(y_test, y_pred))
		metrics['precision'].append(precision_score(y_test, y_pred))
		metrics['recall'].append(recall_score(y_test, y_pred))

	# Calculate the average of each metric across all folds
	average_metrics = {metric: sum(values) / len(values) for metric, values in metrics.items()}

	print(f"Average Accuracy: {average_metrics['accuracy']}")
	print(f"Average Precision: {average_metrics['precision']}")
	print(f"Average Recall: {average_metrics['recall']}")



file_path = '/home/kali/Desktop/17691/hw2/vineyard_weather_1948-2017.csv'
weekly_aggregated = aggregate_data(file_path=file_path)
prediction_model(weekly_aggregated)
