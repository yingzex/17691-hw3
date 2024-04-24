import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Assuming the model uses scaled features

sensitivity = 0.8
specificity = 0.8
storm_chance = 0.5
no_storm_chance = 1 - storm_chance

harvest_now_revenue = 80000
storm_no_mold_revenue = 35000
storm_mold_revenue = 275000
no_storm_no_sugar_revenue = 80000
no_storm_typical_sugar_revenue = 117500
no_storm_high_sugar_revenue = 125000

# Streamlit page configuration
st.title('Winery Decision Model')

# Define initial probabilities
default_probs = {
    'botrytis': 0.1,  # Probability of botrytis if there is a storm
    'no_sugar': 0.6,  # Probability of no sugar increase
    'typical_sugar': 0.3,  # Probability of typical sugar increase
    'high_sugar': 0.1,  # Probability of high sugar increase
}

# User controls to adjust probabilities
botrytis_chance = st.sidebar.slider('Chance of Botrytis (%)', 0, 100, int(default_probs['botrytis'] * 100)) / 100
no_sugar_chance = st.sidebar.slider('Chance of No Sugar Increase (%)', 0, 100, int(default_probs['no_sugar'] * 100)) / 100
typical_sugar_chance = st.sidebar.slider('Chance of Typical Sugar Increase (%)', 0, 100, int(default_probs['typical_sugar'] * 100)) / 100
high_sugar_chance = st.sidebar.slider('Chance of High Sugar Increase (%)', 0, 100, int(default_probs['high_sugar'] * 100)) / 100

# # Convert percentages to probabilities
# probabilities = {
#     'botrytis': botrytis_chance / 100,
#     'no_sugar': no_sugar_chance / 100,
#     'typical_sugar': typical_sugar_chance / 100,
#     'high_sugar': high_sugar_chance / 100
# }

st.write('Please enter the weather data to predict rain.')

# User inputs
prcp = st.number_input('PRCP (Precipitation)', min_value=0.0, format='%.2f')
tmax = st.number_input('TMAX (Max Temperature)', min_value=-100.0, max_value=100.0, format='%.1f')
tmin = st.number_input('TMIN (Min Temperature)', min_value=-100.0, max_value=100.0, format='%.1f')
prcp_lag1 = st.number_input('PRCP_lag1 (Previous Day Precipitation)', min_value=0.0, format='%.2f')
tmax_lag1 = st.number_input('TMAX_lag1 (Previous Day Max Temperature)', min_value=-100.0, max_value=100.0, format='%.1f')
tmin_lag1 = st.number_input('TMIN_lag1 (Previous Day Min Temperature)', min_value=-100.0, max_value=100.0, format='%.1f')
rain_lag1 = st.selectbox('RAIN_lag1 (Did it rain the previous day?)', options=[False, True])


# Function to load the trained model
def load_model():
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize the model
model = load_model()
scaler = StandardScaler()  # This should ideally be the scaler used during training

def calculate_expected_value():
    no_storm_chance_on_predictoin_no_storm = specificity/(specificity+1-sensitivity)
    storm_chance_on_prediction_no_storm = (1-sensitivity)/(specificity+1-sensitivity)
    storm_chance_on_prediction_storm = sensitivity/(sensitivity+1-specificity)
    no_storm_chance_on_prediction_storm = (1-specificity)/(sensitivity+1-specificity)

    no_harvest_on_no_storm_e_value = no_storm_no_sugar_revenue * no_sugar_chance + no_storm_typical_sugar_revenue * typical_sugar_chance + no_storm_high_sugar_revenue * high_sugar_chance
    no_harvest_on_storm_e_value = storm_no_mold_revenue * (1 - botrytis_chance) + storm_mold_revenue * botrytis_chance
    harvest_e_value = harvest_now_revenue
    e_value_without_predictor = harvest_e_value

    no_harvest_on_prediction_no_storm_e_value = no_storm_chance_on_predictoin_no_storm * no_harvest_on_no_storm_e_value + storm_chance_on_prediction_no_storm * no_harvest_on_storm_e_value
    no_harvest_on_prediction_storm_e_value = storm_chance_on_prediction_storm * no_harvest_on_storm_e_value + no_storm_chance_on_prediction_storm * no_harvest_on_no_storm_e_value

    prediction_storm_e_value = max(harvest_e_value, no_harvest_on_prediction_storm_e_value)
    prediction_no_storm_e_value = max(harvest_e_value, no_harvest_on_prediction_no_storm_e_value)

    prediction_storm_chance = 0.5 * (sensitivity+1-specificity)
    prediction_no_storm_chance = 0.5 * (specificity+1-sensitivity)

    e_value_with_predictor = prediction_storm_chance * prediction_storm_e_value + prediction_no_storm_chance * prediction_no_storm_e_value

    predictor_value = e_value_with_predictor - e_value_without_predictor
    return predictor_value, np.random.rand(), "Harvest Now"  # Dummy return

# Calculate e-value and recommended decision
predictor_value, e_value, recommendation = calculate_expected_value()


# Button to make prediction
if st.button('Predict'):
    # Create dataframe from user inputs
    input_df = pd.DataFrame([[prcp, tmax, tmin, prcp_lag1, tmax_lag1, tmin_lag1, rain_lag1]],
                            columns=['PRCP', 'TMAX', 'TMIN', 'PRCP_lag1', 'TMAX_lag1', 'TMIN_lag1', 'RAIN_lag1'])
    
    # Scale the input
    input_scaled = scaler.fit_transform(input_df)  # This should ideally use the scaler fitted during model training

    # Make prediction
    prediction = model.predict(input_scaled)
    result = 'Yes' if prediction[0] == 1 else 'No'
    
    # Display the prediction
    st.write(f'Prediction: Will it rain tomorrow? {result}')

# Display results
st.write(f"Expected value of the decision: ${predictor_value:,.2f}")
st.write(f"Recommended alternative: {recommendation}")
