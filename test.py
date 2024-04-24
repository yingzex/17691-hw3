import numpy as np
import matplotlib.pyplot as plt

def predictor_value(sensitivity):
    specificity = sensitivity
    storm_chance = 0.5
    no_storm_chance = 1 - storm_chance

    harvest_now_revenue = 80000
    storm_no_mold_revenue = 35000
    storm_mold_revenue = 275000
    no_storm_no_sugar_revenue = 80000
    no_storm_typical_sugar_revenue = 117500
    no_storm_high_sugar_revenue = 125000

    no_storm_chance_on_predictoin_no_storm = specificity/(specificity+1-sensitivity)
    storm_chance_on_prediction_no_storm = (1-sensitivity)/(specificity+1-sensitivity)
    storm_chance_on_prediction_storm = sensitivity/(sensitivity+1-specificity)
    no_storm_chance_on_prediction_storm = (1-specificity)/(sensitivity+1-specificity)

    no_harvest_on_no_storm_e_value = no_storm_no_sugar_revenue * 0.6 + no_storm_typical_sugar_revenue * 0.3 + no_storm_high_sugar_revenue * 0.1
    no_harvest_on_storm_e_value = storm_no_mold_revenue * 0.9 + storm_mold_revenue * 0.1
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
    # print(no_harvest_on_prediction_no_storm_e_value, no_harvest_on_prediction_storm_e_value)
    # print(predictor_value)

    decision = prediction_storm_e_value
    if (predictor_value == 0):
        print(sensitivity)
    return predictor_value

sensitivity_values = np.linspace(0, 1, 100)
predictor_values = [predictor_value(s) for s in sensitivity_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sensitivity_values, predictor_values, label='Predictor Value', color='blue')
plt.title('Predictor Value vs Sensitivity/Specificity')
plt.xlabel('Sensitivity/Specificity')
plt.ylabel('Predictor Value ($)')
plt.axhline(0, color='red', linestyle='--', label='Zero Value Line')
plt.legend()
plt.grid(True)
plt.show()
