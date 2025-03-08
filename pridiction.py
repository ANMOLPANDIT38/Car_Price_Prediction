# Load the trained model and feature names
import joblib
import pandas as pd
import numpy as np
loaded_model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Function to take user input and predict price
def predict_car_price(company, year, kms_driven, fuel_type):
    # Create a dataframe with user input
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = np.zeros(len(feature_names))  # Initialize with zeros

    # Set user inputs in dataframe
    input_data["year"] = year
    input_data["kms_driven"] = kms_driven

    # Encode categorical inputs
    company_col = f"company_{company}"
    fuel_col = f"fuel_type_{fuel_type}"
    
    if company_col in input_data.columns:
        input_data[company_col] = 1
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    # Predict price
    predicted_price = loaded_model.predict(input_data)[0]
    return predicted_price

# Example: User input
company = input("Enter car company (e.g., Hyundai, Maruti, Ford): ")
year = int(input("Enter car manufacturing year: "))
kms_driven = int(input("Enter kilometers driven: "))
fuel_type = input("Enter fuel type (Petrol/Diesel/LPG): ")

predicted_price = predict_car_price(company, year, kms_driven, fuel_type)
print(f"Predicted Car Price: ₹{predicted_price:,.2f}")
