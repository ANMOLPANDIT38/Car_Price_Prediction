from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature names
model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess user input
def preprocess_input(company, year, kms_driven, fuel_type):
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = np.zeros(len(feature_names))  # Initialize with zeros

    # Assign values
    input_data["year"] = year
    input_data["kms_driven"] = kms_driven

    # Encode categorical inputs
    company_col = f"company_{company}"
    fuel_col = f"fuel_type_{fuel_type}"

    if company_col in input_data.columns:
        input_data[company_col] = 1
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    return input_data

# Home route - displays input form
@app.route("/")
def home():
    return render_template("index.html")

# API route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        company = request.form["company"]
        year = int(request.form["year"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = request.form["fuel_type"]

        # Preprocess input
        input_data = preprocess_input(company, year, kms_driven, fuel_type)

        # Predict price
        predicted_price = model.predict(input_data)[0]

        return jsonify({"predicted_price": f"₹{predicted_price:,.2f}"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
