import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Global variables
JAVA_SERVER_URL = "http://localhost:80/api/train"
MODEL_PATH = "model/trained_model.pkl"
PLOTS_DIR = "plots"

# Ensure the plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)


def fetch_research_data():
    """Fetch research data dynamically from the Java backend."""
    try:
        # Make a POST request to fetch research data
        response = requests.post(JAVA_SERVER_URL, json={})  # Empty payload if required
        response.raise_for_status()
        research_data = response.json()

        # Ensure the data is a list of dictionaries
        if isinstance(research_data, list):
            return pd.DataFrame(research_data)
        else:
            raise ValueError("Research data is not in the expected format (list of dictionaries).")

    except Exception as e:
        raise RuntimeError(f"Failed to fetch research data: {e}")



def preprocess_research_data(research_data, real_data):
    """
    Preprocess research data to align with real data:
    - Filter by age range
    - Align start point to real data
    - Interpolate missing weeks
    - Scale to match the real data trend
    """
    print(f"Initial research data shape: {research_data.shape}")
    print(f"Initial real data shape: {real_data.shape}")

    # Filter research data to start from week 4
    min_age = 4
    max_age = real_data['age_weeks'].max()
    research_data = research_data[(research_data['age_weeks'] >= min_age) & (research_data['age_weeks'] <= max_age)]

    # Ensure numeric columns are used
    numeric_columns = ['age_weeks', 'pet_weight', 'food_intake']
    research_data = research_data[numeric_columns]
    research_data = research_data.dropna()

    # Group by age_weeks and calculate mean for aggregation
    research_data = research_data.groupby('age_weeks').mean().reset_index()

    # Interpolate missing weeks
    all_weeks = np.arange(min_age, max_age + 1)
    research_data = research_data.set_index('age_weeks').reindex(all_weeks).interpolate(method='linear').reset_index()
    research_data.rename(columns={'index': 'age_weeks'}, inplace=True)

    # Align starting point to match real data
    start_week = min_age
    real_start_weight = real_data.loc[real_data['age_weeks'] == start_week, 'pet_weight'].values[0]
    real_start_food = real_data.loc[real_data['age_weeks'] == start_week, 'food_intake'].values[0]

    research_data.loc[research_data['age_weeks'] == start_week, 'pet_weight'] = real_start_weight
    research_data.loc[research_data['age_weeks'] == start_week, 'food_intake'] = real_start_food

    # Scale research data to match the real data trend
    weight_ratio = real_data['pet_weight'].mean() / research_data['pet_weight'].mean()
    food_ratio = real_data['food_intake'].mean() / research_data['food_intake'].mean()
    research_data['pet_weight'] *= weight_ratio
    research_data['food_intake'] *= food_ratio

    print(f"Preprocessed research data shape: {research_data.shape}")
    return research_data

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve the generated plots as static files."""
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        required_columns = ["age_weeks", "pet_weight", "food_intake"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns in research data."}), 400

        X = df[["age_weeks", "food_intake"]]
        y = df["pet_weight"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        joblib.dump(model, MODEL_PATH)

        return jsonify({"message": "Model trained successfully.", "mse": mse}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_data():
    try:
        payload = request.get_json()
        if not isinstance(payload, dict) or "real_data" not in payload:
            return jsonify({"error": "Invalid payload format. Expected a dictionary with key 'real_data'."}), 400

        real_data = pd.DataFrame(payload["real_data"])
        required_columns = ["age_weeks", "pet_weight", "food_intake"]

        if not all(col in real_data.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns in real data: {required_columns}"}), 400

        # Fetch and preprocess research data
        research_data = fetch_research_data()
        research_data = preprocess_research_data(research_data, real_data)

        if research_data.empty:
            return jsonify({"error": "No matching research data found after preprocessing."}), 400

        model = joblib.load(MODEL_PATH)

        real_data["predicted_weight"] = model.predict(real_data[["age_weeks", "food_intake"]])
        research_data["predicted_weight"] = model.predict(research_data[["age_weeks", "food_intake"]])

        # Sort real data by age_weeks to ensure correct plotting order
        real_data = real_data.sort_values(by="age_weeks")

        # Use a valid Matplotlib style
        plt.style.use('ggplot')  # A built-in style

        # Growth Trend Plot
        plt.figure(figsize=(8, 6))
        plt.plot(research_data['age_weeks'], research_data['predicted_weight'], label='Research Data', color='blue', lw=2)
        plt.plot(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        plt.scatter(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        plt.xlabel('Age (weeks)', fontsize=14)
        plt.ylabel('Weight', fontsize=14)
        plt.title('Growth Trend', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        plt.savefig(f"{PLOTS_DIR}/growth_trend.png")
        plt.close()

        # Food Intake Trend Plot
        plt.figure(figsize=(8, 6))
        plt.plot(research_data['age_weeks'], research_data['food_intake'], label='Research Data', color='blue', lw=2)
        plt.plot(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        plt.scatter(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        plt.xlabel('Age (weeks)', fontsize=14)
        plt.ylabel('Food Intake', fontsize=14)
        plt.title('Food Intake Trend', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        plt.savefig(f"{PLOTS_DIR}/food_intake_trend.png")
        plt.close()

        return jsonify({
            "message": "Visualizations generated successfully.",
            "plots": ["growth_trend.png", "food_intake_trend.png"],
            "health_data": real_data[["age_weeks", "pet_weight", "food_intake"]].to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/descriptive', methods=['POST'])
def descriptive_plots():
    """
    Generate descriptive plots for pet data logs and save them as PNG files.
    """
    try:
        # Get and log the raw payload
        payload = request.get_json()
        print("Raw payload received:", payload)

        # Handle payload format
        if isinstance(payload, list):  # Direct list payload
            pet_data_log_descriptive = pd.DataFrame(payload)
        elif isinstance(payload, dict) and "pet_data_log_descriptive" in payload:  # Nested payload
            pet_data_log_descriptive = pd.DataFrame(payload["pet_data_log_descriptive"])
        else:
            return jsonify({"error": "Invalid payload format. Must be a list or contain 'pet_data_log_descriptive' key."}), 400

        # Log converted DataFrame
        print("Converted DataFrame:\n", pet_data_log_descriptive)

        # Rename columns to match expected format
        column_mapping = {
            "ageWeeks": "age_weeks",
            "petWeight": "pet_weight",
            "bowlWeight": "bowl_weight",
            "reservoirHeight": "reservoir_height"
        }
        pet_data_log_descriptive.rename(columns=column_mapping, inplace=True)

        # Check for the timestamp column
        if "timestamp" not in pet_data_log_descriptive.columns:
            print("Warning: 'timestamp' column is missing. Generating synthetic timestamps.")
            pet_data_log_descriptive["timestamp"] = pd.date_range(
                start="2024-01-01", periods=len(pet_data_log_descriptive), freq="H"
            )
        else:
            # Ensure timestamp is in datetime format
            pet_data_log_descriptive["timestamp"] = pd.to_datetime(
                pet_data_log_descriptive["timestamp"], format='%Y-%m-%d %H:%M:%S', errors='coerce'
            )

        # Check for null timestamps
        if pet_data_log_descriptive["timestamp"].isnull().any():
            print("Error: Invalid timestamp format in data.")
            return jsonify({"error": "Invalid timestamp format in data."}), 400

        # Debug log for timestamps
        print("Timestamps in DataFrame:\n", pet_data_log_descriptive["timestamp"])

        # Create plots directory if it doesn't exist
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Generate Scatter Plot: Bowl Weight vs. Pet Weight
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pet_data_log_descriptive['bowl_weight'],
            pet_data_log_descriptive['pet_weight'],
            color='blue', edgecolor='black', alpha=0.7
        )
        plt.xlabel('Bowl Weight (grams)', fontsize=14)
        plt.ylabel('Pet Weight (grams)', fontsize=14)
        plt.title('Bowl Weight vs. Pet Weight', fontsize=16)
        plt.grid(alpha=0.5)
        scatter_plot_path = f"{PLOTS_DIR}/bowl_vs_pet_weight.png"
        plt.savefig(scatter_plot_path)
        plt.close()

        # Generate Bar Chart: Average Pet Weight by Hour of Day
        pet_data_log_descriptive['hour'] = pet_data_log_descriptive['timestamp'].dt.hour
        weight_by_hour = pet_data_log_descriptive.groupby('hour')['pet_weight'].mean()
        plt.figure(figsize=(10, 6))
        weight_by_hour.plot(kind='bar', color='green', edgecolor='black')
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.ylabel('Average Pet Weight (grams)', fontsize=14)
        plt.title('Average Pet Weight by Hour of the Day', fontsize=16)
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(rotation=0)
        bar_chart_path = f"{PLOTS_DIR}/weight_by_hour.png"
        plt.savefig(bar_chart_path)
        plt.close()

        # Generate Combo Plot: Feedings Per Day vs. Average Pet Weight
        pet_data_log_descriptive['date'] = pet_data_log_descriptive['timestamp'].dt.date
        feedings_per_day = pet_data_log_descriptive.groupby('date').size()
        avg_weight_per_day = pet_data_log_descriptive.groupby('date')['pet_weight'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(feedings_per_day.index, feedings_per_day.values, label='Feedings Per Day', color='blue', lw=2)
        plt.plot(avg_weight_per_day.index, avg_weight_per_day.values, label='Average Pet Weight', color='orange', lw=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Count / Weight', fontsize=14)
        plt.title('Feedings Per Day vs. Average Pet Weight', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        combo_plot_path = f"{PLOTS_DIR}/feedings_vs_weight.png"
        plt.savefig(combo_plot_path)
        plt.close()

        return jsonify({
            "message": "Descriptive plots generated successfully.",
            "plots": [
                "bowl_vs_pet_weight.png",
                "weight_by_hour.png",
                "feedings_vs_weight.png"
            ]
        }), 200
    except Exception as e:
        print("Unhandled exception:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
