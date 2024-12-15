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
import seaborn as sns
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
JAVA_SERVER_URL = "http://localhost:80/api/train"
MODEL_PATH = "model/trained_model.pkl"
PLOTS_DIR = os.path.join(app.root_path, 'plots')
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
    - Automatically align start point with the earliest week in real_data
    - Interpolate missing weeks
    - Scale research data to match the real data trend
    """
    print(f"Initial research data shape: {research_data.shape}")
    print(f"Initial real data shape: {real_data.shape}")

    # Automatically determine the minimum and maximum weeks
    min_age = real_data['age_weeks'].min()
    max_age = real_data['age_weeks'].max()
    print(f"Aligning research data to real data age range: {min_age} to {max_age}")

    # Filter research data to the matching age range
    research_data = research_data[(research_data['age_weeks'] >= min_age) & (research_data['age_weeks'] <= max_age)]

    # Ensure numeric columns are used for processing
    numeric_columns = ['age_weeks', 'pet_weight', 'food_intake']
    research_data = research_data[numeric_columns].dropna()

    # Group by age_weeks and calculate mean to handle duplicates
    research_data = research_data.groupby('age_weeks').mean().reset_index()

    # Interpolate missing weeks
    all_weeks = np.arange(min_age, max_age + 1)
    research_data = (
        research_data.set_index('age_weeks')
        .reindex(all_weeks)
        .interpolate(method='linear')  # Fill missing data
        .reset_index()
        .rename(columns={'index': 'age_weeks'})
    )

    # Align starting point to match real data at the minimum week
    start_week = min_age
    if start_week in real_data['age_weeks'].values and start_week in research_data['age_weeks'].values:
        real_start_weight = real_data.loc[real_data['age_weeks'] == start_week, 'pet_weight'].values[0]
        real_start_food = real_data.loc[real_data['age_weeks'] == start_week, 'food_intake'].values[0]

        research_start_weight = research_data.loc[research_data['age_weeks'] == start_week, 'pet_weight'].values[0]
        research_start_food = research_data.loc[research_data['age_weeks'] == start_week, 'food_intake'].values[0]

        # Adjust the research data to align with the real data at the start week
        weight_offset = real_start_weight - research_start_weight
        food_offset = real_start_food - research_start_food
        research_data['pet_weight'] += weight_offset
        research_data['food_intake'] += food_offset
        print(f"Aligned research data to real data at week {start_week}: weight offset {weight_offset}, food offset {food_offset}")

    # Scale research data to match the overall trend of real data
    weight_ratio = real_data['pet_weight'].mean() / research_data['pet_weight'].mean()
    food_ratio = real_data['food_intake'].mean() / research_data['food_intake'].mean()
    research_data['pet_weight'] *= weight_ratio
    research_data['food_intake'] *= food_ratio
    print(f"Scaled research data with weight ratio {weight_ratio} and food ratio {food_ratio}")

    print(f"Preprocessed research data shape: {research_data.shape}")
    return research_data

@app.route('/plots/<filename>')
def serve_plot(filename):
    file_path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return "File not found", 404
    print(f"Serving plot: {file_path}")
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
        print("Received Payload:", payload)  # Debug
        if not isinstance(payload, dict) or "real_data" not in payload:
            return jsonify({"error": "Invalid payload format. Expected a dictionary with key 'real_data'."}), 400

        # Extract real data from payload
        real_data = pd.DataFrame(payload["real_data"])
        print("Parsed Real Data:", real_data.head())  # Debug
        required_columns = ["age_weeks", "pet_weight", "food_intake"]

        if not all(col in real_data.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns in real data: {required_columns}"}), 400

        # Fetch and preprocess research data
        research_data = fetch_research_data()
        research_data = preprocess_research_data(research_data, real_data)

        if research_data.empty:
            return jsonify({"error": "No matching research data found after preprocessing."}), 400

        # Sort data for consistent plotting
        real_data = real_data.sort_values(by="age_weeks")

        # Create Growth Trend Plot
        plt.figure(figsize=(8, 6))
        plt.plot(research_data['age_weeks'], research_data['pet_weight'], label='Research Data', color='blue', lw=2)
        plt.plot(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        plt.scatter(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        plt.xlabel('Age (weeks)', fontsize=14)
        plt.ylabel('Weight', fontsize=14)
        plt.title('Growth Trend', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        growth_trend_path = f"{PLOTS_DIR}/growth_trend.png"
        plt.savefig(growth_trend_path)
        print(f"Saved Growth Trend at {growth_trend_path}")
        plt.close()

        # Create Food Intake Trend Plot
        plt.figure(figsize=(8, 6))
        plt.plot(research_data['age_weeks'], research_data['food_intake'], label='Research Data', color='blue', lw=2)
        plt.plot(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        plt.scatter(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        plt.xlabel('Age (weeks)', fontsize=14)
        plt.ylabel('Food Intake', fontsize=14)
        plt.title('Food Intake Trend', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        food_intake_trend_path = f"{PLOTS_DIR}/food_intake_trend.png"
        plt.savefig(food_intake_trend_path)
        print(f"Saved Growth Trend at {food_intake_trend_path}")
        plt.close()

        # Construct verdict based on comparison logic
        verdict_data = []
        for _, row in real_data.iterrows():
            research_weight = research_data.loc[research_data["age_weeks"] == row["age_weeks"], "pet_weight"].values[0]
            health_status = "Healthy" if row["pet_weight"] >= research_weight * 0.9 else "Unhealthy"
            verdict_data.append({"age_weeks": row["age_weeks"], "health_status": health_status})

        # Send response with plot paths and verdicts
        return jsonify({
            "message": "Visualization generated successfully.",
            "plots": ["growth_trend.png", "food_intake_trend.png"],

            "verdict_data": verdict_data
        }), 200

    except Exception as e:
        print("Error in visualize_data:", e)
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


        # Scatter Plot: Bowl Weight vs. Pet Weight
        corr = pet_data_log_descriptive['bowl_weight'].corr(pet_data_log_descriptive['pet_weight'])
        scatter_plot_conclusion = (
            "The scatter plot shows the relationship between the amount of food provided (in grams) and the pet's weight. "
            f"The calculated correlation is {corr:.2f}, indicating "
            + ("a strong positive relationship between food portions and pet weight." if corr > 0.7 else
               "a weak or no significant relationship between food portions and pet weight.")
        )
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pet_data_log_descriptive['bowl_weight'],
            pet_data_log_descriptive['pet_weight'],
            color='blue', edgecolor='black', alpha=0.7
        )
        plt.xlabel('Food Portion (grams)', fontsize=14)
        plt.ylabel('Pet Weight (grams)', fontsize=14)
        plt.title('Food Portion vs. Pet Weight', fontsize=16)
        plt.grid(alpha=0.5)
        scatter_plot_path = f"{PLOTS_DIR}/food_portion_vs_pet_weight.png"
        plt.savefig(scatter_plot_path)
        plt.close()

        # Bar Chart: Average Pet Weight by Age in Weeks
        avg_weight_by_age_week = pet_data_log_descriptive.groupby('age_weeks')['pet_weight'].mean()
        max_weight_age = avg_weight_by_age_week.idxmax()
        bar_chart_conclusion = (
            f"The bar chart illustrates the average pet weight across different weeks of age. "
            f"The highest average weight occurs at week {max_weight_age}, where pets weigh approximately "
            f"{avg_weight_by_age_week[max_weight_age]:.2f} grams. Younger pets tend to weigh less, "
            f"while the weight increases steadily as they grow older."
        )
        plt.figure(figsize=(10, 6))
        avg_weight_by_age_week.plot(kind='bar', color='green', edgecolor='black')
        plt.xlabel('Age in Weeks', fontsize=14)
        plt.ylabel('Average Pet Weight (grams)', fontsize=14)
        plt.title('Average Pet Weight by Age in Weeks', fontsize=16)
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(rotation=0)

        bar_chart_path = f"{PLOTS_DIR}/weight_by_age_week.png"
        plt.savefig(bar_chart_path)
        plt.close()

        # Heat Map: Age, Food Portion, Pet Weight
        heatmap_data = pet_data_log_descriptive.pivot(
            index='age_weeks', columns='bowl_weight', values='pet_weight'
        )
        max_weight = heatmap_data.max().max()
        max_weight_position = heatmap_data.stack().idxmax()
        heatmap_conclusion = (
            "The heatmap visualizes the weight of pets based on their age (in weeks) and food portions (in grams). "
            f"The highest observed weight is {max_weight:.2f} grams, which occurs at age {max_weight_position[0]} weeks "
            f"when the food portion is {max_weight_position[1]} grams. This chart helps to identify optimal feeding "
            "portions for pets at various stages of growth."
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.1f', cbar_kws={'label': 'Pet Weight (grams)'})
        plt.title('Heatmap: Age, Food Portion, and Pet Weight', fontsize=16)
        plt.xlabel('Food Portion (grams)', fontsize=14)
        plt.ylabel('Age (weeks)', fontsize=14)

        heatmap_path = f"{PLOTS_DIR}/heatmap_age_food_weight.png"
        plt.savefig(heatmap_path)
        plt.close()

        # Combine conclusions
        conclusions = {
            "scatter_plot": scatter_plot_conclusion,
            "bar_chart": bar_chart_conclusion,
            "heatmap": heatmap_conclusion
        }

        return jsonify({
            "message": "Descriptive plots generated successfully.",
            "plots": [
                "food_portion_vs_pet_weight.png",
                "weight_by_age_week.png",
                "heatmap_age_food_weight.png"
            ],
            "conclusions": conclusions
        }), 200
    except Exception as e:
        print("Unhandled exception:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
