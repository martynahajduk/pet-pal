import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from flask import Flask, request, jsonify
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


def fetch_research_data():
    """Fetch research data from the Java backend."""
    print("Attempting to fetch research data from Java server...")
    try:
        response = requests.post(JAVA_SERVER_URL, json={},timeout=10)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        research_data = response.json()
        if isinstance(research_data, list):
            return pd.DataFrame(research_data)
        raise ValueError("Research data is not in the expected format (list of dictionaries).")
    except Exception as e:
        print(f"Error in fetch_research_data: {e}")
        raise RuntimeError(f"Failed to fetch research data: {e}")


def preprocess_research_data(research_data, real_data):
    """Preprocess research data to align with real data."""
    min_age = real_data['age_weeks'].min()
    max_age = real_data['age_weeks'].max()

    print(f"Real Data Age Range: {min_age} to {max_age}")
    print(f"Initial Research Data:\n{research_data.head()}")


    # Filter research data
    research_data = research_data[
        (research_data['age_weeks'] >= min_age) & (research_data['age_weeks'] <= max_age)
    ]
    research_data = research_data[['age_weeks', 'pet_weight', 'food_intake']].dropna()
    research_data = research_data.groupby('age_weeks').mean().reset_index()

    all_weeks = np.arange(min_age, max_age + 1)
    research_data = (
        research_data.set_index('age_weeks')
        .reindex(all_weeks)
        .interpolate(method='linear')
        .reset_index()
        .rename(columns={'index': 'age_weeks'})
    )
    print("Research Data After Interpolation:\n", research_data)

    if min_age in real_data['age_weeks'].values:
        real_start_weight = real_data.loc[real_data['age_weeks'] == min_age, 'pet_weight'].values[0]
        real_start_food = real_data.loc[real_data['age_weeks'] == min_age, 'food_intake'].values[0]
        research_start_weight = research_data.loc[research_data['age_weeks'] == min_age, 'pet_weight'].values[0]
        research_start_food = research_data.loc[research_data['age_weeks'] == min_age, 'food_intake'].values[0]

        research_data['pet_weight'] += real_start_weight - research_start_weight
        research_data['food_intake'] += real_start_food - research_start_food

    weight_ratio = real_data['pet_weight'].mean() / research_data['pet_weight'].mean()
    food_ratio = real_data['food_intake'].mean() / research_data['food_intake'].mean()
    research_data['pet_weight'] *= weight_ratio
    research_data['food_intake'] *= food_ratio

    return research_data


@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        required_columns = ["age_weeks", "pet_weight", "food_intake", "breed"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns in research data."}), 400

        # Encode breed as a categorical feature
        df = pd.get_dummies(df, columns=['breed'], drop_first=True)

        X = df.drop(columns=["pet_weight"])
        y = df["pet_weight"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        mse = mean_squared_error(y_test, model.predict(X_test))
        joblib.dump(model, MODEL_PATH)

        return jsonify({"message": "Model trained successfully.", "mse": mse}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# API to generate visualizations and anomalies
@app.route('/api/visualize', methods=['POST'])
def visualize_data():
    try:
        print("\n--- Received a request to /api/visualize ---")
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

        # Debugging: Check processed data
        print("Processed Research Data:\n", research_data.head())

        if research_data.empty:
            return jsonify({"error": "No matching research data found after preprocessing."}), 400

        # Sort data for consistent plotting
        real_data = real_data.sort_values(by="age_weeks")

        # 1. Growth Trend Plot
        growth_anomalies = detect_anomalies(
            real_data['age_weeks'], research_data['pet_weight'], real_data['pet_weight']
        )
        print("Detected Growth Anomalies:", growth_anomalies)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(research_data['age_weeks'], research_data['pet_weight'], label='Research Data', color='blue', lw=2)
        ax.plot(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        ax.scatter(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        ax.set_title('Growth Trend')
        ax.set_xlabel('Age (weeks)')
        ax.set_ylabel('Weight')
        ax.legend()
        growth_trend_base64 = plot_to_base64(fig)
        plt.close(fig)

        growth_conclusion = (
            "The Growth Trend indicates deviations in weight at weeks where real data differs "
            "from research expectations. Anomalies are detected where deviations exceed 10%."
        )

        # 2. Food Intake Trend Plot
        food_anomalies = detect_anomalies(
            real_data['age_weeks'], research_data['food_intake'], real_data['food_intake']
        )

        print("Detected Food Intake Anomalies:", food_anomalies)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(research_data['age_weeks'], research_data['food_intake'], label='Research Data', color='blue', lw=2)
        ax.plot(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Line)', color='orange', linestyle='--', lw=2)
        ax.scatter(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        ax.set_title('Food Intake Trend')
        ax.set_xlabel('Age (weeks)')
        ax.set_ylabel('Food Intake')
        ax.legend()
        food_intake_trend_base64 = plot_to_base64(fig)
        plt.close(fig)

        food_conclusion = (
            "The Food Intake Trend shows significant differences in intake at weeks "
            "where observed values deviate from expected intake based on research data."
        )



        # 3. Scatter Plot: Food Intake vs Pet Weight
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(real_data['food_intake'], real_data['pet_weight'], alpha=0.6, color='purple')
        ax.set_title('Food Intake vs Pet Weight')
        ax.set_xlabel('Food Intake (grams)')
        ax.set_ylabel('Pet Weight (grams)')
        scatter_plot_base64 = plot_to_base64(fig)
        plt.close(fig)

        scatter_conclusion = "Scatter plot shows the relationship between Food Intake and Pet Weight."

        # 4. Bar Chart: Average Pet Weight by Week
        avg_weight = real_data.groupby('age_weeks')['pet_weight'].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_weight.plot(kind='bar', color='green', edgecolor='black', ax=ax)
        ax.set_title('Average Pet Weight by Age (Weeks)')
        ax.set_xlabel('Age (Weeks)')
        ax.set_ylabel('Average Weight')
        bar_chart_base64 = plot_to_base64(fig)
        plt.close(fig)

        bar_chart_conclusion = "Bar chart visualizes the average pet weight per week."

        # 5. Histogram: Food Intake Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(real_data['food_intake'], bins=20, color='orange', edgecolor='black')
        ax.set_title('Food Intake Distribution')
        ax.set_xlabel('Food Intake (grams)')
        ax.set_ylabel('Frequency')
        histogram_base64 = plot_to_base64(fig)
        plt.close(fig)
        histogram_conclusion = "Histogram displays the distribution of food intake values."

        # Return response
        return jsonify({
            "growth_trend_base64": growth_trend_base64,
            "food_intake_trend_base64": food_intake_trend_base64,
            "scatter_plot_base64": scatter_plot_base64,
            "bar_chart_base64": bar_chart_base64,
            "histogram_base64": histogram_base64,
            "growth_trend_conclusion": growth_conclusion,
            "food_intake_trend_conclusion": food_conclusion,
            "scatter_plot_conclusion": scatter_conclusion,
            "bar_chart_conclusion": bar_chart_conclusion,
            "histogram_conclusion": histogram_conclusion,
            "growth_anomalies": growth_anomalies,
            "food_anomalies": food_anomalies
        }), 200

    except Exception as e:
        print("Error in visualize_data:", e)
        return jsonify({"error": str(e)}), 500

def detect_anomalies(weeks, expected_data, actual_data, threshold=0.1):
        """Detect anomalies with debugging."""
        print("Running anomaly detection...")
        anomalies = []
        for week, expected, actual in zip(weeks, expected_data, actual_data):
            deviation = abs(expected - actual) / expected if expected > 0 else 0
            if deviation > threshold:
                anomalies.append({
                    "week": int(week),
                    "expected_weight": round(expected, 2),
                    "actual_weight": round(actual, 2),
                    "deviation": round(deviation * 100, 2)
                })
        print(f"Anomalies detected: {anomalies}")
        return anomalies

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 for response."""
    print("Converting plot to base64...")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
