import base64
import io
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from flask import Flask, request, jsonify
from scipy.stats import linregress
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


        required_columns = ["age_weeks", "pet_weight", "food_intake"]
        # Fetch and preprocess research data
        research_data = fetch_research_data()

        # Debugging: Check processed data
        print("Processed Research Data:\n", research_data.head())

        if research_data.empty:
            return jsonify({"error": "No matching research data found after preprocessing."}), 400


        # Extract real data from payload
        real_data = payload['real_data']

        returnList = {}
        for item in real_data :
            # innerList = {}
            df = pd.DataFrame(real_data[item])

            if not all(col in df.columns for col in required_columns):
                returnList[str(item)] = {
                    "hasData": False,
                    "growth_trend_base64": None,
                    "food_intake_trend_base64": None,
                    "scatter_plot_base64": None,
                    "bar_chart_base64": None,
                    "histogram_base64": None,
                    "growth_trend_conclusion": None,
                    "food_intake_trend_conclusion": None,
                    "scatter_plot_conclusion": None,
                    "bar_chart_conclusion": None,
                    "histogram_conclusion": None,
                    "growth_anomalies": None,
                    "food_anomalies": None
                }
                continue

            processed_data = preprocess_research_data(research_data, df)
            df = df.sort_values(by="age_weeks")
            print(item)
            print("Parsed Real Data:\n", df.head())  # Debug

            df.drop(columns=['breed'], inplace=True)

            if not all(col in df.columns for col in required_columns):
                return jsonify({"error": f"Missing required columns in real data: {required_columns}"}), 400

            # df = df.groupby(['age_weeks'])['pet_weight', 'food_intake'].mean()

            # Sort data for consistent plotting
            # df = df.sort_values(by="age_weeks")

            # 1. Growth Trend Plot
            growth_anomalies = detect_anomalies(
                df['age_weeks'], processed_data['pet_weight'], df['pet_weight']
            )
            print("Detected Growth Anomalies:", growth_anomalies)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(processed_data['age_weeks'], processed_data['pet_weight'], label='Research Data', color='blue', lw=2)
            ax.plot(df['age_weeks'], df['pet_weight'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
            ax.scatter(df['age_weeks'], df['pet_weight'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
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
                df['age_weeks'], processed_data['food_intake'], df['food_intake']
            )

            print("Detected Food Intake Anomalies:", food_anomalies)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(processed_data['age_weeks'], processed_data['food_intake'], label='Research Data', color='blue', lw=2)
            ax.plot(df['age_weeks'], df['food_intake'], label='Real Data (Line)', color='orange', linestyle='--', lw=2)
            ax.scatter(df['age_weeks'], df['food_intake'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
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

            # Apply square root transformation
            df['sqrt_pet_weight'] = np.sqrt(real_data['pet_weight'])
            df['sqrt_food_intake'] = np.sqrt(real_data['food_intake'])

        # 3. Scatter Plot: Square Root of Pet Weight vs Square Root of Food Intake
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create scatter plot with larger marker size
        ax.scatter(
            real_data['sqrt_pet_weight'],
            real_data['sqrt_food_intake'],
            alpha=0.5,  # Slightly less transparency to make points more visible
            s=50,  # Increased marker size for higher density perception
            color='purple',
            label='Data Points'
        )

        # Calculate regression line (Intake as a function of Pet Weight)
        slope, intercept, _, _, _ = linregress(real_data['sqrt_pet_weight'], real_data['sqrt_food_intake'])
        regression_line = slope * real_data['sqrt_pet_weight'] + intercept

        # Add regression line to plot
        ax.plot(real_data['sqrt_pet_weight'], regression_line, color='red', label='Regression Line')

        # Add labels, title, and legend
        ax.set_title('Square Root of Pet Weight vs Square Root of Food Intake with Regression Line')
        ax.set_xlabel('Square Root of Pet Weight (grams)')
        ax.set_ylabel('Square Root of Food Intake (grams)')
        ax.legend()

        # Convert plot to Base64
        scatter_plot_base64 = plot_to_base64(fig)

        # Close the figure
        plt.close(fig)

        # Updated conclusion
        scatter_conclusion = (
            "The scatter plot visualizes the square root relationship between pet weight and food intake. "
            "Larger markers improve the visibility of individual data points, and a regression line highlights the trend, making it easier to interpret the transformed relationship."
        )
        # 4. Histogram: Average Pet Weight by Week
        avg_weight = real_data.groupby('age_weeks')['pet_weight'].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(avg_weight, bins=10, color='green', edgecolor='black')
        ax.set_title('Average Pet Weight Distribution by Age (Weeks)')
        ax.set_xlabel('Average Weight')
        ax.set_ylabel('Frequency')
        histogram_base64_avg_weight = plot_to_base64(fig)
        plt.close(fig)

        histogram_conclusion_avg_weight = "Histogram displays the distribution of average pet weight per week."

        # 5. Bar Chart: Food Intake Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        real_data['food_intake'].value_counts().sort_index().plot(kind='bar', color='orange', edgecolor='black',
                                                                  ax=ax)  # Using bar chart instead of histogram
        ax.set_title('Food Intake Distribution')
        ax.set_xlabel('Food Intake (grams)')
        ax.set_ylabel('Frequency')
        bar_chart_base64_food_intake = plot_to_base64(fig)
        plt.close(fig)

        bar_chart_conclusion_food_intake = "Bar chart visualizes the frequency of food intake values."

        returnList[str(item)] = {
            "hasData": True,
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
        }

        # Return response
        # print(returnList)
        return returnList

    except Exception as e:
        print("Error in visualize_data:", e.with_traceback())
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img_base64


if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
