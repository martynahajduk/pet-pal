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
    try:
        response = requests.post(JAVA_SERVER_URL, json={})
        response.raise_for_status()
        research_data = response.json()
        if isinstance(research_data, list):
            return pd.DataFrame(research_data)
        raise ValueError("Research data is not in the expected format (list of dictionaries).")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch research data: {e}")


def preprocess_research_data(research_data, real_data):
    """Preprocess research data to align with real data."""
    min_age = real_data['age_weeks'].min()
    max_age = real_data['age_weeks'].max()

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


def generate_plot_base64(plt_figure):
    """Generate Base64-encoded plot from a matplotlib figure."""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png')
    buf.seek(0)
    encoded_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return encoded_plot


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

        # Helper function for base64 encoding
        def plot_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            encoded_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            return encoded_plot

        # Create Growth Trend Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(research_data['age_weeks'], research_data['pet_weight'], label='Research Data', color='blue', lw=2)
        ax.plot(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        ax.scatter(real_data['age_weeks'], real_data['pet_weight'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        ax.set_xlabel('Age (weeks)', fontsize=14)
        ax.set_ylabel('Weight', fontsize=14)
        ax.set_title('Growth Trend', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.5)
        growth_trend_base64 = plot_to_base64(fig)
        plt.close(fig)

        # Create Food Intake Trend Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(research_data['age_weeks'], research_data['food_intake'], label='Research Data', color='blue', lw=2)
        ax.plot(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Line)', color='orange', lw=2, linestyle='--')
        ax.scatter(real_data['age_weeks'], real_data['food_intake'], label='Real Data (Points)', color='orange', edgecolor='black', s=80)
        ax.set_xlabel('Age (weeks)', fontsize=14)
        ax.set_ylabel('Food Intake', fontsize=14)
        ax.set_title('Food Intake Trend', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.5)
        food_intake_trend_base64 = plot_to_base64(fig)
        plt.close(fig)

        # Scatter Plot: Bowl Weight vs. Pet Weight
        if "food_intake" in real_data.columns:
            corr = real_data['food_intake'].corr(real_data['pet_weight'])
            scatter_plot_conclusion = (
                "The scatter plot shows the relationship between the amount of food provided (in grams) and the pet's weight. "
                f"The calculated correlation is {corr:.2f}, indicating "
                + ("a strong positive relationship between food portions and pet weight." if corr > 0.7 else
                   "a weak or no significant relationship between food portions and pet weight.")
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(real_data['food_intake'], real_data['pet_weight'], alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Food Portion (grams)', fontsize=14)
            ax.set_ylabel('Pet Weight (grams)', fontsize=14)
            ax.set_title('Food Portion vs. Pet Weight', fontsize=16)
            scatter_plot_base64 = plot_to_base64(fig)
            plt.close(fig)
        else:
            scatter_plot_conclusion = "Insufficient data to generate a scatter plot."
            scatter_plot_base64 = None

        # Bar Chart: Average Pet Weight by Age in Weeks
        avg_weight_by_age_week = real_data.groupby('age_weeks')['pet_weight'].mean()
        max_weight_age = avg_weight_by_age_week.idxmax()
        bar_chart_conclusion = (
            f"The bar chart illustrates the average pet weight across different weeks of age. "
            f"The highest average weight occurs at week {max_weight_age}, where pets weigh approximately "
            f"{avg_weight_by_age_week[max_weight_age]:.2f} grams. Younger pets tend to weigh less, "
            "while the weight increases steadily as they grow older."
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_weight_by_age_week.plot(kind='bar', ax=ax, color='green', edgecolor='black')
        ax.set_xlabel('Age in Weeks', fontsize=14)
        ax.set_ylabel('Average Pet Weight (grams)', fontsize=14)
        ax.set_title('Average Pet Weight by Age', fontsize=16)
        ax.grid(axis='y', alpha=0.5)
        ax.tick_params(axis='x', rotation=0)
        bar_chart_base64 = plot_to_base64(fig)
        plt.close(fig)

        # Heat Map: Age, Food Intake, Pet Weight
        heatmap_data = real_data.pivot(index='age_weeks', columns='food_intake', values='pet_weight')
        max_weight = heatmap_data.max().max()
        max_weight_position = heatmap_data.stack().idxmax()
        heatmap_conclusion = (
            f"The heatmap visualizes the weight of pets based on their age (in weeks) and food portions (in grams). "
            f"The highest observed weight is {max_weight:.2f} grams, which occurs at age {max_weight_position[0]} weeks "
            f"when the food portion is {max_weight_position[1]} grams. This chart helps to identify optimal feeding "
            "portions for pets at various stages of growth."
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.1f', cbar_kws={'label': 'Pet Weight (grams)'})
        ax.set_title('Heatmap: Age, Food Intake, and Pet Weight', fontsize=16)
        ax.set_xlabel('Food Intake (grams)', fontsize=14)
        ax.set_ylabel('Age (weeks)', fontsize=14)
        heatmap_base64 = plot_to_base64(fig)
        plt.close(fig)

        # Construct verdict based on comparison logic
        verdict_data = []
        for _, row in real_data.iterrows():
            research_weight = research_data.loc[research_data["age_weeks"] == row["age_weeks"], "pet_weight"].values[0]
            health_status = "Healthy" if row["pet_weight"] >= research_weight * 0.9 else "Unhealthy"
            verdict_data.append({"age_weeks": row["age_weeks"], "health_status": health_status})



        # Send response with base64-encoded plots, conclusions, and verdicts
        return jsonify({
            "message": "Visualization and descriptive plots generated successfully.",
            "growth_trend_base64": growth_trend_base64,
            "food_intake_trend_base64": food_intake_trend_base64,
            "scatter_plot_base64": scatter_plot_base64,
            "bar_chart_base64": bar_chart_base64,
            "heatmap_base64": heatmap_base64,
            "verdict_data": verdict_data,
            "scatter_plot_conclusion": scatter_plot_conclusion,
            "bar_chart_conclusion": bar_chart_conclusion,
            "heatmap_conclusion": heatmap_conclusion
        }), 200

    except Exception as e:
        print("Error in visualize_data:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
