import logging

from flask import Flask, jsonify, request
from train import AcademicStressPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model
model = AcademicStressPredictor()

# Load trained model on startup
try:
    model.load_model("models/academic_stress_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.warning(f"No trained model found. Training new model... Error: {str(e)}")
    try:
        model.train()
        model.save_model()
    except Exception as train_error:
        logger.error(f"Failed to train model: {str(train_error)}")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_trained": model.is_trained,
            "service": "academic_stress_predictor",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()

        # Validate required fields (using flexible field names)
        field_mapping = {
            "academic_stage": ["Academic_Stage", "stage", "academic_stage"],
            "peer_pressure": ["Peer_Pressure", "peer_pressure", "peers"],
            "family_pressure": ["Family_Pressure", "family_pressure", "family"],
            "study_environment": [
                "Study_Environment",
                "study_environment",
                "environment",
            ],
            "coping_strategy": ["Coping_Strategy", "coping_strategy", "coping"],
            "bad_habits": ["Bad_Habits", "bad_habits", "habits"],
            "academic_competition": [
                "Academic_Competition",
                "academic_competition",
                "competition",
            ],
        }

        # Normalize input data
        normalized_data = {}
        for standard_key, possible_keys in field_mapping.items():
            found = False
            for key in possible_keys:
                if key in data:
                    normalized_data[possible_keys[0]] = data[
                        key
                    ]  # Use first (standard) key
                    found = True
                    break
            if not found:
                # Set default values for missing fields
                if standard_key in [
                    "peer_pressure",
                    "family_pressure",
                    "academic_competition",
                ]:
                    normalized_data[possible_keys[0]] = 3
                else:
                    normalized_data[possible_keys[0]] = "Unknown"

        # Make prediction
        prediction_result = model.predict(normalized_data)

        return jsonify(
            {
                "predicted_stress_level": int(
                    prediction_result["predicted_stress_level"][0]
                ),
                "confidence": float(prediction_result["confidence"][0]),
                "stress_probabilities": prediction_result["probabilities"],
                "input_features": normalized_data,
                "interpretation": get_stress_interpretation(
                    prediction_result["predicted_stress_level"][0]
                ),
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_stress_interpretation(stress_level):
    """Provide interpretation of stress level"""
    interpretations = {
        1: "Very Low Stress - You're managing academic pressures very well!",
        2: "Low Stress - Good stress management with minor concerns",
        3: "Moderate Stress - Some academic pressure but manageable",
        4: "High Stress - Significant academic pressure, consider support",
        5: "Very High Stress - Severe stress levels, seek professional help",
    }
    return interpretations.get(stress_level, "Unknown stress level")


@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain the model"""
    try:
        data_path = request.json.get("data_path", "data/raw/academic_stress_level.csv")
        metrics = model.train(data_path)
        model.save_model()
        return jsonify({"message": "Model retrained successfully", "metrics": metrics})
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    """Get feature importance"""
    if not model.is_trained:
        return jsonify({"error": "Model not trained"}), 400

    try:
        # Get feature importance
        importance_data = []
        for i, feature in enumerate(model.feature_columns):
            importance_data.append(
                {
                    "feature": feature,
                    "importance": float(model.model.feature_importances_[i]),
                }
            )

        # Sort by importance
        importance_data.sort(key=lambda x: x["importance"], reverse=True)

        return jsonify(
            {
                "feature_importance": importance_data,
                "total_features": len(importance_data),
            }
        )
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
