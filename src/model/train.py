# =============================================================================
# ML Model Implementation for Academic Stress Level Prediction - CI/CD Pipeline
# =============================================================================

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcademicStressPredictor:
    """
    Academic Stress Level Prediction Model for CI/CD Pipeline Demo

    This model predicts academic stress levels based on:
    - Academic Stage (undergraduate, high school, etc.)
    - Peer Pressure (1-5 scale)
    - Family Pressure (1-5 scale)
    - Study Environment (Peaceful, disrupted, etc.)
    - Coping Strategy (Intellectual, Emotional, etc.)
    - Bad Habits (Yes/No)
    - Academic Competition (1-5 scale)

    Target: Stress Level (1-5 scale)
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.stress_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = []

    def load_data(self, data_path="data/raw/academic_stress_level.csv"):
        """Load and validate dataset"""
        try:
            df = pd.read_csv(data_path)
            logger.info(
                f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns"
            )

            # Clean column names (remove extra spaces, standardize)
            df.columns = df.columns.str.strip()

            # Map expected column names from the dataset description
            column_mapping = {
                "Your Academic Stage": "Academic_Stage",
                "Peer pressure": "Peer_Pressure",
                "Academic pressure from your home": "Family_Pressure",
                "Study Environment": "Study_Environment",
                "What coping strategy you use as a student?": "Coping_Strategy",
                "Do you have any bad habits like smoking, drinking on a daily basis?": "Bad_Habits",
                "What would you rate the academic competition in your student life": "Academic_Competition",
                "Rate your academic stress index": "Stress_Level",
            }

            # Try to find and rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
                elif any(old_name.lower() in col.lower() for col in df.columns):
                    # Fuzzy matching for similar column names
                    matching_col = next(
                        col for col in df.columns if old_name.lower() in col.lower()
                    )
                    df = df.rename(columns={matching_col: new_name})

            return df

        except FileNotFoundError:
            logger.error(f"Dataset not found at {data_path}")
            # Generate sample data for demo purposes if real data not available
            return self._generate_sample_data()

    def _generate_sample_data(self, n_samples=500):
        """Generate sample academic stress dataset for demonstration"""
        np.random.seed(42)

        # Academic stages
        stages = np.random.choice(
            ["undergraduate", "high school", "graduate"], n_samples, p=[0.7, 0.25, 0.05]
        )

        # Pressure levels (1-5 scale)
        peer_pressure = np.random.choice(
            [1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]
        )
        family_pressure = np.random.choice(
            [1, 2, 3, 4, 5], n_samples, p=[0.12, 0.18, 0.35, 0.25, 0.1]
        )

        # Study environment
        environments = np.random.choice(
            ["Peaceful", "disrupted", "moderate"], n_samples, p=[0.5, 0.3, 0.2]
        )

        # Coping strategies
        coping = np.random.choice(
            [
                "Analyze the situation and handle it with intellect",
                "Emotional breakdown (crying a lot)",
                "Physical exercise",
            ],
            n_samples,
            p=[0.6, 0.25, 0.15],
        )

        # Bad habits
        bad_habits = np.random.choice(["No", "Yes"], n_samples, p=[0.85, 0.15])

        # Academic competition (1-5)
        competition = np.random.choice(
            [1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.3, 0.4, 0.1]
        )

        # Generate stress levels based on other factors (realistic relationships)
        stress_levels = []
        for i in range(n_samples):
            # Base stress influenced by pressures and environment
            base_stress = (peer_pressure[i] + family_pressure[i] + competition[i]) / 3

            # Environment adjustments
            if environments[i] == "disrupted":
                base_stress += 0.5
            elif environments[i] == "Peaceful":
                base_stress -= 0.3

            # Coping strategy adjustments
            if "intellect" in coping[i]:
                base_stress -= 0.4
            elif "breakdown" in coping[i]:
                base_stress += 0.6

            # Bad habits increase stress
            if bad_habits[i] == "Yes":
                base_stress += 0.5

            # Add some randomness
            base_stress += np.random.normal(0, 0.3)

            # Convert to 1-5 scale
            stress_level = max(1, min(5, round(base_stress)))
            stress_levels.append(stress_level)

        df = pd.DataFrame(
            {
                "Academic_Stage": stages,
                "Peer_Pressure": peer_pressure,
                "Family_Pressure": family_pressure,
                "Study_Environment": environments,
                "Coping_Strategy": coping,
                "Bad_Habits": bad_habits,
                "Academic_Competition": competition,
                "Stress_Level": stress_levels,
            }
        )

        logger.info("Generated sample academic stress dataset for demonstration")
        return df

    def preprocess_data(self, df):
        """Preprocess the dataset"""
        df = df.copy()

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                )
            else:
                df[col] = df[col].fillna(df[col].median())

        # Separate features and target
        target_col = "Stress_Level"
        feature_cols = [
            col for col in df.columns if col != target_col and col != "Timestamp"
        ]

        X = df[feature_cols].copy()
        y = df[target_col] if target_col in df.columns else None

        # Store feature columns for prediction
        self.feature_columns = feature_cols

        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=["object"]).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories during prediction
                X[col] = X[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else "Unknown")
                try:
                    X[col] = self.label_encoders[col].transform(X[col])
                except ValueError:
                    # If still problematic, assign a default value
                    X[col] = 0

        # Encode target variable if it exists
        if y is not None:
            if not hasattr(self.stress_encoder, "classes_"):
                y = self.stress_encoder.fit_transform(y)
            else:
                y = self.stress_encoder.transform(y)

        return X, y

    def train(self, data_path="data/raw/academic_stress_level.csv"):
        """Train the model"""
        logger.info("Starting model training...")

        # Load and preprocess data
        df = self.load_data(data_path)
        X, y = self.preprocess_data(df)

        if y is None:
            raise ValueError("No target variable found in dataset")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        logger.info(f"Training Accuracy: {train_accuracy:.3f}")
        logger.info(f"Test Accuracy: {test_accuracy:.3f}")
        logger.info(
            f"Cross-validation Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info("Top 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        self.is_trained = True
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_score": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": feature_importance.to_dict("records"),
        }

    def predict(self, features):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure features is a DataFrame with correct columns
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame(features)

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features.columns:
                # Set default values for missing features
                if col in ["Peer_Pressure", "Family_Pressure", "Academic_Competition"]:
                    features[col] = 3  # Neutral value
                else:
                    features[col] = "Unknown"

        # Select only the features used in training
        features = features[self.feature_columns]

        # Preprocess features
        features_processed, _ = self.preprocess_data(features)
        features_scaled = self.scaler.transform(features_processed)

        # Make prediction
        predictions = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)

        # Convert back to original stress levels
        stress_levels = self.stress_encoder.inverse_transform(predictions)

        return {
            "predicted_stress_level": stress_levels,
            "confidence": np.max(prediction_proba, axis=1),
            "probabilities": {
                str(self.stress_encoder.inverse_transform([i])[0]): prob
                for i, prob in enumerate(prediction_proba[0])
            },
        }

    def save_model(self, model_path="models/academic_stress_model.pkl"):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "stress_encoder": self.stress_encoder,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path="models/academic_stress_model.pkl"):
        """Load a trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoders = model_data["label_encoders"]
            self.stress_encoder = model_data["stress_encoder"]
            self.feature_columns = model_data["feature_columns"]
            self.is_trained = model_data["is_trained"]
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            raise
