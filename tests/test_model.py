import numpy as np
import pandas as pd
from src.model.train import AcademicStressPredictor


def test_model_initialization():
    """Test that model initializes correctly"""
    model = AcademicStressPredictor()
    assert model is not None
    assert model.is_trained is False


def test_data_generation():
    """Test sample data generation"""
    model = AcademicStressPredictor()
    df = model._generate_sample_data(100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "Stress_Level" in df.columns
    assert df["Stress_Level"].min() >= 1
    assert df["Stress_Level"].max() <= 5


def test_model_training():
    """Test model training functionality"""
    model = AcademicStressPredictor()
    metrics = model.train()
    assert model.is_trained is True
    assert "train_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert 0 <= metrics["train_accuracy"] <= 1
    assert 0 <= metrics["test_accuracy"] <= 1


def test_prediction():
    """Test prediction functionality with sample data"""
    model = AcademicStressPredictor()
    # Train the model first
    model.train()

    # Test prediction with sample data
    sample_data = {
        "Academic_Stage": "undergraduate",
        "Peer_Pressure": 3,
        "Family_Pressure": 3,
        "Study_Environment": "Peaceful",
        "Coping_Strategy": "Analyze the situation and handle it with intellect",
        "Bad_Habits": "No",
        "Academic_Competition": 3,
    }

    result = model.predict(sample_data)
    assert "predicted_stress_level" in result
    assert "confidence" in result
    assert isinstance(result["predicted_stress_level"][0], (int, np.integer))
    assert 1 <= result["predicted_stress_level"][0] <= 5
    assert 0 <= result["confidence"][0] <= 1


def test_model_save_load():
    """Test model saving and loading"""
    model = AcademicStressPredictor()
    model.train()

    # Save model
    model.save_model("models/test_model.pkl")

    # Load model
    new_model = AcademicStressPredictor()
    new_model.load_model("models/test_model.pkl")

    assert new_model.is_trained is True
    assert len(new_model.feature_columns) > 0
