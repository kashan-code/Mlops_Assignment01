import pytest

from src.api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_index_page(client):
    """Test main index page"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Academic Stress Level Predictor" in response.data
