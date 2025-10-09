import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

@patch("services.image_service.is_image_nsfw", return_value=(False, {}))
@patch("services.image_service.is_image_violent", return_value=(False, "Non-Violent", 99.0))
@patch("models.toxic.toxic_pipeline", return_value=[{"label": "non-toxic", "score": 0.1}])
def test_review_label(mock_toxic, mock_violent, mock_nsfw, client):
    response = client.post("/reviewLabel", json={
        "review": "Món ăn ngon",
        "images": {"img1.jpg": "base64string"}
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert "images" in data
