import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


@patch("services.image_service.is_image_nsfw", return_value=(True, {"class": "GENITALIA_EXPOSED", "score": 0.9}))
@patch("services.image_service.is_image_violent", return_value=(True, "Violent", 95.0))
@patch("models.toxic.toxic_pipeline", return_value=[{"label": "toxic", "score": 0.9}])
@patch("base64.b64decode", return_value=b"fakebytes")
def test_review_label_toxic_and_nsfw(mock_toxic, mock_violent, mock_nsfw, client):
    response = client.post("/reviewLabel", json={
        "review": "You are a stupid fuck!",
        "images": {"bad.jpg": "fakebase64"}
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["label"] == "toxic"
    assert data["images"][0]["nsfw"] is True
    assert data["images"][0]["violent"] is True
