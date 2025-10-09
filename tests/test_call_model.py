import io
import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


@patch("app.predict_image", return_value=("Bánh mì ngọt", 0.95))
def test_call_model_success(mock_predict, client):
    # Tạo giả file ảnh
    data = {
        "file": (io.BytesIO(b"fake image data"), "test.jpg")
    }
    response = client.post("/callModel", content_type="multipart/form-data", data=data)
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Success"
    assert data["productName"] == "Bánh mì ngọt"
    assert data["accuracy"] == 0.95


def test_call_model_missing_file(client):
    response = client.post("/callModel", content_type="multipart/form-data", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["message"] == "Failed"
