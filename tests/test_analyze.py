import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


@patch("app.sent_pipeline", return_value=[{"label": "POS"}])
def test_analyze_positive(mock_sentiment, client):
    response = client.post("/analyze", json={
        "review": ["Món ăn rất ngon"],
        "reviewID": [1]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert data[0]["reviewID"] == 1
    assert "summary" in data[0]
    assert "clauses" in data[0]
    assert "analysis" in data[0]


def test_analyze_invalid_json(client):
    response = client.post("/analyze", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
