from app.schemas import EXAMPLE


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["model_version"]


def test_model_info(client):
    info = client.get("/model").json()
    assert 0.7 < info["metrics"]["roc_auc"] < 0.9
    assert 0 < info["threshold"] < 1


def test_predict_example(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["pd"] < 1
    assert body["decision"] in ("approve", "reject")
    assert (body["decision"] == "approve") == (body["pd"] < body["threshold"])
    assert all(r["contribution"] > 0 for r in body["reasons"])


def test_risk_grows_with_delay(client):
    good = {**EXAMPLE, "PAY_1": -1, "PAY_2": -1}
    bad = {**EXAMPLE, "PAY_1": 3, "PAY_2": 2}
    assert client.post("/predict", json=bad).json()["pd"] > client.post("/predict", json=good).json()["pd"]


def test_validation_errors(client):
    assert client.post("/predict", json={**EXAMPLE, "AGE": 10}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "PAY_1": 15}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "unknown": 1}).status_code == 422
    missing = dict(EXAMPLE)
    missing.pop("LIMIT_BAL")
    assert client.post("/predict", json=missing).status_code == 422


def test_batch(client):
    response = client.post("/predict/batch", json={"applications": [EXAMPLE, {**EXAMPLE, "PAY_1": 2}]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=EXAMPLE)
    text = client.get("/metrics").text
    assert "scoring_requests_total" in text and "scoring_pd_bucket" in text


def test_drift_detects_shift(client):
    assert client.get("/drift").json()["status"] == "not_enough_data"
    shifted = [{**EXAMPLE, "PAY_1": 3, "PAY_2": 3, "LIMIT_BAL": 10_000} for _ in range(60)]
    client.post("/predict/batch", json={"applications": shifted})
    report = client.get("/drift").json()
    assert report["status"] == "alert"
    assert report["features"]["PAY_1"]["level"] == "alert"
