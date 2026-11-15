import json

import numpy as np
import pytest

from app.features import FEATURES, add_features
from app.model import ModelService, pd_to_score
from app.monitoring import bin_shares, psi
from training import pipeline as P


def test_validate_accepts_good_and_rejects_bad(frame):
    assert P.validate(frame)["rows"] == len(frame)
    with pytest.raises(P.DataValidationError, match="AGE"):
        P.validate(frame.assign(AGE=5))
    with pytest.raises(P.DataValidationError, match="нет колонок"):
        P.validate(frame.drop(columns="PAY_1"))
    with pytest.raises(P.DataValidationError, match="бинарная"):
        P.validate(frame.assign(default=2))


def test_split_is_stratified_and_disjoint(frame):
    train, valid, test = P.split(frame)
    assert len(train) + len(valid) + len(test) == len(frame)
    assert not set(train.index) & set(test.index)
    assert abs(train["default"].mean() - test["default"].mean()) < 0.02


def test_train_evaluate_save_load_roundtrip(frame, tmp_path):
    train, valid, test = P.split(frame)
    booster = P.train(train, valid, rounds=200)
    evaluation = P.evaluate(booster, valid, test)
    assert evaluation["metrics"]["roc_auc"] > 0.65
    meta = P.save(booster, evaluation, P.reference_stats(train, booster), frame, len(train), tmp_path)
    service = ModelService.load(tmp_path)
    preds = service.predict(test.head(5))
    np.testing.assert_allclose([p["pd"] for p in preds], booster.predict(add_features(test.head(5))[FEATURES]), rtol=1e-6)
    assert service.version == meta["version"]


def test_champion_challenger(tmp_path):
    assert P.compare_with_production({"roc_auc": 0.7}, tmp_path)[0]  # прод пустой
    (tmp_path / "metadata.json").write_text(json.dumps({"metrics": {"roc_auc": 0.78}}), encoding="utf-8")
    assert P.compare_with_production({"roc_auc": 0.778}, tmp_path)[0]       # в пределах допуска
    assert not P.compare_with_production({"roc_auc": 0.76}, tmp_path)[0]


def test_best_threshold_maximizes_profit():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    assert 0.3 < P.best_threshold(y, p) < 0.6  # одобряем ровно трёх хороших


def test_psi():
    shares = [0.25, 0.25, 0.25, 0.25]
    assert psi(shares, shares) == pytest.approx(0)
    assert psi(shares, [0.7, 0.1, 0.1, 0.1]) > 0.25
    np.testing.assert_allclose(bin_shares(np.array([0, 1, 2, 3]), [1.5]), [0.5, 0.5])


def test_score_scale():
    assert pd_to_score(1 / 51) == 600              # шансы 50:1
    assert pd_to_score(1 / 101) - pd_to_score(1 / 51) == pytest.approx(20, abs=1)  # шансы вдвое лучше → +20


def test_committed_artifact_matches_code():
    service = ModelService.load(P.Path(__file__).resolve().parents[1] / "artifacts")
    assert service.booster.feature_name() == FEATURES
    assert set(service.metadata["reference"]) == {*P.DRIFT_FEATURES, "pd"}
