import pytest

from app.schemas import Application
from bot.bot import ParseError, format_prediction, parse_score_args


def test_parse_builds_valid_application():
    app = parse_score_args("/score 50000 30 2 20000 1500")
    Application(**app)  # проходит валидацию API
    assert app["PAY_1"] == 2 and app["PAY_6"] == 0
    assert app["BILL_AMT3"] == 20000 and app["PAY_AMT6"] == 1500


@pytest.mark.parametrize("text", ["/score", "/score 1 2 3", "/score a b c d e", "/score 50000 10 0 1 1", "/score -5 30 0 1 1"])
def test_parse_errors(text):
    with pytest.raises(ParseError):
        parse_score_args(text)


def test_format_prediction():
    text = format_prediction({"pd": 0.31, "score": 540, "decision": "reject", "threshold": 0.15, "model_version": "v1",
                              "reasons": [{"feature": "PAY_1", "contribution": 0.8}]})
    assert "отказать" in text and "31.0%" in text and "статус платежа" in text
