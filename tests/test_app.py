import pandas as pd
import pytest

from app import (
    analyze_reviews,
    classify_sentiment,
    classify_topic,
    classify_urgency,
    read_uploaded_csv,
    suggest_reply,
    validate_reviews,
)


def test_sentiment_lexicon_and_rating_fallback():
    assert classify_sentiment("Excellent and fast") == "Positive"
    assert classify_sentiment("Broken and terrible") == "Negative"
    assert classify_sentiment("fine", 5) == "Positive"
    assert classify_sentiment("fine") == "Neutral"
    assert classify_sentiment("The product is okay and works as described", 3) == "Neutral"
    assert classify_sentiment("Average quality", 5) == "Neutral"


def test_topic_priority_and_other():
    assert classify_topic("Damaged product") == "Product quality"
    assert classify_topic("Need a refund") == "Returns and refunds"
    assert classify_topic("Nothing to report") == "Other"


def test_observed_topic_regressions():
    assert classify_topic("The product arrived damaged and customer support did not respond") == "Customer support"
    assert classify_topic("The product is average for the price") == "Pricing"
    assert classify_topic("Good packaging and the product works perfectly") == "Packaging"


def test_urgency_rules():
    assert classify_urgency("unsafe product") == "High"
    assert classify_urgency("Bad", 1) == "Medium"
    assert classify_urgency("poor quality") == "Medium"
    assert classify_urgency("works perfectly", 5) == "Low"
    assert classify_urgency("Delivery took longer than expected", 2) == "Medium"
    assert classify_urgency("This is urgent, the device is overheating") == "High"


def test_validation_normalizes_and_drops_blank_rows_without_mutating_input():
    source = pd.DataFrame({" Review_Text ": [" Great ", "  "], "Rating": [5, None]})
    result = validate_reviews(source)
    assert list(result.review_text) == ["Great"]
    assert "sentiment" not in source.columns


@pytest.mark.parametrize("frame", [pd.DataFrame({"text": ["bad"]}), pd.DataFrame({"review_text": ["ok"], "rating": [6]})])
def test_validation_rejects_missing_text_or_bad_rating(frame):
    with pytest.raises(ValueError):
        validate_reviews(frame)


@pytest.mark.parametrize("rating", ["not a number", 0, 6])
def test_validation_rejects_invalid_nonblank_ratings(rating):
    with pytest.raises(ValueError, match="Ratings must"):
        validate_reviews(pd.DataFrame({"review_text": ["review"], "rating": [rating]}))


def test_validation_handles_missing_and_empty_text_safely():
    result = validate_reviews(pd.DataFrame({"review_text": [None, "  ", "usable"], "rating": [None, None, ""]}))
    assert result.review_text.tolist() == ["usable"]
    assert pd.isna(result.loc[0, "rating"])
    with pytest.raises(ValueError, match="no non-empty reviews"):
        validate_reviews(pd.DataFrame({"review_text": [None, "   "]}))


def test_analyze_reviews_adds_columns_and_csv_reader():
    analyzed = analyze_reviews(pd.DataFrame({"review_text": ["Broken item"], "rating": [1]}))
    assert analyzed.loc[0, "topic"] == "Product quality"
    assert analyzed.loc[0, "urgency"] == "Medium"
    parsed = read_uploaded_csv(b"review_text,rating\nGreat,5\n")
    assert parsed.loc[0, "review_text"] == "Great"


def test_ollama_failure_is_actionable(monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(RuntimeError, match="Ollama reply unavailable"):
        suggest_reply("Bad", "Other", "Negative", timeout=0.01)
