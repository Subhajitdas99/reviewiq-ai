"""ReviewIQ AI: small, explainable review analytics MVP."""

from __future__ import annotations

from io import BytesIO
import re
import math
from typing import Any

import pandas as pd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REQUIRED_COLUMNS = ("review_text",)

# Transparent starter lexicon. Extend against a labeled, representative dataset.
POSITIVE_WORDS = {
    "amazing", "excellent", "fantastic", "fast", "good", "great", "happy",
    "helpful", "love", "perfect", "smooth", "wonderful",
}
NEGATIVE_WORDS = {
    "awful", "bad", "broken", "complaint", "damaged", "delay", "delayed",
    "disappointed", "late", "poor", "refund", "terrible", "unhappy", "worse",
}
TOPIC_RULES = {
    "Customer support": {"support", "service", "help", "helped", "respond", "response", "staff"},
    "Packaging": {"packaging", "package", "packaged", "box", "boxing", "parcel"},
    "Pricing": {"price", "pricing", "cost", "expensive", "cheap", "value", "charge", "charged", "affordable"},
    "Returns and refunds": {"refund", "return", "replace", "replacement", "exchange"},
    "Delivery": {"delivery", "shipping", "arrived", "arrival", "courier", "late", "delay", "delayed"},
    "Product quality": {"broken", "damaged", "defect", "defective", "quality", "works", "item", "product"},
}
URGENT_PHRASES = {"asap", "immediately", "urgent", "emergency", "unsafe", "safety", "fraud", "chargeback", "fire", "electric shock", "injured", "hazard"}
NEUTRAL_PHRASES = {"average", "okay", "ok", "as described", "as expected", "nothing special", "does the job"}
NEGATIVE_PHRASES = {"longer than expected"}


def validate_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded review table; never mutate the input."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Upload must be a CSV table.")
    result = frame.copy()
    result.columns = [str(c).strip().lower() for c in result.columns]
    if not all(col in result.columns for col in REQUIRED_COLUMNS):
        raise ValueError("CSV must include a 'review_text' column.")
    if result.empty:
        raise ValueError("CSV contains no review rows.")
    result["review_text"] = result["review_text"].fillna("").astype(str).str.strip()
    result = result[result["review_text"].ne("")].copy()
    if result.empty:
        raise ValueError("CSV has no non-empty reviews.")
    if "rating" in result.columns:
        raw_rating = result["rating"]
        parsed_rating = pd.to_numeric(raw_rating, errors="coerce")
        invalid_text = raw_rating.notna() & raw_rating.astype(str).str.strip().ne("") & parsed_rating.isna()
        if invalid_text.any():
            raise ValueError("Ratings must be numeric values from 1 to 5, or blank.")
        result["rating"] = parsed_rating
        invalid = result["rating"].notna() & ~result["rating"].between(1, 5)
        if invalid.any():
            raise ValueError("Ratings must be between 1 and 5 (or blank).")
    return result.reset_index(drop=True)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _rating_number(rating: Any) -> float | None:
    """Safely normalize a rating for standalone classifier calls."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 1 <= value <= 5 else None


def classify_sentiment(text: str, rating: Any = None) -> str:
    """Rule-based sentiment: lexicon with neutral phrase handling and rating fallback."""
    lowered = str(text or "").lower()
    words = _tokens(lowered)
    if any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in NEGATIVE_PHRASES):
        return "Negative"
    if any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in NEUTRAL_PHRASES):
        return "Neutral"
    score = sum(word in POSITIVE_WORDS for word in words) - sum(word in NEGATIVE_WORDS for word in words)
    # Rating is a fallback only when the text lexicon has no directional evidence.
    rating_value = _rating_number(rating)
    if score == 0 and rating_value is not None:
        if rating_value >= 4:
            return "Positive"
        if rating_value <= 2:
            return "Negative"
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def classify_topic(text: str) -> str:
    """Choose one topic by priority: Support > Packaging > Pricing > Returns > Delivery > Quality."""
    words = set(_tokens(str(text or "")))
    for topic, keywords in TOPIC_RULES.items():
        if words & keywords:
            return topic
    return "Other"


def classify_urgency(text: str, rating: Any = None) -> str:
    """High requires an explicit urgency/safety/escalation cue; other negative is Medium."""
    lowered = str(text or "").lower()
    if any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in URGENT_PHRASES):
        return "High"
    if set(_tokens(lowered)) & {"late", "delay", "delayed"} or "longer than expected" in lowered:
        return "Medium"
    if classify_sentiment(text, rating) == "Negative":
        return "Medium"
    return "Low"


def analyze_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validated reviews with sentiment, topic, and urgency columns."""
    result = validate_reviews(frame)
    ratings = result["rating"] if "rating" in result else pd.Series([None] * len(result))
    result["sentiment"] = [classify_sentiment(t, r) for t, r in zip(result.review_text, ratings)]
    result["topic"] = result.review_text.map(classify_topic)
    result["urgency"] = [classify_urgency(t, r) for t, r in zip(result.review_text, ratings)]
    return result


def read_uploaded_csv(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV is larger than the 10 MB upload limit.")
    try:
        frame = pd.read_csv(BytesIO(data))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    return validate_reviews(frame)


def suggest_reply(review_text: str, topic: str, sentiment: str, model: str = "gemma3:4b",
                  base_url: str = "http://localhost:11434", timeout: float = 15) -> str:
    """Ask a local Ollama model for a short draft; raise a useful error on failure."""
    import requests

    prompt = ("Write a concise, empathetic customer-service reply (2-3 sentences). "
              "Acknowledge the issue, avoid promising an outcome, and invite the customer "
              "to share order details privately. This is a draft for human review.\n"
              f"Topic: {topic}; sentiment: {sentiment}; review: {review_text[:1500]}")
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty reply.")
        return answer
    except Exception as exc:
        raise RuntimeError(f"Ollama reply unavailable ({model}): {exc}") from exc


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReviewIQ AI", page_icon="📊", layout="wide")
    st.title("ReviewIQ AI")
    st.caption("Rule-based review triage · CSV stays in this session")
    st.info("Sentiment, topic, and urgency are heuristic rule-based predictions, not trained ML outputs. They can miss context and should be reviewed by a person.")
    upload = st.file_uploader("Upload reviews CSV (up to 10 MB)", type=["csv"])
    if not upload:
        st.info("CSV needs a `review_text` column; an optional `rating` column should use 1–5.")
        st.download_button("Download sample CSV", open("data/sample_reviews.csv", "rb"), "sample_reviews.csv", "text/csv")
        return
    try:
        raw = read_uploaded_csv(upload.getvalue())
        analyzed = analyze_reviews(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    a, b, c, d = st.columns(4)
    a.metric("Reviews", len(analyzed))
    b.metric("Negative", int((analyzed.sentiment == "Negative").sum()))
    c.metric("High urgency", int((analyzed.urgency == "High").sum()))
    d.metric("Topics", analyzed.topic.nunique())
    st.subheader("Sentiment")
    st.bar_chart(analyzed.sentiment.value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0))
    st.subheader("Complaint topics")
    st.bar_chart(analyzed.topic.value_counts())
    st.subheader("Analyzed reviews")
    st.dataframe(analyzed, use_container_width=True, hide_index=True)
    st.download_button("Export analyzed CSV", analyzed.to_csv(index=False).encode("utf-8"),
                       "reviewiq_analyzed.csv", "text/csv")
    st.subheader("Optional reply suggestion (Ollama)")
    st.caption("Requires an Ollama server and an available model. Suggestions need human review.")
    row = st.selectbox("Choose a review", range(len(analyzed)), format_func=lambda i: analyzed.iloc[i].review_text[:100])
    model = st.selectbox("Local model", ["gemma3:4b", "llama3.2:3b", "qwen3:4b"])
    if st.button("Generate draft reply"):
        with st.spinner("Contacting local Ollama…"):
            try:
                item = analyzed.iloc[row]
                st.write(suggest_reply(item.review_text, item.topic, item.sentiment, model=model))
            except RuntimeError as exc:
                st.warning(str(exc))


if __name__ == "__main__":
    main()
