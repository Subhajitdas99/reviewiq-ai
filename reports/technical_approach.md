# ReviewIQ AI: Technical Approach

## MVP architecture

Streamlit provides a single-session interface. Pandas reads and validates uploaded CSV data in memory, and pure Python functions add sentiment, topic, and urgency labels. The result is shown as summary metrics, charts, and a table, with CSV export. No database or hosted service is required.

## Data contract

- Required: `review_text` (non-empty text).
- Optional: `rating` (numeric 1–5; blank values allowed).
- Other columns are retained and passed through to export.
- Upload size is capped at 10 MB; blank review rows are discarded.

## Classification

All predictions are heuristic and rule-based; no trained ML model is used. Sentiment uses a small positive/negative word lexicon, with explicit moderate/matter-of-fact phrases such as “average”, “okay”, and “as described” treated as Neutral. If there is no text signal, a valid optional rating (4–5 positive, 1–2 negative) is a fallback. Invalid nonblank ratings and numeric ratings outside 1–5 are rejected; missing/blank ratings are accepted. Topics use keyword sets in this priority order: Customer support, Packaging, Pricing, Returns and refunds, Delivery, Product quality, Other. The priority resolves reviews with multiple matching terms to one label. Urgency is High only for explicit urgent, safety, or escalation phrases; routine delays and other negative feedback are Medium, and remaining reviews Low. A rating alone never creates High urgency. The rules and lexicons are in `app.py`; they are a baseline, not a production model.

## Optional reply generation

On click, the app posts a short prompt to the local Ollama `/api/generate` endpoint with `stream=false`, a selected model, and a timeout. Network, HTTP, response, and empty-output failures are surfaced in the UI as a warning. Local inference requires a running Ollama service and sufficient memory; no inference is needed for analytics. Suggestions are explicitly drafts for human review.

## Resource and privacy choices

The analytics path uses Pandas and simple lexical rules; it does not load a transformer or download model weights. The uploaded file is processed in memory and not persisted by the app. Do not include unnecessary personal data. In a hosted deployment, review Streamlit hosting behavior, access controls, retention, and applicable privacy policy.

## Validation

Run `python -m pytest -q` from the project directory. Tests cover schema checks, normalization, missing text and invalid ratings, the observed sample cases, rule outcomes, CSV parsing, and graceful Ollama connection failure. They do not measure classification quality on real-world data or require an Ollama model. Accuracy claims require a representative, labeled evaluation dataset.
