# ReviewIQ AI

An explainable, resource-conscious MVP for customer-review triage. Upload a CSV, inspect sentiment/topic/urgency labels and summary charts, then export the analyzed rows. Reply drafting is optional and uses a local Ollama server only when you click the button.

## Windows setup (PowerShell)

```powershell
cd F:\reviewiq-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Alternatively, run `.\.venv\Scripts\python.exe -m streamlit run app.py` without activation.

## CSV format

Required column: `review_text`. Optional `rating` uses 1–5 (blank is allowed). Other columns are retained. Blank review rows are dropped. Maximum upload size is 10 MB. See [data/sample_reviews.csv](data/sample_reviews.csv).

## Analytics behavior

All displayed predictions are heuristic and rule-based; no trained ML model is used. Sentiment uses a small inspectable word lexicon, explicit moderate-language neutral cues, and rating fallback only when text has no directional or neutral cue. Topic matching selects one label by priority: Customer support, Packaging, Pricing, Returns and refunds, Delivery, Product quality, then Other. High urgency requires an explicit urgency, safety, or escalation phrase; other negative feedback, including ordinary delays and low ratings, is Medium. These rules can miss context, sarcasm, negation, and unfamiliar wording, and are not validated classifiers. Read [reports/ai_opportunity_report.md](reports/ai_opportunity_report.md) and [reports/technical_approach.md](reports/technical_approach.md) for limitations and next steps.

## Optional Ollama reply suggestions

Analytics work without Ollama. To enable reply suggestions, run Ollama locally and make sure the selected model is present and can load with available memory. The app offers `gemma3:4b`, `llama3.2:3b`, and `qwen3:4b`; model availability and resource needs vary. A failed connection, missing model, or timeout is shown as a warning. Drafts need human review.

## Tests

```powershell
python -m pytest -q
```

Tests do not download models or require a running Ollama server.
