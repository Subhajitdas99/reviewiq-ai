# ReviewIQ AI: AI Opportunity Report

## Problem

Teams receive reviews across channels and struggle to find recurring complaints and respond quickly. Manual review triage is slow, inconsistent, and difficult to scale.

## MVP opportunity

ReviewIQ turns a CSV of customer reviews into a compact view of sentiment, complaint themes, and reviews that may need prompt attention. All current sentiment, topic, and urgency predictions are heuristic and rule-based; no trained ML model is used. A small word lexicon classifies sentiment, ordered keyword rules assign one topic, and explicit urgent, safety, or escalation phrases flag High urgency. Ordinary delays and low ratings alone are not High urgency.

## Where AI can help

- Aggregate sentiment and recurring themes across large review volumes.
- Surface negative or potentially time-sensitive feedback for human review.
- Draft empathetic reply suggestions using a local Ollama model when available.
- Later: cluster emerging themes, detect product-specific issues, and learn from support-team corrections.

## Limits and risks

These rules are not validated and can miss context, sarcasm, negation, multilingual content, and domain-specific vocabulary. One review can mention multiple topics but receives one label according to fixed priority. Ratings can disagree with text. Urgency is a triage hint, not a safety or escalation decision. Reply text may be wrong or inappropriate and must be reviewed by a person. Uploaded content may contain personal data, so users should minimize sensitive fields and follow their organization’s data policy.

## Evaluation plan

Build a representative, consented, human-labeled dataset. Measure sentiment macro-F1, topic macro-F1, high-urgency recall, and reviewer agreement by language/product. Inspect errors and subgroup performance before replacing rules. Track reply acceptance/edit rates and complaint resolution outcomes; do not optimize solely for reply fluency.

## Next steps

1. Confirm the CSV schema and intended support workflow with users.
2. Gather and label a privacy-reviewed evaluation sample.
3. Compare the baseline rules with a compact supervised classifier.
4. Add model versioning, auditability, and human feedback before production use.
