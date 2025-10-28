# Context

## Business Objectives
- Produce a single “golden” contacts table suitable for CRM import.
- Keep a transparent audit trail: what merged with what, why, and confidence.
- Support interactive review in OpenRefine for tricky cases.

## Non-Goals (for now)
- No fuzzy street address standardization.
- No contact enrichment from external APIs.

## Privacy & Security
- Treat all contact data as confidential.
- Never upload raw data to third-party services outside this repo/CI.
- CI must redact PII from logs/artifacts.

## Source Truth Hierarchy
When conflicting fields exist:
1. **Email**: any verified corporate domain beats free-mail; otherwise most recent source timestamp wins.
2. **Name**: prefer full legal name if present; otherwise longest tokenized full name.
3. **Company/Title**: prefer sources with explicit fields (LinkedIn > vCard note > Gmail notes).

## Review Flow
1. Run ETL → generate normalized file.
2. Dedupe: deterministic pass, then fuzzy pass → write `candidate_pairs.csv`.
3. Export final CSV for CRM.
