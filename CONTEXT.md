# Context

## Business Objectives

- Produce a consolidated contacts dataset that can be fed directly into CRM or outreach tooling.
- Preserve lineage so every merged record can be traced back to its source rows.
- Score and tag contacts so that warm outreach, referrals, and quality reviews can be prioritized quickly.

## Non-Goals (for now)

- No advanced postal address standardization beyond light normalization.
- No automated enrichment from third-party APIs or data brokers.
- No ML-driven fuzzy dedupe beyond the rule-based heuristics already in the pipeline.

## Privacy & Security

- Treat all contact data as confidential.
- Never upload raw data to third-party services outside this repo/CI.
- CI must redact PII from logs/artifacts.

## Source Truth Handling

Field precedence is handled heuristically in code:

1. **Email** — normalized values are deduped with lowercase labels when provided (VCF/Gmail `TYPE=` tokens such as `pref`/`internet` are ignored). Validity checks remove syntactically invalid addresses while retaining the raw value for review.
2. **Name** — a normalized “best” first name is chosen across sources (nickname-aware), with middle/surname pulled from the highest-priority source that supplies each component; suffixes are preserved.
3. **Company / Title** — sources are prioritized in recency order: LinkedIn > macOS VCF > Gmail. The merge selects the first non-empty value respecting that priority so newer data wins by default.

The `consolidated_lineage.csv` export always records which source rows contributed to each merged contact so discrepancies can be audited.

- `contact_id` generation now incorporates the contributing `(source, source_row_id)` pairs, and the consolidation step aborts if duplicates appear. Investigate any failure here before trusting downstream CSVs.
- `source_count` captures the number of distinct systems (e.g., Gmail vs. LinkedIn) behind a contact, while `source_row_count` reports how many raw records were merged.
- `invalid_emails` and `non_standard_phones` columns in `consolidated_contacts.csv` surface any channel data that normalization had to drop for follow-up remediation.

## Review Flow

1. Run the pipeline commands in order:
   - `contacts-consolidate --config config.yaml`
   - `contacts-validate --config config.yaml`
   - `contacts-confidence --config config.yaml`
   - `contacts-tag --config config.yaml`
2. Inspect `output/consolidated_contacts.csv` and `output/consolidated_lineage.csv` for merge verification.
3. Use `output/validation_report.csv`, `output/confidence_report.csv`, and `output/referral_targets.csv` to drive outreach or CRM import.

## Testing & Privacy

- Install test tooling with `pip install -e '.[tests]'` (or `.[dev]`) and run `python3 -m pytest`. The suite uses only synthetic data, so it is safe to execute in shared environments.
