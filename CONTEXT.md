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
1. **Email** — normalized values are deduped; corporate vs. free-mail is not ranked yet, but duplicates are removed and validity is checked where possible.
2. **Name** — a normalized “best” first name is chosen across sources (nickname-aware), with middle/surname filled from the first non-empty source; suffixes are preserved.
3. **Company / Title** — whichever source offers a value first wins; LinkedIn data naturally tends to dominate because those exports provide explicit company/title fields.

The `consolidated_lineage.csv` export always records which source rows contributed to each merged contact so discrepancies can be audited.

## Review Flow
1. Run the pipeline commands in order:
   - `contacts-consolidate --config config.yaml`
   - `contacts-validate --config config.yaml`
   - `contacts-confidence --config config.yaml`
   - `contacts-tag --config config.yaml`
2. Inspect `output/consolidated_contacts.csv` and `output/consolidated_lineage.csv` for merge verification.
3. Use `output/validation_report.csv`, `output/confidence_report.csv`, and `output/referral_targets.csv` to drive outreach or CRM import.
