# Context & Goals

## Business Objectives

- Deliver a CRM-ready consolidated contacts dataset sourced from LinkedIn, macOS Contacts (VCF), and Gmail exports.
- Preserve lineage and provenance (`consolidated_lineage.csv`) so every merged record is auditable.
- Provide scoring/tagging to accelerate outreach triage and warm introductions.

## Desired Outcomes

- **Accuracy**: High precision merges that prefer the most recent source (LinkedIn > macOS VCF > Gmail) while retaining alternate channel data for review.
- **Transparency**: Outputs retain malformed channels in-line (labelled `invalid`), provide lineage counts for every contact, and include a flattened home/work/other view for quick CRM validation.
- **Operational Readiness**: Pipeline commands (`make pipeline` or stage-specific CLI) are fast enough for iterative reruns and produce consistent artifacts for downstream tools.

## Non-Goals (for now)

- Advanced postal address standardization beyond the current lightweight heuristics.
- Automated enrichment from third-party APIs or data brokers.
- Machine-learning-based dedupe models (rule-based similarity only).

## Data Sources & Expectations

| Source | Format | Strengths | Caveats |
|--------|--------|-----------|---------|
| LinkedIn export | CSV | Fresh company/title, LinkedIn URL, reliable name components | Email often missing; phone labels absent |
| macOS Contacts | VCF v3 | Structured name parts, organisation hierarchy/department hints, labelled phones/emails | Professional metadata often stale; vCard escaping must be normalized |
| Gmail export | CSV | Multiple emails/phones with labels, addresses, department, notes | Names are free-form; many unlabeled phones (defaulted to `other`) |

## Privacy & Security

- Treat all contact data as confidential and keep raw exports within the repo boundary.
- CI/logs must redact PII.
- Example datasets used in tests/docs are synthetic.

## Evaluation & Review Flow

- Run pipeline stages in order (`contacts-consolidate` → `contacts-validate` → `contacts-confidence` → `contacts-tag`).
- Review `output/consolidated_contacts.csv` and `...lineage.csv` for merge sanity.
- Use `output/validation_report.csv`, `output/confidence_report.csv`, and `output/referral_targets.csv` to drive CRM import decisions.
- Insight notebooks (`scripts/*_insights.ipynb`) surface top contacts and flagged channel issues.

## Success Signals

- No duplicate `contact_id` values (pipeline aborts if detected).
- Invalid/uncertain channel data is visible (retained with label `invalid`).
- Lineage columns (`source`, `source_row_id`, `source_count`) are populated for every consolidated record.
- Tests (`make test`) run on synthetic data and pass without exposing PII.
