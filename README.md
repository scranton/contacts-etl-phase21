# Contact Cleansing Pipeline

ETL toolkit for unifying contact exports from LinkedIn, macOS Contacts (VCF), and Gmail into a single, scored, and tagged dataset that is ready for CRM import or outreach triage.

---

## Quick Start

1. **Prepare configuration**

   ```bash
   cp config.example.yaml config.yaml
   # edit paths, logging level, tagging heuristics, etc.
   ```

2. **Bootstrap the project (virtualenv + dev deps)**

   ```bash
   make venv install
   ```

3. **Run the full pipeline**

   ```bash
   make pipeline
   ```

   Outputs land in `output/` (changeable via `config.yaml:outputs.dir` or `--out-dir` overrides, which every CLI stage honors).

---

## What You Get

| Stage | Command | Key Files |
|-------|---------|-----------|
| Consolidation | `contacts-consolidate --config config.yaml` | `output/consolidated_contacts.csv`, `output/consolidated_lineage.csv`, `output/flattened_contacts.csv` |
| Validation | `contacts-validate --config config.yaml` | `output/validation_report.csv`, `output/contact_quality_scored.csv` |
| Confidence | `contacts-confidence --config config.yaml` | `output/confidence_report.csv`, `output/confidence_summary.csv` |
| Tagging & Referral | `contacts-tag --config config.yaml` | `output/tagged_contacts.csv`, `output/referral_targets.csv` |

Consolidated contacts include:

- Normalized emails/phones/addresses with lowercased labels (deduped with preference for the freshest source: LinkedIn > macOS VCF > Gmail).
- Inline channel visibility: malformed emails/phones are retained with label `invalid` so you can see the raw values without scanning auxiliary columns.
- Source lineage and counts (`source_count`, `source_row_count`) for auditability, plus a `department` field sourced from Gmail and VCF data when available.
- A flattened projection (`output/flattened_contacts.csv`) that surfaces the first `home`, `work`, and `other` email/phone/address for CRM systems that need fixed columns.

---

## Notebooks & Diagnostics

- `scripts/confidence_insights.ipynb` – top high-confidence contacts with channel details.
- `scripts/referral_insights.ipynb` – highest referral priority rows (falls back to `tagged_contacts.csv` if needed).
- `scripts/invalid_email_insights.ipynb` – quick view of contacts whose emails failed validation.
- `scripts/non_standard_phone_insights.ipynb` – non-standard phone numbers that require review.

Open notebooks from the repo root so they can locate `config.yaml` and the configured output directory.

---

## Configuration & Logging

- `config.yaml` controls input paths, output directory, normalization knobs, dedupe thresholds, validation weights (`quality.*` scores), tagging heuristics, and logging (`logging.level`).
- CLI overrides exist for most options (including `--out-dir`). Logging level precedence is: environment variable `CONTACTS_ETL_LOG_LEVEL` > CLI `--log-level` > `config.yaml` setting > default `WARNING`.

---

## Testing

```bash
make install
make test
```

`flake8 src tests` and `mypy src` are included in the dev extra for quick linting/type checks.

---

## Repo Layout Snapshot

```text
contacts-etl-phase21/
├── config.yaml           # project configuration (copy from config.example.yaml)
├── data/                 # raw exports (vcf/csv)
├── output/               # generated datasets
├── scripts/              # insight notebooks
├── src/contacts_etl/     # pipeline code
└── tests/                # unit tests
```

For goals, desired outcomes, and heuristic details see `CONTEXT.md` and `DESIGN_NOTES.md` respectively.
