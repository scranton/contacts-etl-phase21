
# Contact Cleansing Pipeline

Merge contacts exported from macOS Contacts (vCard), Gmail, and LinkedIn into a unified dataset, enrich the records, and prioritize outreach targets.

---

## Requirements

- Python **3.12+** (project targets 3.12.4)
- `pip` (or `uv`, `pipx`, etc.) for dependency management
- All pipeline dependencies install with the base package; extras only add tooling:
  - `tests` adds just `pytest`
  - `dev` installs the full testing/tooling stack (`pytest`, `pytest-cov`, `mypy`, `flake8`, `black`, `isort`)

Data files are expected under the repository by default:

- `data/mac.vcf` — export from macOS Contacts
- `data/gmail.csv` — Gmail contacts export
- `data/linkedin.csv` — LinkedIn connections export

You can override paths in `config.yaml`.

---

## Project Layout

```shell
contacts-etl-phase21/
├── config.yaml
├── data/
│   ├── gmail.csv
│   ├── linkedin.csv
│   └── mac.vcf
├── output/               # created by the pipeline
├── src/
│   └── contacts_etl/
│       ├── combine_contacts.py
│       ├── validate_quality.py
│       ├── confidence_report.py
│       ├── tag_contacts.py
│       ├── common.py
│       └── __init__.py
├── tests/
│   └── test_combine_helpers.py
├── scripts/
│   └── inspect.ipynb
├── pyproject.toml
└── README.md
```

---

## Environment Setup

1. Copy the sample configuration and adjust paths/weights:

   ```bash
   cp config.example.yaml config.yaml
   # edit config.yaml to point at your data and tweak tagging settings
   ```

2. Create a virtual environment and install the package in editable mode:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[dev]'
   ```

   *(use `pip install -e .` for the lean runtime; add `[tests]` if you only need pytest)*.

3. Verify the CLI entry points are on your PATH: `contacts-consolidate --help`.

---

## IDE Quickstarts

### JetBrains DataSpell

1. **Open** the project root in DataSpell.
2. Create a new interpreter (`.venv`, Conda, or Pyenv) and install with `pip install -e '.[dev]'`.
3. Use the built-in **Run/Debug Configurations**:
   - Create a new *Python* configuration per command (`contacts-consolidate`, `contacts-validate`, etc.).
   - Set the working directory to the project root and pass `--config config.yaml`.
4. Use DataSpell notebooks (`scripts/inspect.ipynb`) for ad-hoc inspection if needed.

### Visual Studio Code

1. Open the folder in VS Code and select the `.venv` interpreter from the command palette (`Python: Select Interpreter`).
2. Install the recommended extensions (`ms-python.python`).
3. Create `.vscode/launch.json` entries or use the integrated terminal:

   ```bash
   source .venv/bin/activate
   contacts-consolidate --config config.yaml
   ```

4. Configure *Run Task* definitions if you prefer one-click execution of pipeline stages.

---

## Analysis Notebooks

- `scripts/confidence_insights.ipynb` — highlights the top 25 contacts from `confidence_report.csv` in the `very_high` and `high` buckets, including source counts and available contact channels.
- `scripts/referral_insights.ipynb` — surfaces the highest `referral_priority_score` entries from `referral_targets.csv` (or `tagged_contacts.csv` as a fallback) so you can line up warm introductions quickly.

Open either notebook in VS Code Jupyter (Python 3.12+ kernel) after regenerating the pipeline outputs to explore the latest results.

---

## Pipeline Overview

| Stage | Command | Key Output Files | Purpose |
|-------|---------|------------------|---------|
| Consolidation | `contacts-consolidate --config config.yaml` | `output/consolidated_contacts.csv`, `output/consolidated_lineage.csv` | Merge sources, normalize data, track lineage (adds `source_row_count` and enforces unique `contact_id`) |
| Validation | `contacts-validate --config config.yaml` | `output/validation_report.csv`, `output/contact_quality_scored.csv` | Score data quality for emails, phones, and addresses |
| Confidence | `contacts-confidence --config config.yaml` | `output/confidence_report.csv`, `output/confidence_summary.csv` | Derive overall contact confidence scores and distribution |
| Tagging & Referral | `contacts-tag --config config.yaml` | `output/tagged_contacts.csv`, `output/referral_targets.csv` | Apply configured tags, relationship categories, and referral priority |

All commands respect the `outputs.dir` setting in `config.yaml`. Adjust this if you want results in a different location.

---

## Pain Points & Done Criteria

**Consolidation (`contacts-consolidate`)**

- Pain points: ambiguous duplicates without corroborators, inconsistent naming (nicknames vs. legal names), stray emails embedded in name fields.
- Done when: every merged record has traceable lineage rows, first/last names are populated when possible, and high-risk duplicates are isolated in the lineage output for manual review.

**Validation (`contacts-validate`)**

- Pain points: missing libraries for deep validation, false positives from badly formatted exports, incomplete phone metadata.
- Done when: quality scores populate for each contact, invalid channel details surface in the JSON detail columns, and success metrics print in the CLI summary.

**Confidence (`contacts-confidence`)**

- Pain points: contacts without corroborators skewing averages, stale lineage counts, weighting drift when upstream heuristics change.
- Done when: every contact gains a bounded 0–100 confidence score, bucket summaries appear in the console, and the `confidence_report.csv` aligns with validation metrics.

**Tagging & Referral (`contacts-tag`)**

- Pain points: config drift (missing prior companies/domains), absent confidence scores, unexpected characters in notes/addresses.
- Done when: `tagged_contacts.csv` includes tags, relationship categories, referral priority scores, and `referral_targets.csv` is prioritized for outreach.

Keep this section current whenever heuristics change so contributors and reviewers can judge success quickly.

---

## Tagging Rules (configurable)

The `tagging` section of `config.yaml` controls the heuristics. Default rules map to:

| Tag | Trigger Examples | Relationship Category |
|-----|------------------|-----------------------|
| `martial_arts` | Mentions of Tai Chi, Wu Dao, Kung Fu, Shaolin | personal |
| `nutcracker_performance` | Mentions of Nutcracker, Cherub, Jose Mateo | personal |
| `work_colleague` | Prior employer matches or email domains | professional |
| `local_south_shore` | Massachusetts city in configured local list | local_referral |

Referral priority is computed as `0.6 * confidence_score + tag bonuses`, capped at 100.

---

## Warm Outreach Example

```bash
contacts-tag --config config.yaml
# Filter output/tagged_contacts.csv for:
#   relationship_category in ("personal", "professional")
#   referral_priority_score >= 60
```

This yields `output/referral_targets.csv` sorted by referral priority.

---

## Testing & Tooling

- Install test tooling only:

  ```bash
  pip install -e '.[tests]'
  ```

- Run unit tests (requires `tests` or `dev` extras):

  ```bash
  python3 -m pytest
  ```

- Static analysis and linting:

  ```bash
  mypy src
  flake8 src tests
  black --check src tests
  isort --check-only src tests
  ```

---

## Additional Notes

- The consolidation stage now raises an error if duplicate `contact_id` values are produced; this catches determinism or merge regressions early.
- `source_count` reflects the number of distinct source systems contributing to a contact, while `source_row_count` captures the total number of source rows merged.
- All CSV outputs are fully quoted for compatibility with Excel and Pandas.
- Phone normalization defaults to the United States (`normalization.default_phone_country`).
- Confidence and referral scores are capped at 100.
- Each pipeline phase can be re-run independently; downstream stages read the latest CSVs from `outputs.dir`.
- Tests and documentation use synthetic names, emails, and phone numbers—never commit real PII.

Happy consolidating!
