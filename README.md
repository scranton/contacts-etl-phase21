
# Contact Cleansing Pipeline

Merge contacts exported from macOS Contacts (vCard), Gmail, and LinkedIn into a unified dataset, enrich the records, and prioritize outreach targets.

---

## Requirements

- Python **3.9 – 3.11**
- `pip` (or `uv`, `pipx`, etc.) for dependency management
- Optional extras:
  - `validation` extra enables email/phone validation (`email-validator`, `phonenumberslite`, `dnspython`, `usaddress`)
  - `dedupe` extra enables advanced record linkage (`recordlinkage`, `numpy`)
  - `dev` extra installs testing/tooling (`pytest`, `pytest-cov`, `mypy`, `flake8`, `black`, `isort`)

Data files are expected under the repository by default:

- `data/mac.vcf` — export from macOS Contacts
- `data/gmail.csv` — Gmail contacts export
- `data/linkedin.csv` — LinkedIn connections export

You can override paths in `config.yaml`.

---

## Project Layout

```
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
   pip install -e '.[validation,dedupe,dev]'
   ```
   *(omit extras you do not need)*.

3. Verify the CLI entry points are on your PATH: `contacts-consolidate --help`.

---

## IDE Quickstarts

### JetBrains DataSpell

1. **Open** the project root in DataSpell.
2. Create a new interpreter (`.venv`, Conda, or Pyenv) and install with `pip install -e '.[validation,dedupe,dev]'`.
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

## Pipeline Overview

| Stage | Command | Key Output Files | Purpose |
|-------|---------|------------------|---------|
| Consolidation | `contacts-consolidate --config config.yaml` | `output/consolidated_contacts.csv`, `output/consolidated_lineage.csv` | Merge sources, normalize data, track lineage |
| Validation | `contacts-validate --config config.yaml` | `output/validation_report.csv`, `output/contact_quality_scored.csv` | Score data quality for emails, phones, and addresses |
| Confidence | `contacts-confidence --config config.yaml` | `output/confidence_report.csv`, `output/confidence_summary.csv` | Derive overall contact confidence scores and distribution |
| Tagging & Referral | `contacts-tag --config config.yaml` | `output/tagged_contacts.csv`, `output/referral_targets.csv` | Apply configured tags, relationship categories, and referral priority |

All commands respect the `outputs.dir` setting in `config.yaml`. Adjust this if you want results in a different location.

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

- Run unit tests (requires `dev` extras):
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

- All CSV outputs are fully quoted for compatibility with Excel and Pandas.
- Phone normalization defaults to the United States (`normalization.default_phone_country`).
- Confidence and referral scores are capped at 100.
- Each pipeline phase can be re-run independently; downstream stages read the latest CSVs from `outputs.dir`.

Happy consolidating!
