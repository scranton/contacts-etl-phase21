
# contacts-etl

Contact consolidation + validation + confidence + tagging toolkit.
Designed for use in **JetBrains DataSpell** or any Python environment.

---

## Quickstart (DataSpell)

1. **Open** this folder (`contacts-etl`) in DataSpell.
2. Create a **virtual environment** (Pyenv/Conda/venv).
3. Install the package (editable mode) with optional extras:
   ```bash
   pip install -e '.[validation,dedupe,dev]'
   ```
   *(add `,dedupe` if you plan to use advanced record linkage later)*

4. Copy `config.example.yaml` to `config.yaml` and adjust:
   - Input file paths
   - Prior companies/domains (for work colleagues)
   - Local cities list (for South Shore targeting)

5. Run the full pipeline:

   ```bash
   # Phase 1: Consolidate multiple sources
   contacts-consolidate --config config.yaml

   # Phase 1.0: Validate & score basic data quality
   contacts-validate --config config.yaml

   # Phase 1.5: Compute overall confidence scores
   contacts-confidence --config config.yaml

   # Phase 2: Apply tagging & referral prioritization
   contacts-tag --config config.yaml
   ```

All outputs are written to the folder defined in your `outputs.dir`
(default `/mnt/data`).

---

## Outputs Overview

| Stage | Command | Key Output Files | Description |
|--------|----------|------------------|--------------|
| **Consolidation** | `contacts-consolidate` | `consolidated_contacts.csv`, `consolidated_lineage.csv` | Unified master contact list with lineage tracking |
| **Validation** | `contacts-validate` | `validation_report.csv`, `contact_quality_scored.csv` | Email/phone/address validation & quality scores |
| **Confidence** | `contacts-confidence` | `confidence_report.csv`, `confidence_summary.csv` | Overall per-contact confidence (0–100) |
| **Tagging & Referral** | `contacts-tag` | `tagged_contacts.csv`, `referral_targets.csv` | Tags, categories, and referral priority scores |

---

## Tagging & Referral Rules

Configured in the `tagging:` section of your YAML config.

| Tag | Trigger (examples) | Relationship Category |
|-----|--------------------|------------------------|
| `martial_arts` | “Tai Chi”, “Wu An”, “Wu Dao”, “Shaolin”, “Martial Arts” | personal |
| `nutcracker_performance` | “Nutcracker”, “Cherub”, “Jose Mateo”, “Ballet” | personal |
| `work_colleague` | Company or email domain matches prior employers (GridGain, Red Hat, Tetrate, etc.) | professional |
| `local_south_shore` | Address city matches configured local cities (Braintree, Quincy, Dedham…) | local_referral |

`referral_priority_score = (0.6 × confidence_score) + tag bonuses`, capped at 100.

Outputs include:
- **`tags`** — pipe-delimited list of matched tags  
- **`relationship_category`** — personal / professional / local_referral / uncategorized  
- **`referral_priority_score`** — overall referral potential (0–100)

---

## Example: Warm Outreach Export

```bash
contacts-tag --config config.yaml
# Then filter tagged_contacts.csv for:
# relationship_category in ("personal", "professional")
# and referral_priority_score >= 60
```
This generates `referral_targets.csv` sorted by referral priority, ideal for outreach or CRM import.

---

## Project Structure

```
contacts-etl/
├── pyproject.toml
├── README.md
├── config.example.yaml
└── src/
    └── contacts_etl/
        ├── combine_contacts.py
        ├── validate_quality.py
        ├── confidence_report.py
        ├── tag_contacts.py
        ├── common.py
        └── __init__.py
```

---

## Notes

- All CSVs are fully quoted for safe import into Excel or Pandas.
- Default country for phone normalization is **US**.
- Confidence and referral scores are capped at **100**.
- The pipeline is modular: you can rerun any phase independently.

---
