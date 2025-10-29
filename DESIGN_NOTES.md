# Design Notes

Implementation details, heuristics, and extension guidelines for the contacts ETL pipeline.

---

## 1. Source Loaders

### 1.1 LinkedIn CSV (`_load_linkedin_csv`)
- Fields ingested: first/last name, company, title, LinkedIn URL, optional email.
- No phone labels provided; emails arrive without `TYPE=` metadata.
- Each row is assigned `source="linkedin"`, `source_row_id=str(index)`.

### 1.2 Gmail CSV (`_load_gmail_csv`)
- Captures up to N emails/phones/addresses by iterating over `E-mail N - Value` columns, splitting multi-value cells (`foo@bar ::: baz@qux`) and deduplicating via `OrderedDict`.
- Labels: normalized to lowercase; duplicates favour a non-empty label if present.
- Phones: `Phone N - Label` is normalized and used when not blank; duplicates unify on value.
- Addresses: deduped on normalized address payload (label ignored in key but preserved when present).
- Emits `source="gmail"`.

### 1.3 macOS Contacts VCF (`_load_vcards`)
- Parses vCard 3.0 records. For each entry:
  - Names (`FN`, `N`) populate structured fields.
  - Emails (`EMAIL`) and phones (`TEL`) parse `TYPE=` parameters while ignoring `pref` and `internet`. Vendor extensions (`X-foo`) drop the `x-` prefix. Preferred label order: `work`, `home`, `other` for emails; `mobile`, `cell`, `iphone`, `work`, `home`, etc. for phones.
  - Addresses (`ADR`) deduped similar to Gmail.
  - Notes (`NOTE`), company (`ORG`), title (`TITLE`), and LinkedIn URL (`URL`) captured when present.
- Emits `source="mac_vcf"`, `source_row_id=str(index)`.

**Extension tips**: New loaders must populate `source`, `source_row_id`, and fill lists (`emails`, `phones`, `addresses`) with already-normalized structures so the downstream normalization step can operate consistently.

---

## 2. Normalization

Implemented in `normalization.py` and used via `normalize_contact_record`.

### 2.1 Names
- Uses nickname map (`_NICKMAP`) and weighted counts to choose best first name (`choose_best_first_name`). Source priority is applied afterward (LinkedIn > mac_vcf > Gmail) for middle/last/suffix.
- Multi-part surnames handled via particle detection.
- Emails embedded in name fields are stripped/collected by `strip_emails_from_text_and_capture`.

### 2.2 Emails
- `normalize_email_collection` dedupes on normalized address, fallback to regex when `email_validator` unavailable.
- Labels normalized to lowercase; optional `TYPE=` tokens from VCF/Gmail are honoured except `pref`/`internet`/`x-*` which are filtered.
- Invalid emails recorded in `invalid_emails` (retained in lineage and consolidated outputs).

### 2.3 Phones
- `normalize_phone_collection` attempts E.164 formatting via `phonenumbers`. Failed formatting records trimmed raw value in `non_standard_phones` while keeping label when possible.
- Deduped on normalized number; label retained if non-empty.

### 2.4 Addresses
- `normalize_address` fills missing city/state/postal when embedded in street line; ISO 2-letter country/state normalization.
- `normalize_address_collection` dedupes dictionary payload (excluding label) and preserves label when one copy supplies it.

---

## 3. Merge Heuristics (`combine_contacts.py`)

### 3.1 Bucketing & Pairing
- Records bucketed by normalized last name; fallback to full name/email/phone or unique `__blank_{idx}`.
- `MergeEvaluator` scores pairs via first-name similarity (nickname aware), corroborators (email/phone/address/linkedin), suffix match.
- Merge allowed if score ≥ threshold or meets relaxed criteria + corroboration. `require_corroborator=True` enforces at least one channel match.

### 3.2 Source Priority
- Metadata selection (company, title, suffixes, LinkedIn URL, middle/last/maiden names) respects priority: LinkedIn (3) > mac_vcf (2) > Gmail (1). Helper `_choose_by_priority` picks the highest-priority non-empty field.
- Emails/phones/addresses from all sources are combined, deduped, and labels preserved as described above.

### 3.3 Output Assembly
- Contact rows include core fields plus joined channel summaries (`emails`, `phones`, etc.). Extra metadata (`invalid_emails`, `non_standard_phones`) stored in `record.extra` and emitted to CSV.
- Lineage rows capture raw values (`source_emails_raw`, `source_phones_raw`) plus normalization drop details for audit.

---

## 4. Scoring & Tagging

### 4.1 Validation (`contacts-validate`)
- Email score: +40 if all valid, +20 if at least one valid.
- Phone score: +30 if all valid, +15 if any valid.
- Address score: +30 if any address has street + (city or postal).
- Output includes detail columns (`emails_detail`, etc.) and summary metrics.

### 4.2 Confidence (`contacts-confidence`)
- Weighted aggregate using validation quality, corroborators, lineage depth, LinkedIn presence, company/title, validated channels, and completeness of name.
- Buckets: `very_high` (≥80), `high` (≥60), `medium` (≥40), `low` (<40).

### 4.3 Tagging (`contacts-tag`)
- Uses `TagEngine` with configurable prior companies/domains/local cities.
- Tags: `martial_arts`, `nutcracker_performance`, `work_colleague`, `local_south_shore` (extendable in config).
- Relationship category prioritized: personal → professional → local → uncategorized.
- Referral priority = `0.6 * confidence_score + tag weights` (capped at 100).
- Notes from Gmail/VCF merged into `notes_blob` when available.

---

## 5. Configuration & Logging

- `config.yaml` defines inputs, output directory, normalization settings, dedupe thresholds, validation options, tagging heuristics, and `logging.level`.
- CLI flags mirror most keys; precedence is `CONTACTS_ETL_LOG_LEVEL` env var > CLI `--log-level` > config value > default `WARNING`.

---

## 6. Testing & CI Notes

- Unit tests (`make test`) cover helper functions (nickname equivalence, VCF row IDs, normalization fallbacks).
- Future work: synthetic end-to-end fixtures, confidence score edge cases, regression diffs on consolidated outputs.
- Keep tests/data synthetic to avoid leaking PII.

---

## 7. Operational Checklist

- Review `output/consolidated_contacts.csv` for duplicates or unexpected drop counts after changes.
- Check insight notebooks for invalid emails/non-standard phones before sharing outputs.
- Regenerate outputs (`make pipeline`) after tweaking normalization or merge heuristics.
