# Design Notes

These notes capture the intent behind the contact consolidation pipeline so future contributors understand why the current heuristics exist and how to extend them safely.

---

## 1. Data Sources & Expectations

| Source | Format | Key Fields | Reliability |
|--------|--------|------------|-------------|
| macOS Contacts | VCF (v3) | Structured name components, free-form notes, multiple phones/emails | High for personal details, sparse for company/title |
| Gmail Export | CSV | Multiple labeled emails/phones, addresses, notes | Medium; labels matter, but names are often free-form |
| LinkedIn Export | CSV | First/last name, company, position, profile URL, optional email | High for professional attributes |

Each loader normalizes into a shared contact dictionary (`emails`, `phones`, `addresses`, `company`, `title`, `source`, `source_row_id`). The merge pipeline assumes **string** fields with leading/trailing whitespace already trimmed and multi-value fields represented as lists of dictionaries.

---

## 2. Consolidation Heuristics

### 2.1 Name Normalization

- Best-first-name selection uses nickname equivalence (`William` ≈ `Bill`) and frequency weighting so the most corroborated variant wins.
- Multi-part surnames (e.g., `de la Cruz`) are preserved by scanning for particles.
- Embedded emails are stripped from name fields to avoid polluting dedupe keys.

**When extending**: keep `strip_suffixes_and_parse_name` as the single entry point so suffix logic and maiden-name extraction stay consistent across sources.

### 2.2 Email Handling

- Every email passes through `validate_email_safe`. If the optional `email_validator` dependency is available, deliverability is checked; otherwise a regex fallback is used.
- Duplicate emails are removed while preserving the first-seen label (e.g., `work`, `home`).
- Emails extracted from free-text fields (names, notes) are appended with an empty label so downstream phases can still locate them.

### 2.3 Phone & Address Normalization

- `normalize_phone` wraps E.164 formatting via `phonenumbers` when installed; without it the fallback enforces a `+` country code and basic digit rules.
- Addresses retain all components but do a light parse to fill in missing city/state/postal when the street field includes them inline.
- Full postal standardization is intentionally out of scope; complex fixes should happen downstream in tools like OpenRefine.

### 2.4 Merge Decisioning

- Records are bucketed by normalized last name to reduce comparisons.
- Pairwise scoring includes:
  - First-name similarity (nickname aware)
  - Exact overlap of emails, phones, addresses (two or more components), LinkedIn URL
  - Generational suffix match
- A successful merge requires either:
  - Total score ≥ `merge_score_threshold`, or
  - First-name similarity ≥ threshold *and* total score ≥ `relaxed_merge_threshold`
- If either contact lacks a clear first/last name, at least one corroborator (email/phone/address/linkedin) must match.
- When both contacts provide first names, we only merge if the normalized names match, the pair is nickname-equivalent, or there is explicit email/LinkedIn corroboration. This prevents housemates sharing an address or phone from collapsing.
- Global `require_corroborator` can tighten the rule set via `config.yaml`.

**Guiding principle**: default to high precision so manual review handles the ambiguous cases.

---

## 3. Lineage Tracking

For every merged contact:

- `contact_id` is a deterministic UUID built from normalized name, deduped emails/phones, and the contributing `(source, source_row_id)` pairs. Including lineage keys keeps IDs stable while ensuring distinct people in the same household stay separate.
- `consolidated_lineage.csv` contains one row per source record with full snapshots of original fields (even if later normalized). This is the canonical place to investigate a merge and verify provenance.
- VCF rows now receive sequential `source_row_id` values to align with notes extraction.
- `consolidated_contacts.csv` reports both `source_count` (unique source systems) and `source_row_count` (total contributing rows). The builder fails fast if any duplicate `contact_id` slips through.

When adding new sources, ensure:

1. Source loader sets `source` and `source_row_id`.
2. Loader emits raw notes/emails/phones so normalization can dedupe consistently.

---

## 4. Scoring & Tagging Intent

### 4.1 Validation Quality Score (`contacts-validate`)

- Email: +40 if all emails are syntactically valid, +20 if at least one valid.
- Phone: +30 if all phones are valid, +15 if at least one valid.
- Address: +30 when any address includes street plus city or postal code.
- Output includes JSON details so analysts can drill down per contact.

**Extensions**: consider weighting corporate domains higher once DNS checks are reliable in the deployment environment.

### 4.2 Confidence Score (`contacts-confidence`)

- Base 0–40 from validation quality (scaled).
- Corroborators (email/phone/address/linkedin) up to +20.
- Lineage depth bonus (up to +10) rewards contacts seen in multiple sources.
- LinkedIn URL and company/title add professional context (up to +15 combined).
- Validated channels add up to +10, presence of full name adds +5.

Scores map into buckets (`very_high`, `high`, `medium`, `low`) for dashboards.

### 4.3 Tagging & Referral (`contacts-tag`)

- Tags:
  - `martial_arts`, `nutcracker_performance` capture personal affinity signals.
  - `work_colleague` flags prior employer/domain matches.
  - `local_south_shore` uses MA cities configured in YAML.
- Relationship categories follow a priority order: personal → professional → local → uncategorized.
- Referral priority = `0.6 * confidence_score + tag bonuses` (capped at 100).
- Notes from Gmail/VCF feed a `notes_blob` column to aid manual review.

**Future ideas**: add negative tags (e.g., “do not contact”), integrate CRM status, or allow tag weight overrides per deployment.

---

## 5. Configuration Philosophy

`config.yaml` is the single source of truth for:

- Input paths (`inputs`)
- Output directory (`outputs.dir`)
- Normalization toggles (default country, suffix lists)
- Dedupe thresholds & nickname equivalence flag
- Validation options (MX lookup, strict phone validation)
- Tagging parameters (companies, domains, local cities)

CLI arguments mirror most keys with the convention **CLI > config > default**. When adding new config options, extend both the CLI parser and `build` function so library and CLI usage stay in sync.

---

## 6. Testing Roadmap (Future)

- Add fixture-driven tests for multi-source merges with conflicting data.
- Cover confidence score edge cases (e.g., zero corroborators but high quality).
- Integration smoke test that runs the full pipeline on a synthetic dataset and diff-checks outputs.

Until then, unit tests highlight the most fragile heuristics:

- Nickname equivalence toggle
- VCF row-id assignment and notes propagation
- Phone normalization fallbacks

---

## 7. Operational Notes

- CSV outputs are fully quoted (`csv.QUOTE_ALL`) to keep Excel import safe.
- Phone normalization assumes US country defaults; consider parameterizing per deployment.
- The repository avoids shipping PII in tests. Example data should remain synthetic.
- CI (if enabled) must redact any email/phone details from logs.

Maintaining these practices ensures the pipeline’s behavior stays predictable, auditable, and privacy-conscious.
