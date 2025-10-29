from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict, OrderedDict
from dataclasses import replace
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from .common import (
    ContactRecord,
    Email,
    LineageEntry,
    MergeEvaluator,
    NormalizationSettings,
    choose_best_first_name,
    deterministic_uuid,
    format_phone_e164_safe,
    is_valid_phone_safe,
    load_config,
    nickname_equivalent,
    normalize_contact_record,
    normalize_text_key,
    read_csv_with_optional_header,
    safe_get,
    warn_missing,
)
from .config_loader import PipelineConfig
from .logging_utils import configure_logging
from .models import Address, Phone

# use module logger instead of configuring logging at import time
logger = logging.getLogger(__name__)

DEFAULT_GEN = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
DEFAULT_PROF = {
    "phd",
    "pmp",
    "csm",
    "spc6",
    "mba",
    "cissp",
    "crisc",
    "cscp",
    "cams",
    "cpa",
    "cfa",
    "pe",
    "cisa",
    "cism",
    "cfe",
    "cma",
    "ceh",
    "itil",
    "sixsigma",
    "leansixsigma",
    "esq",
    "jd",
}


PHONE_VALUE_PATTERN = re.compile(r"\+?\d[\d\s()./-]{6,}\d")

SOURCE_PRIORITY = {
    "linkedin": 3,
    "mac_vcf": 2,
    "gmail": 1,
}


def _source_priority(record: ContactRecord) -> int:
    return SOURCE_PRIORITY.get((record.source or "").lower(), 0)


def _choose_by_priority(
    records: List[ContactRecord], getter: Callable[[ContactRecord], str]
) -> str:
    best_value = ""
    best_priority = -1
    for record in records:
        value = getter(record)
        if not value:
            continue
        priority = _source_priority(record)
        if priority > best_priority:
            best_priority = priority
            best_value = value
    return best_value


def _build_normalization_settings(config: PipelineConfig) -> NormalizationSettings:
    keep_suffixes = config.normalization.keep_generational_suffixes or list(DEFAULT_GEN)
    prof_suffixes = config.normalization.professional_suffixes or list(DEFAULT_PROF)
    return NormalizationSettings.from_args(
        keep_suffixes, prof_suffixes, config.normalization.default_phone_country
    )


def _load_linkedin_csv(path: Optional[str]) -> List[ContactRecord]:
    if not path:
        return []
    if warn_missing(path, "LinkedIn"):
        return []
    df = read_csv_with_optional_header(path, header_starts_with="First Name,Last Name,URL")
    records: List[ContactRecord] = []
    for idx, row in df.iterrows():
        linkedin_url = safe_get(row, "URL")
        if "linkedin.com" not in linkedin_url.lower():
            linkedin_url = ""
        primary_email = safe_get(row, "Email Address")
        emails = [Email(value=primary_email, label="work")] if primary_email else []
        full_name_raw = " ".join([safe_get(row, "First Name"), safe_get(row, "Last Name")]).strip()
        record = ContactRecord(
            full_name_raw=full_name_raw,
            company=safe_get(row, "Company"),
            title=safe_get(row, "Position"),
            linkedin_url=linkedin_url,
            source="linkedin",
            source_row_id=str(idx),
            emails=emails,
        )
        records.append(record)
    return records


def _extract_phone_values(raw: str) -> List[str]:
    if not raw:
        return []
    candidates: List[str] = []
    for part in re.split(r"[\r\n|;]+", raw):
        part = part.strip()
        if not part:
            continue
        matches = PHONE_VALUE_PATTERN.findall(part)
        if matches:
            candidates.extend(match.strip() for match in matches)
        else:
            candidates.append(part)
    return [candidate for candidate in candidates if candidate]


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _normalize_phone_label(label: str) -> str:
    return _normalize_label(label)


def _record_phone(
    phone_map: "OrderedDict[str, str]", raw_value: str, label: str
) -> None:
    value = (raw_value or "").strip()
    if not value:
        return
    label_norm = _normalize_phone_label(label)
    current = phone_map.get(value)
    if current is None or (not current and label_norm):
        phone_map[value] = label_norm


def _record_email(
    email_map: "OrderedDict[str, str]", raw_value: str, label: str
) -> None:
    value = (raw_value or "").strip()
    if not value:
        return
    label_norm = _normalize_label(label)
    current = email_map.get(value)
    if current is None or (not current and label_norm):
        email_map[value] = label_norm


def _extract_type_tokens(params: List[str]) -> List[str]:
    raw_tokens: List[str] = []
    for param in params:
        if not param:
            continue
        if "=" in param:
            key, val = param.split("=", 1)
            if key.strip().lower() == "type":
                for token in re.split(r"[;,]", val):
                    token = token.strip()
                    if token:
                        raw_tokens.append(token.lower())
        else:
            raw_tokens.extend(
                token.strip().lower() for token in param.split(",") if token.strip()
            )

    normalized: List[str] = []
    for token in raw_tokens:
        if token in {"pref", "internet"}:
            continue
        if token.startswith("x-"):
            token = token[2:]
        if token:
            normalized.append(token)
    return normalized


def _extract_email_values(raw: str) -> List[str]:
    if not raw:
        return []
    candidates: List[str] = []
    for part in re.split(r"[\r\n|;]+", raw):
        part = part.strip()
        if not part:
            continue
        # Emails sometimes embedded in text like "foo@bar ::: baz@qux"
        subparts = [segment.strip() for segment in re.split(r":::+", part) if segment.strip()]
        if len(subparts) > 1:
            candidates.extend(subparts)
        else:
            candidates.append(part)
    return [candidate for candidate in candidates if candidate]


def _load_gmail_csv(path: Optional[str]) -> List[ContactRecord]:
    if not path:
        return []
    if warn_missing(path, "GMail"):
        return []
    df = read_csv_with_optional_header(path)
    records: List[ContactRecord] = []
    for idx, row in df.iterrows():
        email_map: "OrderedDict[str, str]" = OrderedDict()
        for column in row.index:
            if not str(column).startswith("E-mail ") or not str(column).endswith(" - Value"):
                continue
            email_value_raw = safe_get(row, column)
            if not email_value_raw:
                continue
            label_col = str(column).replace(" - Value", " - Type")
            email_label = _normalize_label(safe_get(row, label_col))
            for extracted in _extract_email_values(email_value_raw):
                _record_email(email_map, extracted, email_label)
        emails = [Email(value=value, label=label) for value, label in email_map.items()]

        phones_map: "OrderedDict[str, str]" = OrderedDict()
        for column in row.index:
            if not str(column).startswith("Phone ") or not str(column).endswith(" - Value"):
                continue
            phone_value_raw = safe_get(row, column)
            if not phone_value_raw:
                continue
            label_col = str(column).replace(" - Value", " - Label")
            phone_label = _normalize_phone_label(safe_get(row, label_col))
            for extracted in _extract_phone_values(phone_value_raw):
                _record_phone(phones_map, extracted, phone_label)

        phones = [Phone(value=value, label=label) for value, label in phones_map.items()]

        address_map: "OrderedDict[str, Address]" = OrderedDict()
        address_ids: Set[str] = set()
        for column in row.index:
            match = re.match(r"Address (\d+) - ", str(column))
            if match:
                address_ids.add(match.group(1))
        for addr_id in sorted(address_ids, key=lambda value: int(value)):
            address_entry = Address(
                po_box=safe_get(row, f"Address {addr_id} - PO Box"),
                extended=safe_get(row, f"Address {addr_id} - Extended Address"),
                street=safe_get(row, f"Address {addr_id} - Street")
                or safe_get(row, f"Address {addr_id} - Formatted"),
                city=safe_get(row, f"Address {addr_id} - City"),
                state=safe_get(row, f"Address {addr_id} - Region"),
                postal_code=safe_get(row, f"Address {addr_id} - Postal Code"),
                country=safe_get(row, f"Address {addr_id} - Country"),
                label=_normalize_label(safe_get(row, f"Address {addr_id} - Label")),
            )
            key_payload = address_entry.to_dict()
            key_payload.pop("label", None)
            key = json.dumps(key_payload, sort_keys=True)
            if any(
                getattr(address_entry, field)
                for field in ("street", "city", "state", "postal_code", "country", "po_box")
            ):
                existing = address_map.get(key)
                if existing is None or (not existing.label and address_entry.label):
                    address_map[key] = address_entry
        addresses = list(address_map.values())
        raw_full = " ".join(
            [safe_get(row, "First Name"), safe_get(row, "Middle Name"), safe_get(row, "Last Name")]
        ).strip()
        record = ContactRecord(
            full_name_raw=raw_full,
            suffix=safe_get(row, "Name Suffix"),
            company=safe_get(row, "Organization Name"),
            title=safe_get(row, "Organization Title"),
            source="gmail",
            source_row_id=str(idx),
            emails=emails,
            phones=phones,
            addresses=addresses,
            notes=safe_get(row, "Notes"),
        )
        records.append(record)
    return records


def _load_vcards(path: Optional[str]) -> List[ContactRecord]:
    if not path:
        return []
    if warn_missing(path, "Mac VCF"):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    blocks = [block for block in content.split("END:VCARD") if "BEGIN:VCARD" in block]
    records: List[ContactRecord] = []
    for idx, block in enumerate(blocks):
        b = f"{block}END:VCARD"
        record = ContactRecord(source="mac_vcf", source_row_id=str(idx))
        phone_map: "OrderedDict[str, str]" = OrderedDict()
        address_map: "OrderedDict[str, Address]" = OrderedDict()
        email_map: "OrderedDict[str, str]" = OrderedDict()
        for line in b.splitlines():
            if line.startswith("FN:"):
                record.full_name_raw = line[3:].strip()
            elif line.startswith("N:"):
                components = line[2:].split(";")
                record.last_name = components[0].strip() if len(components) > 0 else ""
                record.first_name = components[1].strip() if len(components) > 1 else ""
                record.middle_name = components[2].strip() if len(components) > 2 else ""
                record.suffix = components[3].strip() if len(components) > 3 else ""
                if not record.full_name_raw:
                    record.full_name_raw = " ".join(
                        filter(
                            None,
                            [
                                record.first_name,
                                record.middle_name,
                                record.last_name,
                                record.suffix,
                            ],
                        )
                    ).strip()
            elif line.upper().startswith("EMAIL") and ":" in line:
                params_part, value = line.split(":", 1)
                params = params_part.split(";")[1:]
                email_tokens = _extract_type_tokens(params)
                preferred_email_labels = [
                    "work",
                    "home",
                    "other",
                ]
                label = ""
                for preferred in preferred_email_labels:
                    if preferred in email_tokens:
                        label = preferred
                        break
                if not label and email_tokens:
                    label = email_tokens[0]
                _record_email(email_map, value, label)
            elif line.upper().startswith("TEL") and ":" in line:
                params_part, value = line.split(":", 1)
                params = params_part.split(";")[1:]
                tokens = _extract_type_tokens(params)
                label = ""
                preferred_order = [
                    "mobile",
                    "cell",
                    "iphone",
                    "work",
                    "home",
                    "main",
                    "fax",
                    "pager",
                    "other",
                    "voice",
                ]
                for preferred in preferred_order:
                    if preferred in tokens:
                        label = preferred
                        break
                if not label and tokens:
                    label = tokens[0]
                _record_phone(phone_map, value, label)
            elif line.startswith("ADR"):
                parts = line.split(":", 1)[1].split(";")
                address = Address(
                    po_box=parts[0].strip() if len(parts) > 0 else "",
                    extended=parts[1].strip() if len(parts) > 1 else "",
                    street=parts[2].strip() if len(parts) > 2 else "",
                    city=parts[3].strip() if len(parts) > 3 else "",
                    state=parts[4].strip() if len(parts) > 4 else "",
                    postal_code=parts[5].strip() if len(parts) > 5 else "",
                    country=parts[6].strip() if len(parts) > 6 else "",
                )
                key_payload = address.to_dict()
                key_payload.pop("label", None)
                key = json.dumps(key_payload, sort_keys=True)
                existing = address_map.get(key)
                if existing is None or (not existing.label and address.label):
                    address_map[key] = address
            elif line.startswith("ORG:"):
                record.company = line[4:].strip()
            elif line.startswith("TITLE:"):
                record.title = line[6:].strip()
            elif line.startswith("URL:") and "linkedin.com" in line.lower():
                record.linkedin_url = line[4:].strip()
            elif line.startswith("NOTE:"):
                record.notes = line[5:].strip()
        record.emails = [Email(value=value, label=label) for value, label in email_map.items()]
        record.phones = [Phone(value=value, label=label) for value, label in phone_map.items()]
        record.addresses = list(address_map.values())
        records.append(record)
    return records


def _load_sources(config: PipelineConfig) -> List[ContactRecord]:
    records: List[ContactRecord] = []
    records.extend(_load_linkedin_csv(config.inputs.get("linkedin_csv")))
    records.extend(_load_gmail_csv(config.inputs.get("gmail_csv")))
    records.extend(_load_vcards(config.inputs.get("mac_vcf")))
    return records


def _normalize_records(
    records: List[ContactRecord], settings: NormalizationSettings
) -> List[ContactRecord]:
    normalized: List[ContactRecord] = []
    for record in records:
        normalized.append(normalize_contact_record(replace(record), settings))
    return normalized


def _bucket_records(records: List[ContactRecord]) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = normalize_text_key(record.last_name)
        if not key:
            key = normalize_text_key(record.full_name)
        if not key and record.emails:
            key = normalize_text_key(record.emails[0].value)
        if not key and record.phones:
            key = normalize_text_key(record.phones[0].value)
        if not key:
            key = f"__blank_{idx}"
        buckets[key].append(idx)
    return buckets


def _cluster_indices(
    records: List[ContactRecord], evaluator: MergeEvaluator, config: PipelineConfig
) -> Dict[int, List[int]]:
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    buckets = _bucket_records(records)
    for key, indices in buckets.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                left = records[indices[i]]
                right = records[indices[j]]
                signals = evaluator.compute(left, right)

                score = signals.score
                high = config.dedupe.merge_score_threshold
                relaxed = config.dedupe.relaxed_merge_threshold
                first_threshold = config.dedupe.first_name_similarity_threshold

                ok = (score >= high) or (
                    signals.first_similarity >= first_threshold and score >= relaxed
                )

                either_nameless = not (left.first_name and left.last_name) or not (
                    right.first_name and right.last_name
                )
                if either_nameless and not signals.has_corroborator:
                    ok = False

                if left.first_name and right.first_name:
                    left_first = normalize_text_key(left.first_name)
                    right_first = normalize_text_key(right.first_name)
                    nickname_eq = evaluator.nickname_equivalence and nickname_equivalent(
                        left.first_name, right.first_name
                    )
                    names_align = bool(left_first and right_first and left_first == right_first)
                    linkedin_match = bool(
                        left.linkedin_url and left.linkedin_url == right.linkedin_url
                    )
                    if not (names_align or nickname_eq or signals.emails_overlap or linkedin_match):
                        ok = False

                if left.source.lower() == "linkedin" or right.source.lower() == "linkedin":
                    if not signals.emails_overlap:
                        last_eq = normalize_text_key(left.last_name) == normalize_text_key(
                            right.last_name
                        )
                        gen_eq = normalize_text_key(left.suffix) == normalize_text_key(right.suffix)
                        first_eq = normalize_text_key(left.first_name) == normalize_text_key(
                            right.first_name
                        )
                        nickname_eq = evaluator.nickname_equivalence and nickname_equivalent(
                            left.first_name, right.first_name
                        )
                        if not (last_eq and (first_eq or nickname_eq) and gen_eq):
                            ok = False

                if config.dedupe.require_corroborator:
                    ok = ok and signals.has_corroborator

                if ok:
                    union(indices[i], indices[j])

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(records)):
        clusters[find(idx)].append(idx)
    return clusters


def _normalize_phone_value(value: str, default_country: str) -> Tuple[str, bool]:
    raw_value = value or ""
    cleaned_primary = format_phone_e164_safe(raw_value, default_country=default_country)
    if cleaned_primary and is_valid_phone_safe(cleaned_primary):
        return cleaned_primary, True
    compact = re.sub(r"\s+", "", raw_value)
    if compact and compact != raw_value:
        cleaned_compact = format_phone_e164_safe(compact, default_country=default_country)
        if cleaned_compact and is_valid_phone_safe(cleaned_compact):
            return cleaned_compact, True
    trimmed = raw_value.strip()
    return trimmed, False


def _merge_cluster(
    indices: List[int],
    normalized_records: List[ContactRecord],
    raw_records: List[ContactRecord],
    default_country: str,
) -> Tuple[ContactRecord, List[LineageEntry]]:
    cluster_records = [normalized_records[idx] for idx in indices]
    best_first, _ = choose_best_first_name(cluster_records)

    template = cluster_records[0]
    middle = _choose_by_priority(cluster_records, lambda r: r.middle_name)
    last = _choose_by_priority(cluster_records, lambda r: r.last_name)
    maiden = _choose_by_priority(cluster_records, lambda r: r.maiden_name)
    suffix = _choose_by_priority(cluster_records, lambda r: r.suffix)
    prof_suffixes = _choose_by_priority(cluster_records, lambda r: r.suffix_professional)
    company = _choose_by_priority(cluster_records, lambda r: r.company)
    title = _choose_by_priority(cluster_records, lambda r: r.title)
    linkedin = _choose_by_priority(cluster_records, lambda r: r.linkedin_url)

    all_emails: Dict[str, str] = {}
    all_phones: Dict[str, str] = {}
    cluster_invalid_emails: Set[str] = set()
    cluster_non_standard_phones: Set[str] = set()
    all_addresses: List[Dict[str, str]] = []
    seen_addr_keys: set[str] = set()
    for idx, record in zip(indices, cluster_records):
        record_extra = record.extra or {}
        cluster_invalid_emails.update(record_extra.get("invalid_emails", []))
        cluster_non_standard_phones.update(record_extra.get("non_standard_phones", []))
        for email in record.emails:
            all_emails[email.value] = email.label
        for phone in record.phones:
            normalized_value, is_confident = _normalize_phone_value(phone.value, default_country)
            if not normalized_value:
                continue
            if not is_confident:
                rendered = f"{normalized_value}::{phone.label}" if phone.label else normalized_value
                cluster_non_standard_phones.add(rendered)
                continue
            existing_label = all_phones.get(normalized_value)
            if existing_label:
                # Prefer non-empty labels or keep existing confident label
                if phone.label and not existing_label:
                    all_phones[normalized_value] = phone.label
            else:
                all_phones[normalized_value] = phone.label
        for address in record.addresses:
            as_dict = address.to_dict()
            key = json.dumps(as_dict, sort_keys=True)
            if key not in seen_addr_keys:
                seen_addr_keys.add(key)
                all_addresses.append(as_dict)

    deduped_addresses_json = json.dumps(all_addresses, ensure_ascii=False)

    full_name_clean = " ".join(filter(None, [best_first, middle, last, suffix])).strip()
    lineage_keys = [
        f"{record.source}:{record.source_row_id}"
        for record in cluster_records
        if record.source and record.source_row_id
    ]

    key_material = "::".join(
        [
            full_name_clean,
            company,
            title,
            ";".join(sorted(all_emails.keys())),
            ";".join(sorted(all_phones.keys())),
            "|".join(sorted(lineage_keys)),
        ]
    ).strip()
    contact_id = deterministic_uuid(key_material or full_name_clean or template.source_row_id)

    merged = ContactRecord(
        contact_id=contact_id,
        full_name=full_name_clean,
        first_name=best_first,
        middle_name=middle,
        last_name=last,
        maiden_name=maiden,
        suffix=suffix,
        suffix_professional=prof_suffixes,
        company=company,
        title=title,
        linkedin_url=linkedin,
        emails=[Email(value=value, label=all_emails[value]) for value in sorted(all_emails.keys())],
        phones=[Phone(value=value, label=all_phones[value]) for value in sorted(all_phones.keys())],
        addresses=[],
    )
    merged.addresses = [Address.from_mapping(address) for address in all_addresses]

    lineage_entries: List[LineageEntry] = []
    for idx in indices:
        record = normalized_records[idx]
        raw_record = raw_records[idx]
        record_extra = record.extra or {}
        lineage_entries.append(
            LineageEntry(
                contact_id=contact_id,
                source=record.source,
                source_row_id=record.source_row_id,
                source_full_name=record.full_name_raw,
                source_company=record.company,
                source_title=record.title,
                source_emails="|".join(email.value for email in record.emails),
                source_phones="|".join(phone.value for phone in record.phones),
                source_addresses_json=json.dumps(
                    [address.to_dict() for address in record.addresses], ensure_ascii=False
                ),
                source_emails_raw="|".join(email.value for email in raw_record.emails),
                source_phones_raw="|".join(phone.value for phone in raw_record.phones),
                invalid_emails="|".join(record_extra.get("invalid_emails", [])),
                non_standard_phones="|".join(record_extra.get("non_standard_phones", [])),
            )
        )

    unique_sources = {record.source for record in cluster_records if record.source}
    merged.extra["addresses_json"] = deduped_addresses_json
    merged.extra["source_count"] = len(unique_sources) or len(cluster_records)
    merged.extra["source_row_count"] = len(cluster_records)
    if cluster_invalid_emails:
        merged.extra["invalid_emails"] = sorted(cluster_invalid_emails)
        logger.info(
            "Contact %s dropped %d invalid email(s): %s",
            contact_id,
            len(cluster_invalid_emails),
            ", ".join(list(cluster_invalid_emails)[:5]),
        )
    if cluster_non_standard_phones:
        merged.extra["non_standard_phones"] = sorted(cluster_non_standard_phones)
        logger.info(
            "Contact %s flagged %d non-standard phone(s): %s",
            contact_id,
            len(cluster_non_standard_phones),
            ", ".join(list(cluster_non_standard_phones)[:5]),
        )

    return merged, lineage_entries


def build(
    args: argparse.Namespace, config: Optional[PipelineConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = config or load_config(args)
    normalization_settings = _build_normalization_settings(config)

    raw_records = _load_sources(config)
    normalized_records = _normalize_records(raw_records, normalization_settings)

    evaluator = MergeEvaluator(nickname_equivalence=config.dedupe.enable_nickname_equivalence)
    clusters = _cluster_indices(normalized_records, evaluator, config)

    merged_contacts: List[ContactRecord] = []
    lineage_records: List[LineageEntry] = []
    for indices in clusters.values():
        merged, lineage = _merge_cluster(
            indices,
            normalized_records,
            raw_records,
            normalization_settings.default_phone_country,
        )
        merged_contacts.append(merged)
        lineage_records.extend(lineage)

    contacts_rows = []
    for record in merged_contacts:
        extra = record.extra or {}
        contacts_rows.append(
            {
                "contact_id": record.contact_id,
                "full_name": record.full_name,
                "first_name": record.first_name,
                "middle_name": record.middle_name,
                "last_name": record.last_name,
                "maiden_name": record.maiden_name,
                "suffix": record.suffix,
                "suffix_professional": record.suffix_professional,
                "company": record.company,
                "title": record.title,
                "linkedin_url": record.linkedin_url,
                "emails": "|".join(f"{email.value}::{email.label}" for email in record.emails),
                "phones": "|".join(f"{phone.value}::{phone.label}" for phone in record.phones),
                "addresses_json": extra.get("addresses_json", "[]"),
                "invalid_emails": "|".join(sorted(set(extra.get("invalid_emails", [])))),
                "non_standard_phones": "|".join(sorted(set(extra.get("non_standard_phones", [])))),
                "source_count": extra.get("source_count", 1),
                "source_row_count": extra.get("source_row_count", extra.get("source_count", 1)),
            }
        )

    lineage_rows = [entry.to_dict() for entry in lineage_records]

    contacts_df = pd.DataFrame(contacts_rows)
    if not contacts_df.empty:
        duplicates = contacts_df[contacts_df["contact_id"].duplicated(keep=False)]
        if not duplicates.empty:
            duplicate_ids = ", ".join(sorted(duplicates["contact_id"].unique())[:5])
            raise ValueError(
                f"duplicate contact_id detected in consolidated output: {duplicate_ids}"
            )

    lineage_df = pd.DataFrame(lineage_rows)
    return contacts_df, lineage_df


def main() -> int:

    parser = argparse.ArgumentParser(description="Consolidate contacts from multiple sources.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--linkedin-csv", type=str, default=None)
    parser.add_argument("--gmail-csv", type=str, default=None)
    parser.add_argument("--mac-vcf", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--default-phone-country", type=str, default=None)
    parser.add_argument("--first-name-similarity-threshold", type=float, default=None)
    parser.add_argument("--merge-score-threshold", type=float, default=None)
    parser.add_argument("--relaxed-merge-threshold", type=float, default=None)
    parser.add_argument("--require-corroborator", action="store_true")
    parser.add_argument(
        "--nickname-equivalence",
        dest="enable_nickname_equivalence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable nickname equivalence when matching first names (default: on).",
    )
    parser.add_argument("--keep-generational-suffixes", nargs="*", default=None)
    parser.add_argument("--professional-suffixes", nargs="*", default=None)
    parser.add_argument("--log-level", type=str, default=None, help="Override logging level")
    args = parser.parse_args()

    config = load_config(args)
    configure_logging(config, level_override=args.log_level)
    contacts_df, lineage_df = build(args, config=config)

    out_dir = config.outputs.dir
    contacts_path = out_dir / "consolidated_contacts.csv"
    lineage_path = out_dir / "consolidated_lineage.csv"
    contacts_df.to_csv(str(contacts_path), index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    lineage_df.to_csv(str(lineage_path), index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    logger.info("Saved: %s", contacts_path)
    logger.info("Saved: %s", lineage_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
