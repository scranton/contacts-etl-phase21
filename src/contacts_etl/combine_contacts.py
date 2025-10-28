from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common import (
    ContactRecord,
    Email,
    LineageEntry,
    MergeEvaluator,
    NormalizationSettings,
    choose_best_first_name,
    deterministic_uuid,
    load_config,
    nickname_equivalent,
    normalize_contact_record,
    normalize_text_key,
    read_csv_with_optional_header,
    safe_get,
    warn_missing,
)
from .config_loader import PipelineConfig
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


def _build_normalization_settings(config: PipelineConfig) -> NormalizationSettings:
    keep_suffixes = config.normalization.keep_generational_suffixes or list(DEFAULT_GEN)
    prof_suffixes = config.normalization.professional_suffixes or list(DEFAULT_PROF)
    return NormalizationSettings.from_args(
        keep_suffixes, prof_suffixes, config.normalization.default_phone_country
    )


def _load_linkedin_csv(path: Optional[str]) -> List[ContactRecord]:
    if warn_missing(path, "LinkedIn"):
        return []
    df = read_csv_with_optional_header(path, header_starts_with="First Name,Last Name,URL")
    records: List[ContactRecord] = []
    for idx, row in df.iterrows():
        linkedin_url = safe_get(row, "URL")
        if "linkedin.com" not in linkedin_url.lower():
            linkedin_url = ""
        email_value = safe_get(row, "Email Address")
        emails = [Email(value=email_value, label="work")] if email_value else []
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


def _load_gmail_csv(path: Optional[str]) -> List[ContactRecord]:
    if warn_missing(path, "GMail"):
        return []
    df = read_csv_with_optional_header(path)
    records: List[ContactRecord] = []
    for idx, row in df.iterrows():
        emails: List[Email] = []
        for n in range(1, 5):
            address = safe_get(row, f"E-mail {n} - Value")
            if not address:
                continue
            label = safe_get(row, f"E-mail {n} - Type").lower()
            emails.append(Email(value=address, label=label))
        phones: List[Phone] = []
        for n in range(1, 5):
            number = safe_get(row, f"Phone {n} - Value")
            if not number:
                continue
            label = safe_get(row, f"Phone {n} - Label").lower()
            phones.append(Phone(value=number, label=label))
        addresses: List[Address] = []
        for n in range(1, 4):
            address = Address(
                po_box=safe_get(row, f"Address {n} - PO Box"),
                extended="",  # or pull a real field if one exists in the CSV
                street=safe_get(row, f"Address {n} - Street")
                or safe_get(row, f"Address {n} - Formatted"),
                city=safe_get(row, f"Address {n} - City"),
                state=safe_get(row, f"Address {n} - Region"),
                postal_code=safe_get(row, f"Address {n} - Postal Code"),
                country=safe_get(row, f"Address {n} - Country"),
                label=safe_get(row, f"Address {n} - Label"),
            )
            if any(
                getattr(address, field)
                for field in ("street", "city", "state", "postal_code", "country", "po_box")
            ):
                addresses.append(address)
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
    if warn_missing(path, "Mac VCF"):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    blocks = [block for block in content.split("END:VCARD") if "BEGIN:VCARD" in block]
    records: List[ContactRecord] = []
    for idx, block in enumerate(blocks):
        b = f"{block}END:VCARD"
        record = ContactRecord(source="mac_vcf", source_row_id=str(idx))
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
            elif "EMAIL" in line and ":" in line:
                value = line.split(":", 1)[1].strip()
                record.emails.append(Email(value=value, label=""))
            elif "TEL" in line and ":" in line:
                value = line.split(":", 1)[1].strip()
                record.phones.append(Phone(value=value, label=""))
            elif line.startswith("ADR"):
                value = line.split(":", 1)[1].split(";")
                address = Address(
                    po_box=value[0].strip() if len(value) > 0 else "",
                    extended=value[1].strip() if len(value) > 1 else "",
                    street=value[2].strip() if len(value) > 2 else "",
                    city=value[3].strip() if len(value) > 3 else "",
                    state=value[4].strip() if len(value) > 4 else "",
                    postal_code=value[5].strip() if len(value) > 5 else "",
                    country=value[6].strip() if len(value) > 6 else "",
                )
                record.addresses.append(address)
            elif line.startswith("ORG:"):
                record.company = line[4:].strip()
            elif line.startswith("TITLE:"):
                record.title = line[6:].strip()
            elif line.startswith("URL:") and "linkedin.com" in line.lower():
                record.linkedin_url = line[4:].strip()
            elif line.startswith("NOTE:"):
                record.notes = line[5:].strip()
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
        key = normalize_text_key(record.last_name) or f"__blank_{idx}"
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


def _merge_cluster(
    indices: List[int], records: List[ContactRecord]
) -> Tuple[ContactRecord, List[LineageEntry]]:
    cluster_records = [records[idx] for idx in indices]
    best_first, _ = choose_best_first_name(cluster_records)

    template = cluster_records[0]
    middle = next((record.middle_name for record in cluster_records if record.middle_name), "")
    last = next((record.last_name for record in cluster_records if record.last_name), "")
    maiden = next((record.maiden_name for record in cluster_records if record.maiden_name), "")
    suffix = next((record.suffix for record in cluster_records if record.suffix), "")
    prof_suffixes = next(
        (record.suffix_professional for record in cluster_records if record.suffix_professional), ""
    )
    company = next((record.company for record in cluster_records if record.company), "")
    title = next((record.title for record in cluster_records if record.title), "")
    linkedin = next((record.linkedin_url for record in cluster_records if record.linkedin_url), "")

    all_emails: Dict[str, str] = {}
    all_phones: Dict[str, str] = {}
    all_addresses: List[Dict[str, str]] = []
    seen_addr_keys: set[str] = set()
    for record in cluster_records:
        for email in record.emails:
            all_emails[email.value] = email.label
        for phone in record.phones:
            all_phones[phone.value] = phone.label
        for address in record.addresses:
            as_dict = address.to_dict()
            key = json.dumps(as_dict, sort_keys=True)
            if key not in seen_addr_keys:
                seen_addr_keys.add(key)
                all_addresses.append(as_dict)

    deduped_addresses_json = json.dumps(all_addresses, ensure_ascii=False)

    full_name_clean = " ".join(filter(None, [best_first, middle, last, suffix])).strip()
    emails_str = "|".join(f"{value}::{label}" for value, label in sorted(all_emails.items()))
    phones_str = "|".join(f"{value}::{label}" for value, label in sorted(all_phones.items()))

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

    lineage_entries = []
    for idx, record in zip(indices, cluster_records):
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
            )
        )

    unique_sources = {record.source for record in cluster_records if record.source}
    merged.extra["addresses_json"] = deduped_addresses_json
    merged.extra["source_count"] = len(unique_sources) or len(cluster_records)
    merged.extra["source_row_count"] = len(cluster_records)

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
        merged, lineage = _merge_cluster(indices, normalized_records)
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
    logging.basicConfig(level=logging.INFO)

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
    args = parser.parse_args()

    config = load_config(args)
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
