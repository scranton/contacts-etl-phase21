from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict, OrderedDict
from dataclasses import replace
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    normalize_country_iso2,
    normalize_text_key,
    read_csv_with_optional_header,
    safe_get,
    warn_missing,
)
from .config_loader import PipelineConfig
from .logging_utils import configure_logging
from .models import Address, Phone
from .normalization import STATE_ABBR

# use module logger instead of configuring logging at import time
logger = logging.getLogger(__name__)

DEFAULT_GEN = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
DEFAULT_PROF = {
    "phd",
    "pmp",
    "csm",
    "spc6",
    "ccim",
    "phr",
    "shrm",
    "shrmcp",
    "cp",
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
    "ms",
    "rdn",
    "ld",
    "mpa",
    "ise",
    "md",
    "mph",
}

DEFAULT_PREFIXES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "mx",
    "dr",
    "prof",
    "sir",
    "madam",
    "madame",
    "lady",
    "lord",
    "rev",
    "reverend",
    "fr",
    "father",
    "pastor",
    "rabbi",
    "imam",
    "hon",
    "judge",
}


PHONE_VALUE_PATTERN = re.compile(r"\+?\d[\d\s()./-]{6,}\d")
GOOGLE_MULTI_VALUE_SPLIT = re.compile(r":::+")
POSTAL_CODE_PATTERN = re.compile(r"\b[0-9A-Za-z]{3,10}(?:-[0-9A-Za-z]{3,4})?\b")
STATE_POSTAL_PATTERN = re.compile(
    r"^\s*([A-Za-z]{2})[\s,]+(\d{3,10}(?:-[0-9A-Za-z]{3,4})?)\s*$"
)
CITY_STATE_POSTAL_PATTERN = re.compile(
    r"^\s*(.+?)[,\s]+([A-Za-z]{2})[\s,]+(\d{3,10}(?:-[0-9A-Za-z]{3,4})?)\s*$"
)
STATE_CODE_SET = set(STATE_ABBR.values())
PHONE_EXTENSION_PATTERN = re.compile(r"^(?:ext\.?|extension|x)?\s*(\d{1,6})$", re.IGNORECASE)
PHONE_INLINE_EXTENSION_PATTERN = re.compile(
    r"^(?P<number>.+?)(?:[\s,;/]*(?:ext\.?|extension|x)\s*(?P<ext>\d{1,6})"
    r"|p(?P<ext2>\d{1,6})#)\s*$",
    re.IGNORECASE,
)
STATE_NAME_SET = set(STATE_ABBR.keys())
COUNTRY_TOKENS = {
    "united states",
    "united states of america",
    "usa",
    "us",
    "canada",
    "mexico",
    "united kingdom",
    "uk",
    "england",
    "scotland",
    "wales",
    "northern ireland",
}
STREET_KEYWORDS = {
    "street",
    "st",
    "st.",
    "road",
    "rd",
    "rd.",
    "avenue",
    "ave",
    "ave.",
    "boulevard",
    "blvd",
    "blvd.",
    "lane",
    "ln",
    "ln.",
    "drive",
    "dr",
    "dr.",
    "court",
    "ct",
    "ct.",
    "circle",
    "cir",
    "cir.",
    "way",
    "parkway",
    "pkwy",
    "pkwy.",
    "highway",
    "hwy",
    "hwy.",
    "trail",
    "trl",
    "trl.",
    "loop",
    "plaza",
    "plz",
    "suite",
    "ste",
    "unit",
    "apt",
    "apartment",
    "floor",
    "fl",
    "building",
    "bldg",
    "bldg.",
}
_GMAIL_LABEL_WARNED: Set[str] = set()

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
    name_prefixes = config.normalization.name_prefixes or list(DEFAULT_PREFIXES)
    return NormalizationSettings.from_args(
        keep_suffixes,
        prof_suffixes,
        name_prefixes,
        config.normalization.default_phone_country,
        config.normalization.drop_invalid_emails,
        config.normalization.drop_invalid_phones,
        config.normalization.email_dns_mx_check,
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


def _extract_phone_values(raw: str) -> List[Tuple[str, str]]:
    if not raw:
        return []
    candidates: List[Tuple[str, str]] = []
    for part in re.split(r"[\r\n|;]+", raw):
        part = part.strip()
        if not part:
            continue
        segments = _split_google_multi_values(part) or [part]
        for segment in segments:
            base_segment, inline_ext = _strip_phone_extension(segment)
            matches = PHONE_VALUE_PATTERN.findall(base_segment)
            if matches:
                for idx, match in enumerate(matches):
                    ext = inline_ext if inline_ext and idx == len(matches) - 1 else ""
                    candidates.append((match.strip(), ext))
            else:
                stripped = base_segment.strip()
                if stripped:
                    candidates.append((stripped, inline_ext))
    cleaned = [candidate for candidate in candidates if candidate[0]]
    return _merge_phone_extensions(cleaned)


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _normalize_phone_label(label: str) -> str:
    return _normalize_label(label)


def _parse_gmail_label(raw_label: str, channel: str) -> Tuple[str, bool]:
    label = (raw_label or "").strip()
    is_preferred = False
    if label.startswith("*"):
        is_preferred = True
        label = label.lstrip("*").strip()
    lowered = label.lower()
    normalized = ""
    phone_mobile_tokens = {"mobile", "cell", "iphone"}
    if channel == "phone" and any(token in lowered for token in phone_mobile_tokens):
        normalized = "mobile"
    elif "work" in lowered:
        normalized = "work"
    elif "home" in lowered:
        normalized = "home"
    elif "other" in lowered:
        normalized = "other"
    elif lowered:
        normalized = "other"
        if lowered not in _GMAIL_LABEL_WARNED:
            logger.info("GMail %s label normalized to other: %s", channel, label)
            _GMAIL_LABEL_WARNED.add(lowered)
    return normalized, is_preferred


def _record_phone(
    phone_map: "OrderedDict[Tuple[str, str], str]",
    raw_value: str,
    label: str,
    extension: str = "",
) -> None:
    value = (raw_value or "").strip()
    ext = (extension or "").strip()
    if not value:
        return
    label_norm = _normalize_phone_label(label)
    key = (value, ext)
    current = phone_map.get(key)
    if current is None or (not current and label_norm):
        phone_map[key] = label_norm


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


def _split_google_multi_values(raw: str) -> List[str]:
    if not raw:
        return []
    segments = [segment.strip() for segment in GOOGLE_MULTI_VALUE_SPLIT.split(raw)]
    return [segment for segment in segments if segment]


def _unescape_vcard_value(value: str) -> str:
    if not value:
        return ""
    replacements = {
        r"\\;": ";",
        r"\\,": ",",
        r"\\n": "\n",
        r"\\N": "\n",
        r"\\\\": "\\",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _strip_phone_extension(segment: str) -> Tuple[str, str]:
    segment = segment.strip()
    if ";" in segment:
        base, candidate = segment.rsplit(";", 1)
        candidate = candidate.strip()
        if candidate.isdigit() and 1 <= len(candidate) <= 6:
            return base.strip(), candidate
    match = PHONE_INLINE_EXTENSION_PATTERN.match(segment)
    if match:
        ext_value = match.group("ext") or match.group("ext2")
        if ext_value:
            number = (match.group("number") or "").strip(" ,;/")
            extension = ext_value.strip()
            if number:
                return number, extension
    return segment, ""


def _merge_phone_extensions(values: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    merged: List[Tuple[str, str]] = []
    for value, extension in values:
        stripped_value = (value or "").strip()
        extension = (extension or "").strip()
        if not stripped_value:
            if extension and merged and not merged[-1][1]:
                prev_value, _ = merged[-1]
                merged[-1] = (prev_value, extension)
            continue
        match = PHONE_EXTENSION_PATTERN.match(stripped_value)
        if not extension and match and merged and not merged[-1][1]:
            prev_value, _ = merged[-1]
            merged[-1] = (prev_value, match.group(1))
            continue
        merged.append((stripped_value, extension))
    return merged


def _format_phone_with_extension(value: str, extension: str) -> str:
    return f"{value}x{extension}" if extension else value


def _extract_email_values(raw: str) -> List[str]:
    if not raw:
        return []
    candidates: List[str] = []
    for part in re.split(r"[\r\n|;]+", raw):
        part = part.strip()
        if not part:
            continue
        subparts = _split_google_multi_values(part) or [part]
        candidates.extend(subparts)
    return [candidate for candidate in candidates if candidate]


def _expand_address_variants(components: Dict[str, str]) -> List[Dict[str, str]]:
    split_components = {
        field: _split_google_multi_values(value) for field, value in components.items()
    }
    max_len = max((len(values) for values in split_components.values()), default=0)
    if max_len <= 1:
        return [
            {
                field: (values[0] if values else "")
                for field, values in split_components.items()
            }
        ]

    variants: List[Dict[str, str]] = []
    for idx in range(max_len):
        variant: Dict[str, str] = {}
        for field, values in split_components.items():
            if values:
                variant[field] = values[idx] if idx < len(values) else values[-1]
            else:
                variant[field] = ""
        variants.append(variant)
    return variants


def _split_address_lines(value: str) -> List[str]:
    if not value:
        return []
    lines: List[str] = []
    for chunk in re.split(r"[\r\n]+", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        subparts = _split_google_multi_values(chunk) or [chunk]
        lines.extend(subparts)
    return lines


def _looks_like_country(value: str) -> bool:
    lowered = (value or "").strip().lower()
    if not lowered:
        return False
    if lowered in COUNTRY_TOKENS:
        return True
    country = normalize_country_iso2(lowered)
    if country and country.lower() != lowered:
        return True
    return False


def _detect_state_token(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered in STATE_ABBR:
        return STATE_ABBR[lowered]
    if len(token) == 2 and token.isalpha():
        upper = token.upper()
        if upper in STATE_CODE_SET:
            return upper
    return ""


def _is_probable_street_line(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in STATE_NAME_SET or lowered in STATE_CODE_SET or lowered in COUNTRY_TOKENS:
        return False
    if CITY_STATE_POSTAL_PATTERN.match(text):
        return False
    if "," in text:
        before, after = text.split(",", 1)
        if _detect_state_token(after):
            return False
    if POSTAL_CODE_PATTERN.fullmatch(text):
        return False
    if _looks_like_country(text):
        return False
    if any(char.isdigit() for char in text):
        return True
    tokens = re.split(r"[\s,]+", lowered)
    return any(token in STREET_KEYWORDS for token in tokens if token)


def _maybe_extract_city_line_details(city_value: str, components: Dict[str, str]) -> None:
    text = (city_value or "").strip()
    if not text:
        return
    match = CITY_STATE_POSTAL_PATTERN.match(text)
    if match:
        city_candidate, state_candidate, postal_candidate = match.groups()
        if city_candidate:
            components["city"] = city_candidate.strip()
        if state_candidate and not components["state"]:
            state_detected = _detect_state_token(state_candidate)
            if state_detected:
                components["state"] = state_detected
        if postal_candidate and not components["postal_code"]:
            components["postal_code"] = postal_candidate.strip()
        return
    if "," in text and not components["state"]:
        before, after = text.split(",", 1)
        state_candidate = _detect_state_token(after)
        if state_candidate:
            components["city"] = before.strip() or components["city"]
            components["state"] = state_candidate


def _prepare_gmail_address_components(row: Any, addr_id: str) -> Dict[str, str]:
    components = {
        "po_box": safe_get(row, f"Address {addr_id} - PO Box"),
        "extended": safe_get(row, f"Address {addr_id} - Extended Address"),
        "street": safe_get(row, f"Address {addr_id} - Street"),
        "city": safe_get(row, f"Address {addr_id} - City"),
        "state": safe_get(row, f"Address {addr_id} - Region"),
        "postal_code": safe_get(row, f"Address {addr_id} - Postal Code"),
        "country": safe_get(row, f"Address {addr_id} - Country"),
    }
    components = {field: (value or "").strip() for field, value in components.items()}

    line_candidates = _split_address_lines(components["street"])
    redundancies = {
        components["city"].lower(),
        components["state"].lower(),
        components["postal_code"].lower(),
        components["country"].lower(),
    }
    redundancies = {value for value in redundancies if value}

    if line_candidates:
        should_replace_street = (
            not components["street"] or "\n" in components["street"] or "\r" in components["street"]
        )
        remaining = list(line_candidates)
        if should_replace_street:
            street_line = ""
            street_idx = None
            for idx, candidate in enumerate(remaining):
                lowered = candidate.lower()
                if lowered in redundancies:
                    continue
                if _is_probable_street_line(candidate):
                    street_line = candidate
                    street_idx = idx
                    break
            if street_idx is not None:
                remaining = remaining[street_idx + 1 :]
                components["street"] = street_line
            else:
                # fall back to the first non-redundant line if we never detected a street
                while remaining:
                    candidate = remaining.pop(0)
                    if candidate.lower() in redundancies:
                        continue
                    street_line = candidate
                    break
                components["street"] = street_line
        else:
            remaining = remaining[1:] if remaining else []

        additional_street_parts: List[str] = []
        filtered_remaining: List[str] = []
        for candidate in remaining:
            lowered = candidate.lower()
            if lowered in redundancies:
                continue
            if _is_probable_street_line(candidate):
                additional_street_parts.append(candidate)
            else:
                filtered_remaining.append(candidate)
        remaining = filtered_remaining

        if additional_street_parts:
            parts = [part for part in [components["street"], *additional_street_parts] if part]
            components["street"] = ", ".join(parts)

        if remaining and not components["city"]:
            city_line = remaining.pop(0)
            components["city"] = city_line
            _maybe_extract_city_line_details(city_line, components)

        for line in remaining:
            if not line:
                continue
            assigned = False
            match = STATE_POSTAL_PATTERN.match(line)
            if match:
                state_candidate, postal_candidate = match.groups()
                if state_candidate and not components["state"]:
                    normalized_state = _detect_state_token(state_candidate)
                    components["state"] = normalized_state or state_candidate.strip()
                if postal_candidate and not components["postal_code"]:
                    components["postal_code"] = postal_candidate.strip()
                assigned = True
            if not assigned and not components["state"]:
                normalized_state = _detect_state_token(line)
                if normalized_state:
                    components["state"] = normalized_state
                    assigned = True
            if not assigned and not components["postal_code"]:
                postal_match = POSTAL_CODE_PATTERN.search(line)
                if postal_match:
                    components["postal_code"] = postal_match.group(0).strip()
                    assigned = True
            if not assigned and not components["country"]:
                normalized_country = normalize_country_iso2(line)
                if normalized_country:
                    components["country"] = line.strip()
                    assigned = True
            if not assigned and not components["city"]:
                components["city"] = line.strip()
    return components


def _load_gmail_csv(path: Optional[str]) -> List[ContactRecord]:
    if not path:
        return []
    if warn_missing(path, "GMail"):
        return []
    df = read_csv_with_optional_header(path)
    records: List[ContactRecord] = []
    for idx, row in df.iterrows():
        email_map: "OrderedDict[str, str]" = OrderedDict()
        preferred_channels: Dict[str, List[Any]] = {"emails": [], "phones": [], "addresses": []}
        for column in row.index:
            if not str(column).startswith("E-mail ") or not str(column).endswith(" - Value"):
                continue
            email_value_raw = safe_get(row, column)
            if not email_value_raw:
                continue
            label_col = str(column).replace(" - Value", " - Label")
            email_label_raw = safe_get(row, label_col)
            email_label, email_pref = _parse_gmail_label(email_label_raw, "email")
            for extracted in _extract_email_values(email_value_raw):
                _record_email(email_map, extracted, email_label)
                if email_pref:
                    preferred_channels["emails"].append(extracted)
        emails = [Email(value=value, label=label) for value, label in email_map.items()]

        phones_map: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        for column in row.index:
            if not str(column).startswith("Phone ") or not str(column).endswith(" - Value"):
                continue
            phone_value_raw = safe_get(row, column)
            if not phone_value_raw:
                continue
            label_col = str(column).replace(" - Value", " - Label")
            phone_label_raw = safe_get(row, label_col)
            phone_label, phone_pref = _parse_gmail_label(phone_label_raw, "phone")
            for extracted_value, phone_extension in _extract_phone_values(phone_value_raw):
                _record_phone(phones_map, extracted_value, phone_label, phone_extension)
                if phone_pref:
                    preferred_channels["phones"].append(extracted_value)

        phones = [
            Phone(value=value, extension=extension, label=label)
            for (value, extension), label in phones_map.items()
        ]

        address_map: "OrderedDict[str, Address]" = OrderedDict()
        address_ids: Set[str] = set()
        for column in row.index:
            match = re.match(r"Address (\d+) - ", str(column))
            if match:
                address_ids.add(match.group(1))
        for addr_id in sorted(address_ids, key=lambda value: int(value)):
            addr_label_raw = safe_get(row, f"Address {addr_id} - Label")
            addr_label, addr_pref = _parse_gmail_label(addr_label_raw, "address")
            components = _prepare_gmail_address_components(row, addr_id)
            for variant in _expand_address_variants(components):
                address_entry = Address(
                    po_box=variant.get("po_box", ""),
                    extended=variant.get("extended", ""),
                    street=variant.get("street", ""),
                    city=variant.get("city", ""),
                    state=variant.get("state", ""),
                    postal_code=variant.get("postal_code", ""),
                    country=variant.get("country", ""),
                    label=addr_label,
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
                        if addr_pref:
                            preferred_channels["addresses"].append(address_entry.to_dict())
        addresses = list(address_map.values())
        first_name = safe_get(row, "First Name")
        middle_name = safe_get(row, "Middle Name")
        last_name = safe_get(row, "Last Name")
        name_prefix = safe_get(row, "Name Prefix")
        name_suffix = safe_get(row, "Name Suffix")
        raw_full = " ".join(
            part
            for part in [name_prefix, first_name, middle_name, last_name, name_suffix]
            if part
        ).strip()
        extra_payload: Dict[str, Any] = {}
        preferred_clean = {k: v for k, v in preferred_channels.items() if v}
        if preferred_clean:
            extra_payload["preferred_channels"] = preferred_clean
        record = ContactRecord(
            full_name_raw=raw_full,
            prefix=name_prefix,
            suffix=name_suffix,
            nickname=safe_get(row, "Nickname"),
            company=safe_get(row, "Organization Name"),
            title=safe_get(row, "Organization Title"),
            source="gmail",
            source_row_id=str(idx),
            emails=emails,
            phones=phones,
            addresses=addresses,
            notes=safe_get(row, "Notes"),
            extra=extra_payload,
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
        phone_map: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        address_map: "OrderedDict[str, Address]" = OrderedDict()
        email_map: "OrderedDict[str, str]" = OrderedDict()
        for raw_line in b.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("FN:"):
                record.full_name_raw = line[3:].strip()
            elif line.startswith("N:"):
                components = line[2:].split(";")
                record.last_name = components[0].strip() if len(components) > 0 else ""
                record.first_name = components[1].strip() if len(components) > 1 else ""
                record.middle_name = components[2].strip() if len(components) > 2 else ""
                record.prefix = components[3].strip() if len(components) > 3 else ""
                record.suffix = components[4].strip() if len(components) > 4 else ""
                if not record.full_name_raw:
                    record.full_name_raw = " ".join(
                        filter(
                            None,
                            [
                                record.prefix,
                                record.first_name,
                                record.middle_name,
                                record.last_name,
                                record.suffix,
                            ],
                        )
                    ).strip()
            elif ":" in line and line.split(":", 1)[0].upper().split(";")[0].endswith("NICKNAME"):
                record.nickname = line.split(":", 1)[1].strip()
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
                value = _unescape_vcard_value(value)
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
                base_value, inline_ext = _strip_phone_extension(value.strip())
                _record_phone(phone_map, base_value, label, inline_ext)
            elif line.startswith("ADR"):
                if ":" not in line:
                    continue
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
        record.phones = [
            Phone(value=value, extension=extension, label=label)
            for (value, extension), label in phone_map.items()
        ]
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

                left_name_candidates = MergeEvaluator.first_name_candidates(left)
                right_name_candidates = MergeEvaluator.first_name_candidates(right)

                def _has_core_name(record: ContactRecord, candidates: List[str]) -> bool:
                    return bool(candidates and record.last_name)

                either_nameless = (
                    not _has_core_name(left, left_name_candidates)
                    or not _has_core_name(right, right_name_candidates)
                )
                if either_nameless and not signals.has_corroborator:
                    ok = False

                if left_name_candidates and right_name_candidates:
                    names_align = any(
                        normalize_text_key(a) == normalize_text_key(b)
                        for a in left_name_candidates
                        for b in right_name_candidates
                        if a and b
                    )
                    nickname_eq = evaluator.nickname_equivalence and any(
                        nickname_equivalent(a, b)
                        for a in left_name_candidates
                        for b in right_name_candidates
                    )
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
                        first_eq = False
                        if left_name_candidates and right_name_candidates:
                            first_eq = any(
                                normalize_text_key(a) == normalize_text_key(b)
                                for a in left_name_candidates
                                for b in right_name_candidates
                                if a and b
                            )
                        nickname_eq = evaluator.nickname_equivalence and any(
                            nickname_equivalent(a, b)
                            for a in left_name_candidates
                            for b in right_name_candidates
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
    prefix = _choose_by_priority(cluster_records, lambda r: r.prefix)
    middle = _choose_by_priority(cluster_records, lambda r: r.middle_name)
    last = _choose_by_priority(cluster_records, lambda r: r.last_name)
    maiden = _choose_by_priority(cluster_records, lambda r: r.maiden_name)
    suffix = _choose_by_priority(cluster_records, lambda r: r.suffix)
    prof_suffixes = _choose_by_priority(cluster_records, lambda r: r.suffix_professional)
    nickname_value = _choose_by_priority(cluster_records, lambda r: r.nickname)
    company = _choose_by_priority(cluster_records, lambda r: r.company)
    title = _choose_by_priority(cluster_records, lambda r: r.title)
    linkedin = _choose_by_priority(cluster_records, lambda r: r.linkedin_url)

    all_emails: Dict[str, str] = {}
    all_phones: Dict[Tuple[str, str], str] = {}
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
                rendered_value = _format_phone_with_extension(
                    normalized_value, phone.extension
                )
                rendered = f"{rendered_value}::{phone.label}" if phone.label else rendered_value
                cluster_non_standard_phones.add(rendered)
                key = (rendered_value, phone.extension or "")
                if key not in all_phones:
                    all_phones[key] = phone.label or "invalid"
                continue
            key = (normalized_value, phone.extension or "")
            existing_label = all_phones.get(key)
            if existing_label:
                # Prefer non-empty labels or keep an existing confident label
                if phone.label and not existing_label:
                    all_phones[key] = phone.label
            else:
                all_phones[key] = phone.label
        for address in record.addresses:
            as_dict = address.to_dict()
            addr_key = json.dumps(as_dict, sort_keys=True)
            if addr_key not in seen_addr_keys:
                seen_addr_keys.add(addr_key)
                all_addresses.append(as_dict)

    deduped_addresses_json = json.dumps(all_addresses, ensure_ascii=False)

    full_name_clean = " ".join(filter(None, [prefix, best_first, middle, last, suffix])).strip()
    lineage_keys = [
        f"{record.source}:{record.source_row_id}"
        for record in cluster_records
        if record.source and record.source_row_id
    ]

    phone_key_components = [
        _format_phone_with_extension(value, extension) for value, extension in all_phones.keys()
    ]
    key_material = "::".join(
        [
            full_name_clean,
            company,
            title,
            ";".join(sorted(all_emails.keys())),
            ";".join(sorted(phone_key_components)),
            "|".join(sorted(lineage_keys)),
        ]
    ).strip()
    contact_id = deterministic_uuid(key_material or full_name_clean or template.source_row_id)

    merged = ContactRecord(
        contact_id=contact_id,
        full_name=full_name_clean,
        prefix=prefix,
        first_name=best_first,
        middle_name=middle,
        last_name=last,
        maiden_name=maiden,
        suffix=suffix,
        suffix_professional=prof_suffixes,
        nickname=nickname_value,
        company=company,
        title=title,
        linkedin_url=linkedin,
        emails=[Email(value=value, label=all_emails[value]) for value in sorted(all_emails.keys())],
        phones=[
            Phone(value=value, extension=extension, label=all_phones[(value, extension)])
            for value, extension in sorted(all_phones.keys())
        ],
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
                source_prefix=record.prefix,
                source_company=record.company,
                source_title=record.title,
                source_emails="|".join(email.value for email in record.emails),
                source_phones="|".join(
                    _format_phone_with_extension(phone.value, phone.extension)
                    for phone in record.phones
                ),
                source_addresses_json=json.dumps(
                    [address.to_dict() for address in record.addresses], ensure_ascii=False
                ),
                source_emails_raw="|".join(email.value for email in raw_record.emails),
                source_phones_raw="|".join(
                    _format_phone_with_extension(phone.value, phone.extension)
                    for phone in raw_record.phones
                ),
            )
        )

    unique_sources = {record.source for record in cluster_records if record.source}
    merged.extra["addresses_json"] = deduped_addresses_json
    merged.extra["source_count"] = len(unique_sources) or len(cluster_records)
    merged.extra["source_row_count"] = len(cluster_records)
    if cluster_invalid_emails:
        logger.info(
            "Contact %s encountered %d invalid email(s): %s",
            contact_id,
            len(cluster_invalid_emails),
            ", ".join(list(cluster_invalid_emails)[:5]),
        )
    if cluster_non_standard_phones:
        logger.info(
            "Contact %s encountered %d non-standard phone(s): %s",
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
                "prefix": record.prefix,
                "first_name": record.first_name,
                "middle_name": record.middle_name,
                "last_name": record.last_name,
                "maiden_name": record.maiden_name,
                "suffix": record.suffix,
                "suffix_professional": record.suffix_professional,
                "nickname": record.nickname,
                "company": record.company,
                "title": record.title,
                "linkedin_url": record.linkedin_url,
                "emails": "|".join(f"{email.value}::{email.label}" for email in record.emails),
                "phones": "|".join(
                    f"{_format_phone_with_extension(phone.value, phone.extension)}::{phone.label}"
                    for phone in record.phones
                ),
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
    parser.add_argument("--name-prefixes", nargs="*", default=None)
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
