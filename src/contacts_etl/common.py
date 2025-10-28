from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from .config_loader import PipelineConfig, load_pipeline_config
from .merge import MergeEvaluator, MergeSignals
from .models import Address, ContactRecord, Email, LineageEntry, Phone
from .normalization import (
    NormalizationSettings,
    address_keys_for_match,
    format_phone_e164_safe,
    guess_name_from_email_local,
    is_valid_phone_safe,
    nickname_equivalent,
    normalize_contact_record,
    normalize_country_iso2,
    normalize_email_collection,
    normalize_phone_collection,
    normalize_state,
    normalize_text_key,
    parse_name_multi_last,
    read_csv_with_optional_header,
    reconcile_name_from_email_and_last,
    safe_get,
    seq_ratio,
    strip_emails_from_text_and_capture,
    strip_suffixes_and_parse_name,
    uniq_list_of_dicts,
    validate_email_safe,
    warn_missing,
    choose_best_first_name,
)

__all__ = [
    "Address",
    "ContactRecord",
    "Email",
    "LineageEntry",
    "MergeEvaluator",
    "MergeSignals",
    "NormalizationSettings",
    "Phone",
    "PipelineConfig",
    "address_keys_for_match",
    "deterministic_uuid",
    "format_phone_e164_safe",
    "guess_name_from_email_local",
    "is_valid_phone_safe",
    "load_pipeline_config",
    "nickname_equivalent",
    "normalize_contact_record",
    "normalize_country_iso2",
    "normalize_email_collection",
    "normalize_phone_collection",
    "normalize_text_key",
    "normalize_state",
    "parse_name_multi_last",
    "read_csv_with_optional_header",
    "reconcile_name_from_email_and_last",
    "safe_get",
    "seq_ratio",
    "strip_emails_from_text_and_capture",
    "strip_suffixes_and_parse_name",
    "uniq_list_of_dicts",
    "validate_email_safe",
    "warn_missing",
    "choose_best_first_name",
]


def deterministic_uuid(namespace_str: str) -> str:
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, namespace_str))


def load_config(args: Any) -> PipelineConfig:
    return load_pipeline_config(args)


def to_contact_record(payload: Dict[str, Any]) -> ContactRecord:
    return ContactRecord.from_mapping(payload)


def ensure_contact_record(obj: Any) -> ContactRecord:
    if isinstance(obj, ContactRecord):
        return obj
    if isinstance(obj, dict):
        return ContactRecord.from_mapping(obj)
    raise TypeError(f"Unsupported contact payload type: {type(obj)!r}")
