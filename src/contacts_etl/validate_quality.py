import argparse
import csv
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml  # type: ignore[import-untyped]

from .common import load_config
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


def pct(n, d):
    return round((n / d * 100.0), 2) if d else 0.0


def validate_email_list(emails_field: object) -> Tuple[int, int, List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    if not isinstance(emails_field, str) or not emails_field.strip():
        return 0, 0, details
    parts = [p for p in emails_field.split("|") if p.strip()]
    valid_count: int = 0
    for p in parts:
        email = p.split("::")[0].strip()
        label = p.split("::")[1].strip() if "::" in p else ""
        lowered_label = label.lower()
        is_valid = bool(email) and lowered_label != "invalid"

        valid_count += 1 if is_valid else 0
        details.append({"email": email, "label": label, "valid": is_valid})
    return valid_count, len(parts), details


def validate_phone_list(phones_field: str) -> Tuple[int, int, List[str]]:
    """
    Validate a pipe-delimited phones field like "value::label|value2::label2".
    Returns (valid_count, total_count, valid_values_list).
    Treats any entry with label 'invalid' as invalid (regardless of value).
    """
    if not phones_field:
        return 0, 0, []

    parts = [p for p in str(phones_field).split("|") if p and p.strip()]
    total = len(parts)
    valid_values: List[str] = []

    for p in parts:
        val = p.split("::", 1)[0].strip()
        label = p.split("::", 1)[1].strip() if "::" in p else ""
        if val and label.lower() != "invalid":
            valid_values.append(val)

    return len(valid_values), total, valid_values


def parse_addresses_json(addresses_json: object) -> Tuple[int, int, List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    if not isinstance(addresses_json, str) or not addresses_json.strip():
        return 0, 0, details
    try:
        addrs = json.loads(addresses_json)
    except (ValueError, json.JSONDecodeError, TypeError):
        logger.info("Skipping malformed address payload: %s", addresses_json[:120])
        return 0, 0, details
    valid_count = 0
    for a in addrs:
        street = (a.get("street", "") or "").strip()
        city = (a.get("city", "") or "").strip()
        state = (a.get("state", "") or "").strip()
        postal = (a.get("postal_code", "") or "").strip()
        country = (a.get("country", "") or "").strip()
        is_valid = bool(street) and bool(city or postal)
        details.append(
            {
                "street": street,
                "city": city,
                "state": state,
                "postal_code": postal,
                "country": country,
                "valid": is_valid,
            }
        )
        valid_count += 1 if is_valid else 0
    return valid_count, len(addrs), details


def _load_flattened_map(path: str) -> Dict[str, Dict[str, str]]:
    flattened: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return flattened
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:  # pragma: no cover
        logger.warning("Unable to read flattened contacts: %s", exc)
        return flattened
    for _, row in df.iterrows():
        cid = str(row.get("contact_id", ""))
        if cid:
            flattened[cid] = {str(col): str(row[col]) for col in df.columns}
    return flattened


def main():
    parser = argparse.ArgumentParser(description="Validate & score consolidated contacts.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contacts-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None, help="Override logging level")

    args = parser.parse_args()
    config = load_config(args)
    configure_logging(config, level_override=args.log_level)
    quality_cfg = getattr(config, "quality", None)
    email_full_score = getattr(quality_cfg, "email_full_score", 40)
    email_partial_score = getattr(quality_cfg, "email_partial_score", 20)
    phone_full_score = getattr(quality_cfg, "phone_full_score", 30)
    phone_partial_score = getattr(quality_cfg, "phone_partial_score", 15)
    address_any_score = getattr(quality_cfg, "address_any_score", 30)
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    outputs: Dict[str, Any] = cfg.get("outputs", {}) or {}

    contacts_csv = args.contacts_csv or os.path.join(
        outputs.get("dir", os.getcwd()), "consolidated_contacts.csv"
    )
    out_dir = args.out_dir or outputs.get("dir", os.getcwd())
    df = pd.read_csv(contacts_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    flattened_map = _load_flattened_map(os.path.join(out_dir, "flattened_contacts.csv"))

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        e_valid, e_total, e_details = validate_email_list(row.get("emails", ""))
        p_valid, p_total, p_details = validate_phone_list(row.get("phones", ""))
        a_valid, a_total, a_details = parse_addresses_json(row.get("addresses_json", ""))
        cid = str(row.get("contact_id", ""))
        department_value = str(row.get("department", "") or "").strip()
        flattened = flattened_map.get(cid, {})
        home_email_present = 1 if str(flattened.get("home_email", "")).strip() else 0
        work_email_present = 1 if str(flattened.get("work_email", "")).strip() else 0
        home_phone_present = 1 if str(flattened.get("home_phone", "")).strip() else 0
        work_phone_present = 1 if str(flattened.get("work_phone", "")).strip() else 0
        home_address_present = 1 if str(flattened.get("home_address", "")).strip() else 0
        work_address_present = 1 if str(flattened.get("work_address", "")).strip() else 0
        rec = {
            "contact_id": cid,
            "full_name": row.get("full_name", ""),
            "company": row.get("company", ""),
            "title": row.get("title", ""),
            "department": department_value,
            "linkedin_url": row.get("linkedin_url", ""),
            "email_valid_count": e_valid,
            "email_total": e_total,
            "phone_valid_count": p_valid,
            "phone_total": p_total,
            "addr_valid_count": a_valid,
            "addr_total": a_total,
            "emails_detail": json.dumps(e_details, ensure_ascii=False),
            "phones_detail": json.dumps(p_details, ensure_ascii=False),
            "addresses_detail": json.dumps(a_details, ensure_ascii=False),
            "department_missing": 0 if department_value else 1,
            "home_email_present": home_email_present,
            "work_email_present": work_email_present,
            "home_phone_present": home_phone_present,
            "work_phone_present": work_phone_present,
            "home_address_present": home_address_present,
            "work_address_present": work_address_present,
        }
        # Simple score (can parametrize later)
        if 0 < e_total == e_valid:
            email_score = email_full_score
        elif e_valid > 0:
            email_score = email_partial_score
        else:
            email_score = 0

        if 0 < p_total == p_valid:
            phone_score = phone_full_score
        elif p_valid > 0:
            phone_score = phone_partial_score
        else:
            phone_score = 0

        addr_score = address_any_score if a_valid > 0 else 0
        rec["quality_score"] = email_score + phone_score + addr_score
        records.append(rec)

    validation_df = pd.DataFrame(records)
    out_validation = os.path.join(out_dir, "validation_report.csv")
    validation_df.to_csv(out_validation, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    scored_df = df.merge(
        validation_df[
            [
                "contact_id",
                "email_valid_count",
                "email_total",
                "phone_valid_count",
                "phone_total",
                "addr_valid_count",
                "addr_total",
                "quality_score",
                "department_missing",
                "home_email_present",
                "work_email_present",
                "home_phone_present",
                "work_phone_present",
                "home_address_present",
                "work_address_present",
            ]
        ],
        on="contact_id",
        how="left",
    )
    out_scored = os.path.join(out_dir, "contact_quality_scored.csv")
    scored_df.to_csv(out_scored, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    total = len(scored_df)
    has_any_email = sum(1 for x in scored_df["email_total"] if str(x).strip() and int(x or 0) > 0)
    has_any_phone = sum(1 for x in scored_df["phone_total"] if str(x).strip() and int(x or 0) > 0)
    has_any_addr = sum(1 for x in scored_df["addr_total"] if str(x).strip() and int(x or 0) > 0)
    print(
        {
            "contacts_total": total,
            "has_any_email_pct": round(has_any_email / total * 100, 2) if total else 0,
            "has_any_phone_pct": round(has_any_phone / total * 100, 2) if total else 0,
            "has_any_address_pct": round(has_any_addr / total * 100, 2) if total else 0,
        }
    )
    print(f"Saved: {out_validation}")
    print(f"Saved: {out_scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
