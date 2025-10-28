import argparse
import csv
import os
from typing import Dict, Tuple

import pandas as pd

from contacts_etl.common import load_config, warn_missing
from contacts_etl.config_loader import PipelineConfig
from contacts_etl.logging_utils import configure_logging


def pct(n, d):
    return round((n / d * 100.0), 2) if d else 0.0


def load_validation_map(validation_csv) -> Dict[str, Dict[str, int]]:
    """Returns mapping contact_id -> metrics dict with int values."""
    v: Dict[str, Dict[str, int]] = {}
    if warn_missing(validation_csv, "validation CSV"):
        return v
    df = pd.read_csv(validation_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    # Cast numeric fields safely
    for col in [
        "email_valid_count",
        "email_total",
        "phone_valid_count",
        "phone_total",
        "addr_valid_count",
        "addr_total",
        "quality_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for _, r in df.iterrows():
        v[str(r["contact_id"])] = {
            "email_valid_count": int(r.get("email_valid_count", 0)),
            "email_total": int(r.get("email_total", 0)),
            "phone_valid_count": int(r.get("phone_valid_count", 0)),
            "phone_total": int(r.get("phone_total", 0)),
            "addr_valid_count": int(r.get("addr_valid_count", 0)),
            "addr_total": int(r.get("addr_total", 0)),
            "quality_score": int(r.get("quality_score", 0)),
        }
    return v


def compute_corroborators(row):
    """Count distinct corroborators present: email, phone, address, linkedin"""
    corr = 0
    if str(row.get("emails", "")).strip():
        corr += 1
    if str(row.get("phones", "")).strip():
        corr += 1
    if (
        str(row.get("addresses_json", "")).strip()
        and str(row.get("addresses_json", "")).strip() != "[]"
    ):
        corr += 1
    if str(row.get("linkedin_url", "")).strip():
        corr += 1
    return corr


def has_company_title(row):
    return 1 if (str(row.get("company", "")).strip() or str(row.get("title", "")).strip()) else 0


def confidence_score(row, vmap: Dict[str, Dict[str, int]]) -> int:
    """0-100, additive with caps. Weighted for practicality."""
    cid = str(row.get("contact_id", ""))
    vm: Dict[str, int] = vmap.get(cid, {})
    email_valid = 0 < vm.get("email_total", 0) == vm.get("email_valid_count", 0)
    phone_valid = 0 < vm.get("phone_total", 0) == vm.get("phone_valid_count", 0)
    addr_any_valid = vm.get("addr_valid_count", 0) > 0
    lineage_depth = int(row.get("source_count", 1))

    score: float = 0.0

    # Base: previous quality score (scaled to 0-40)
    base_quality = min(int(vm.get("quality_score", 0)), 100)
    score += round(base_quality * 0.4, 0)

    # Corroborators (0-20)
    score += min(compute_corroborators(row) * 5, 20)

    # Lineage depth bonus (up to +10)
    if lineage_depth >= 3:
        score += 10
    elif lineage_depth == 2:
        score += 6
    else:
        score += 2

    # LinkedIn + company/title (up to +15)
    if str(row.get("linkedin_url", "")).strip():
        score += 8
    if has_company_title(row):
        score += 7

    # All-valid channels bonus (up to +10)
    if email_valid:
        score += 5
    if phone_valid:
        score += 3
    if addr_any_valid:
        score += 2

    # Consistency heuristics (up to +5): name + last name present, non-empty full_name
    if str(row.get("first_name", "")).strip() and str(row.get("last_name", "")).strip():
        score += 3
    if str(row.get("full_name", "")).strip():
        score += 2

    return int(max(0, min(100, score)))


def _resolve_paths(args: argparse.Namespace, config: PipelineConfig) -> Tuple[str, str, str]:
    outputs_dir = config.outputs.dir
    contacts_csv = (
        getattr(args, "contacts_csv", None)
        or config.inputs.get("contacts_csv")
        or str(outputs_dir / "consolidated_contacts.csv")
    )
    validation_csv = getattr(args, "validation_csv", None) or str(
        outputs_dir / "validation_report.csv"
    )
    out_dir = getattr(args, "out_dir", None) or outputs_dir
    return contacts_csv, validation_csv, str(out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compute confidence scores and summary for consolidated contacts."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contacts-csv", type=str, default=None)
    parser.add_argument("--validation-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None, help="Override logging level")

    args = parser.parse_args()
    config = load_config(args)
    configure_logging(config, level_override=args.log_level)
    contacts_csv, validation_csv, out_dir = _resolve_paths(args, config)

    # Load data
    contacts_df = pd.read_csv(contacts_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    vmap = load_validation_map(validation_csv)

    # Compute per-contact confidence
    scores = []
    for _, row in contacts_df.iterrows():
        s = confidence_score(row, vmap)
        scores.append(s)
    contacts_df["confidence_score"] = scores
    # Buckets
    bins = []
    for s in scores:
        if s >= 80:
            bins.append("very_high")
        elif s >= 60:
            bins.append("high")
        elif s >= 40:
            bins.append("medium")
        else:
            bins.append("low")
    contacts_df["confidence_bucket"] = bins

    # Save per-contact report
    out_contact = os.path.join(out_dir, "confidence_report.csv")
    contacts_df.to_csv(out_contact, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # Summary
    total = len(contacts_df)
    bucket_counts: Dict[str, int] = contacts_df["confidence_bucket"].value_counts().to_dict()
    bucket_pct: Dict[str, float] = (
        contacts_df["confidence_bucket"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )

    summary: Dict[str, object] = {
        "total_contacts": total,
        "avg_confidence": round(sum(scores) / total, 2) if total else 0.0,
        "bucket_counts": bucket_counts,
        "bucket_pct": bucket_pct,
    }
    # Save summary CSV
    rows = []
    for b in ["very_high", "high", "medium", "low"]:
        rows.append(
            {
                "bucket": b,
                "count": bucket_counts.get(b, 0),
                "pct": bucket_pct.get(b, 0.0),
            }
        )
    out_summary = os.path.join(out_dir, "confidence_summary.csv")
    pd.DataFrame(rows).to_csv(out_summary, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(summary)
    print(f"Saved: {out_contact}")
    print(f"Saved: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
