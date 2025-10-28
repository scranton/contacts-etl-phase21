from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from .common import load_config, read_csv_with_optional_header, safe_get, warn_missing
from .config_loader import PipelineConfig
from .logging_utils import configure_logging
from .tagging import TagEngine, TaggingSettings

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CITIES = [
    "braintree",
    "quincy",
    "weymouth",
    "dedham",
    "milton",
    "hingham",
    "needham",
    "brookline",
    "cambridge",
    "somerville",
    "boston",
]


def _load_gmail_notes(path: Optional[str]) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    if not path:
        return notes
    if warn_missing(path, "GMail"):
        return notes
    df = read_csv_with_optional_header(path)
    if "Notes" not in df.columns:
        return notes
    for idx, row in df.iterrows():
        note = safe_get(row, "Notes")
        if note:
            notes[str(idx)] = note
    return notes


def _load_vcf_notes(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if warn_missing(path, "Mac VCF"):
        return {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    results: Dict[str, str] = {}
    blocks = content.split("END:VCARD")
    idx = 0
    for block in blocks:
        if "BEGIN:VCARD" not in block:
            continue
        for line in block.splitlines():
            if line.startswith("NOTE:"):
                results[str(idx)] = line[5:].strip()
                break
        idx += 1
    return results


def _resolve_paths(args: argparse.Namespace, config: PipelineConfig) -> tuple[str, str, str]:
    outputs_dir = config.outputs.dir
    contacts_csv = (
        getattr(args, "contacts_csv", None)
        or config.inputs.get("contacts_csv")
        or str(outputs_dir / "consolidated_contacts.csv")
    )
    lineage_csv = getattr(args, "lineage_csv", None) or str(
        outputs_dir / "consolidated_lineage.csv"
    )
    out_dir = getattr(args, "out_dir", None) or outputs_dir
    return contacts_csv, lineage_csv, str(out_dir)


def _build_notes_map(
    lineage_csv: str, gmail_notes: Dict[str, str], vcf_notes: Dict[str, str]
) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    try:
        lineage_df = pd.read_csv(
            lineage_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        logger.warning("Unable to build notes map from lineage: %s", exc)
        return notes
    for contact_id, chunk in lineage_df.groupby("contact_id"):
        snippets: List[str] = []
        for _, row in chunk.iterrows():
            source = safe_get(row, "source")
            row_id = safe_get(row, "source_row_id")
            if source == "gmail" and row_id in gmail_notes:
                snippets.append(gmail_notes[row_id])
            elif source == "mac_vcf" and row_id in vcf_notes:
                snippets.append(vcf_notes[row_id])
        if snippets:
            notes[str(contact_id)] = " | ".join(snippets)
    return notes


def build(args: argparse.Namespace, config: Optional[PipelineConfig] = None):
    config = config or load_config(args)
    contacts_csv, lineage_csv, out_dir = _resolve_paths(args, config)

    gmail_notes = _load_gmail_notes(config.inputs.get("gmail_csv"))
    vcf_notes = _load_vcf_notes(config.inputs.get("mac_vcf"))
    notes_map = _build_notes_map(lineage_csv, gmail_notes, vcf_notes)

    df = pd.read_csv(contacts_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    confidence_path = config.outputs.dir / "confidence_report.csv"
    if confidence_path.exists():
        conf_df = pd.read_csv(
            confidence_path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL
        )[["contact_id", "confidence_score"]]
        df = df.merge(conf_df, on="contact_id", how="left")

    tagging_settings = TaggingSettings(
        prior_companies=config.tagging.prior_companies or [],
        prior_domains=config.tagging.prior_domains or [],
        local_cities=config.tagging.local_cities or DEFAULT_LOCAL_CITIES,
    )
    tag_engine = TagEngine(tagging_settings)

    tag_values = []
    primary_values = []
    notes_values = []
    sanitized_rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sanitized = {col: safe_get(row, col) for col in df.columns}
        contact_id = sanitized.get("contact_id", "")
        if contact_id in notes_map:
            sanitized["notes_blob"] = notes_map[contact_id]
        else:
            sanitized.setdefault("notes_blob", "")
        tags, primary = tag_engine.tag_record(sanitized)
        tag_str = "|".join(sorted(tags)) if tags else ""
        tag_values.append(tag_str)
        primary_values.append(primary)
        notes_values.append(sanitized.get("notes_blob", ""))
        sanitized["tags"] = tag_str
        sanitized_rows.append(sanitized)

    df["tags"] = tag_values
    df["relationship_category"] = primary_values
    if "confidence_score" not in df.columns:
        df["confidence_score"] = 0
    df["notes_blob"] = notes_values

    confidence_as_str = df["confidence_score"].astype(str).tolist()
    for sanitized_record, confidence in zip(sanitized_rows, confidence_as_str):
        sanitized_record["confidence_score"] = confidence

    try:
        df["referral_priority_score"] = [
            TagEngine.compute_referral_priority(record) for record in sanitized_rows
        ]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unable to calculate referral_priority_score: %s", exc)
        df["referral_priority_score"] = None

    out_contacts = os.path.join(out_dir, "tagged_contacts.csv")
    df.to_csv(out_contacts, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    top = df.sort_values(["referral_priority_score", "confidence_score"], ascending=[False, False])
    out_targets = os.path.join(out_dir, "referral_targets.csv")
    top.to_csv(out_targets, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(f"Saved: {out_contacts}")
    print(f"Saved: {out_targets}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Tag and categorize contacts; compute referral priority."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contacts-csv", type=str, default=None)
    parser.add_argument("--lineage-csv", type=str, default=None)
    parser.add_argument("--gmail-csv", type=str, default=None)
    parser.add_argument("--mac-vcf", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None, help="Override logging level")
    args = parser.parse_args()
    config = load_config(args)
    configure_logging(config, level_override=args.log_level)
    return build(args, config=config)


if __name__ == "__main__":
    raise SystemExit(main())
