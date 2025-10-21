import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Set

import pandas as pd

MARTIAL_KWS = [r"tai\s*chi", r"wu\s*an", r"wu\s*dao", r"kung\s*fu", r"shaolin", r"martial\s*arts"]
NUTCRACKER_KWS = [r"nutcracker", r"\bcherub(s)?\b", r"jose\s*mateo", r"ballet"]
# Default local cities; override in config
DEFAULT_LOCAL_CITIES = ["braintree","quincy","weymouth","dedham","milton","hingham","needham","brookline","cambridge","somerville","boston"]

def any_kw(text, patterns):
    t = (text or "").lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False

def extract_domain_set(emails_field):
    domains: Set[str] = set()
    if not isinstance(emails_field, str) or not emails_field.strip():
        return domains
    parts = [p for p in emails_field.split("|") if p.strip()]
    for p in parts:
        email = p.split("::")[0].strip().lower()
        if "@" in email:
            domains.add(email.split("@")[1])
    return domains

def tag_row(row, cfg):
    tags=set()
    text_blob = " ".join([
        str(row.get("company","")), str(row.get("title","")), str(row.get("linkedin_url",""))
    ]).lower()

    # from lineage_notes added by optional enrichment (if present)
    if "notes_blob" in row and row["notes_blob"]:
        text_blob += " " + str(row["notes_blob"]).lower()

    # martial arts
    if any_kw(text_blob, MARTIAL_KWS):
        tags.add("martial_arts")

    # nutcracker
    if any_kw(text_blob, NUTCRACKER_KWS):
        tags.add("nutcracker_performance")

    # work colleague: via prior companies or email domains
    prior_companies = [c.lower() for c in cfg.get("prior_companies", [])]
    prior_domains = [d.lower() for d in cfg.get("prior_domains", [])]
    company = str(row.get("company","")).lower()
    if company and any(pc in company for pc in prior_companies):
        tags.add("work_colleague")
    domains = extract_domain_set(row.get("emails",""))
    if any(any(pd1 in dom for pd1 in prior_domains) for dom in domains):
        tags.add("work_colleague")

    # local cities from addresses_json
    try:
        addrs = json.loads(row.get("addresses_json","") or "[]")
    except Exception:
        addrs = []
    local_cities = [c.lower() for c in cfg.get("local_cities", DEFAULT_LOCAL_CITIES)]
    for a in addrs:
        city = str(a.get("city","")).lower()
        if city and any(city == lc or lc in city for lc in local_cities):
            tags.add("local_south_shore"); break

    # primary relationship category: personal > professional > local_referral
    primary = ""
    if "martial_arts" in tags or "nutcracker_performance" in tags:
        primary = "personal"
    elif "work_colleague" in tags or str(row.get("linkedin_url","")).strip():
        primary = "professional"
    elif "local_south_shore" in tags:
        primary = "local_referral"
    else:
        primary = "uncategorized"

    return tags, primary

def referral_priority(row, confidence_weight=0.6, tag_weights=None):
    """
    Combine confidence_score (0-100) with tag proximity.
    tag_weights default:
      martial_arts +30, nutcracker +25, work_colleague +20, local_south_shore +10
    """
    if tag_weights is None:
        tag_weights = {"martial_arts":30, "nutcracker_performance":25, "work_colleague":20, "local_south_shore":10}
    conf: float = 0.0
    try:
        conf = float(row.get("confidence_score", 0) or 0)
    except Exception:
        conf = 0.0
    score: float = conf * confidence_weight
    tags = set(str(row.get("tags","")).split("|")) if row.get("tags") else set()
    score += sum(tag_weights.get(t, 0) for t in tags)
    # cap at 100
    return int(min(100, round(score, 0)))

def load_gmail_notes(path):
    """Return dict (source_row_id -> notes text) for gmail CSV."""
    notes: Dict[str, str] = {}
    if not path or not os.path.exists(path): return notes
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "Notes" not in df.columns: return notes
    for i, row in df.iterrows():
        n = str(row.get("Notes","")).strip()
        if n:
            notes[str(i)] = n
    return notes


def load_vcf_notes(path):
    """Basic NOTE extraction from VCF, returns dict of (index -> note)."""
    results: Dict[str, str] = {}
    if not path or not os.path.exists(path): return results
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = re.split(r"END:VCARD", content)
    idx = 0
    for b in blocks:
        if "BEGIN:VCARD" not in b: continue
        m = re.search(r"^NOTE:(.+)$", b, flags=re.MULTILINE)
        if m:
            results[str(idx)] = m.group(1).strip()
        idx += 1
    return results

def build(args):
    import yaml  # type: ignore
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    outputs = cfg.get("outputs", {})
    tagging_cfg = cfg.get("tagging", {})
    inputs = cfg.get("inputs", {})

    contacts_csv = args.contacts_csv or os.path.join(outputs.get("dir", os.getcwd()), "consolidated_contacts.csv")
    lineage_csv = args.lineage_csv or os.path.join(outputs.get("dir", os.getcwd()), "consolidated_lineage.csv")
    out_dir = args.out_dir or outputs.get("dir", os.getcwd())

    # optional sources for notes
    gmail_csv = args.gmail_csv or inputs.get("gmail_csv")
    mac_vcf = args.mac_vcf or inputs.get("mac_vcf")

    df = pd.read_csv(contacts_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
    # join confidence if present
    conf_path = os.path.join(outputs.get("dir", os.getcwd()), "confidence_report.csv")
    if os.path.exists(conf_path):
        conf_df = pd.read_csv(conf_path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)[["contact_id","confidence_score"]]
        df = df.merge(conf_df, on="contact_id", how="left")

    # Build notes blob per contact by looking up lineage → source row notes
    notes_map = {}
    try:
        lin = pd.read_csv(lineage_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)
        gmail_notes = load_gmail_notes(gmail_csv)
        vcf_notes = load_vcf_notes(mac_vcf)
        for cid, chunk in lin.groupby("contact_id"):
            texts=[]
            for _, r in chunk.iterrows():
                src = str(r.get("source",""))
                rid = str(r.get("source_row_id",""))
                if src == "gmail" and rid in gmail_notes:
                    texts.append(gmail_notes[rid])
                elif src == "mac_vcf" and rid in vcf_notes:
                    texts.append(vcf_notes[rid])
            if texts:
                notes_map[cid] = " | ".join(texts)
    except Exception:
        pass

    # Apply tagging
    tag_list=[]; primary_list=[]; notes_blob_list=[]
    for _, row in df.iterrows():
        cid = row.get("contact_id","")
        if cid in notes_map:
            row["notes_blob"] = notes_map[cid]
        tags, primary = tag_row(row, {
            "prior_companies": tagging_cfg.get("prior_companies", []),
            "prior_domains": tagging_cfg.get("prior_domains", []),
            "local_cities": tagging_cfg.get("local_cities", DEFAULT_LOCAL_CITIES)
        })
        tag_list.append("|".join(sorted(tags)) if tags else "")
        primary_list.append(primary)
        notes_blob_list.append(notes_map.get(cid, ""))

    df["tags"] = tag_list
    df["relationship_category"] = primary_list
    if "confidence_score" not in df.columns:
        df["confidence_score"] = 0
    df["notes_blob"] = notes_blob_list

    # referral priority
    df["referral_priority_score"] = [referral_priority(r) for r in df.to_dict(orient="records")]

    # Save outputs
    out_contacts = os.path.join(out_dir, "tagged_contacts.csv")
    df.to_csv(out_contacts, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # top targets
    top = df.copy()
    top = top.sort_values(["referral_priority_score","confidence_score"], ascending=[False, False])
    out_targets = os.path.join(out_dir, "referral_targets.csv")
    top.to_csv(out_targets, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(f"Saved: {out_contacts}")
    print(f"Saved: {out_targets}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Tag and categorize contacts; compute referral priority.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contacts-csv", type=str, default=None)
    parser.add_argument("--lineage-csv", type=str, default=None)
    parser.add_argument("--gmail-csv", type=str, default=None)
    parser.add_argument("--mac-vcf", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    return build(args)

if __name__ == "__main__":
    raise SystemExit(main())
