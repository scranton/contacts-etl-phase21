import argparse
import csv
import json
import logging
import os
import re
import uuid
import unicodedata
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Optional

import yaml

from collections import Counter, defaultdict
from pathlib import Path
from io import StringIO

import pandas as pd
from pandas import DataFrame

from .common import (
    normalize_state, normalize_country_iso2, strip_suffixes_and_parse_name, deterministic_uuid, seq_ratio
)

# Optional libs: import specific symbols when available to simplify call sites
try:
    from email_validator import validate_email, EmailNotValidError  # type: ignore
    HAS_EMAIL_VALIDATOR = True
except ImportError:
    # provide fallbacks so static analysis doesn't complain about undefined names
    validate_email = None  # type: ignore
    EmailNotValidError = Exception  # type: ignore
    HAS_EMAIL_VALIDATOR = False

try:
    import phonenumbers
    HAS_PHONENUMBERS = True
except ImportError:
    phonenumbers = None  # type: ignore
    HAS_PHONENUMBERS = False

# use module logger instead of configuring logging at import time
logger = logging.getLogger(__name__)

DEFAULT_GEN = {"jr","sr","ii","iii","iv","v","vi"}
DEFAULT_PROF = {"phd","pmp","csm","spc6","mba","cissp","crisc","cscp","cams","cpa","cfa","pe","cisa","cism","cfe","cma","ceh","itil","sixsigma","leansixsigma","esq","jd"}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# --- small reusable helpers to reduce repetition ---

def _sget(row: Any, key: str) -> str:
    """Safe string retrieval from a mapping/row and strip whitespace.

    Accepts dict-like objects and pandas Series.
    """
    try:
        return str(row.get(key, "") or "").strip()
    except Exception:
        # pandas.Series doesn't always implement dict.get with same signature
        try:
            return str(row[key] if key in row else "").strip()
        except Exception:
            return ""


def _warn_missing(path: Optional[str], label: str) -> bool:
    """Log a warning if path is missing; returns True if missing."""
    if not path or not os.path.exists(path):
        logger.warning("%s path missing: %s", label, path)
        return True
    return False


def read_csv_with_optional_header(path: Optional[str], header_starts_with: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV, but allow for a noisy prefix by scanning for a header line that starts with header_starts_with.

    If header_starts_with is None the file is read directly with pandas.
    If path is None, return an empty DataFrame.
    """
    if not path:
        return pd.DataFrame()
    if not header_starts_with:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read().splitlines()
    header_idx = None
    for i, line in enumerate(txt[:100]):
        if line.strip().startswith(header_starts_with):
            header_idx = i
            break
    if header_idx is None:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_csv(StringIO("\n".join(txt[header_idx:])), dtype=str, keep_default_na=False)


def uniq_list_of_dicts(lst: List[Dict[str, Any]], key: str = "value") -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in lst:
        v1 = d.get(key, "")
        if v1 and v1 not in seen:
            seen.add(v1)
            out.append(d)
    return out


# --- Nickname equivalence helpers (unchanged behavior) ---

def _norm(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).lower()


_NICKMAP = {
    "william": {"william","will","bill","billy","liam"},
    "robert": {"robert","rob","bob","bobby","robby"},
    "richard": {"richard","rich","rick","ricky","dick"},
    "edward": {"edward","ed","eddie","ted","teddy","ned"},
    "margaret": {"margaret","meg","maggie","peggy"},
    "elizabeth": {"elizabeth","liz","beth","lizzy","eliza","liza","betsy"},
    "katherine": {"katherine","kathy","kate","katie","cathy","cait"},
    "alexander": {"alexander","alex","sasha"},
    "james": {"james","jim","jimmy","jamie"},
    "john": {"john","jack","johnny"},
    "jonathan": {"jonathan","jon","john"},
    "joseph": {"joseph","joe","joey"},
    "matthew": {"matthew","matt"},
    "michael": {"michael","mike","mikey"},
    "jeffrey": {"jeffrey","jeff"},
    "andrew": {"andrew","andy","drew"},
    "steven": {"steven","steve","stevie"},
    "christopher": {"christopher","chris","kit"},
    "patrick": {"patrick","pat","paddy"},
    "nicholas": {"nicholas","nick","nicky"},
    "catherine": {"catherine","cathy","kate","katie"},
    "francis": {"francis","frank","frankie"},
}

# Reverse index for fast membership
_VARIANT_TO_ROOT = {}
for root, variants in _NICKMAP.items():
    for v in variants:
        _VARIANT_TO_ROOT[_norm(v)] = root


def nickname_root(first: str) -> str:
    """Return canonical root if first name is a known nickname; else normalized first name."""
    nm = _norm(first)
    return _VARIANT_TO_ROOT.get(nm, nm)


def nickname_equivalent(a: str, b: str) -> bool:
    """True if both names map to same nickname root (e.g., 'mike' vs 'michael')."""
    if not a or not b:
        return False
    return nickname_root(a) == nickname_root(b)


def full_name_key(rec: Dict[str, Any]) -> str:
    """Normalized 'first last [gen]' key; ignores middle; used for strict LN merges."""
    first = _norm(rec.get("first_name", ""))
    last = _norm(rec.get("last_name", ""))
    gen = _norm(rec.get("suffix", ""))
    parts = [p for p in (first, last, gen) if p]
    return " ".join(parts)


def is_linkedin_rec(rec: Dict[str, Any]) -> bool:
    return (rec.get("source", "") or "").lower() == "linkedin"


def is_email_like(text: Optional[str]) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(EMAIL_RE.match(t))


def strip_emails_from_text_and_capture(name: Optional[str], emails_accum: List[Dict[str, str]]):
    """Extract emails embedded in a text field, append to emails_accum, and return cleaned text."""
    if not name:
        return ""
    found = re.findall(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", name)
    for em in found:
        emails_accum.append({"value": em, "label": ""})
    out = name
    for em in found:
        out = out.replace(em, "").strip()
    return out


def guess_name_from_email_local(local: str) -> Tuple[str, str]:
    parts = [p for p in re.split(r"[._-]+", local) if p]
    first = parts[0].title() if parts else ""
    last = parts[1].title() if len(parts) > 1 else ""
    return first, last


def reconcile_name_from_email_and_last(local: str, last: str) -> str:
    """If local looks like first-initial and last (e.g., mtodd & Todd), return 'M'."""
    local = (local or "").lower()
    last = (last or "").lower()
    if last and local.endswith(last) and len(local) > len(last):
        prefix = local[:-len(last)]
        if 1 <= len(prefix) <= 2:
            return prefix[0].upper()
    return ""


def clean_email_val(e: Optional[str]) -> str:
    e = (e or "").strip()
    if not e:
        return ""
    if HAS_EMAIL_VALIDATOR:
        try:
            return validate_email(e, check_deliverability=False).normalized
        except EmailNotValidError:
            return ""
    e = e.replace(" ", "").lower()
    return e if re.match(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", e) else ""


def normalize_phone(num: Any, default_country: str = "US") -> str:
    if not num or str(num).strip() == "":
        return ""
    raw = str(num).strip()
    if HAS_PHONENUMBERS:
        try:
            pn = phonenumbers.parse(raw, None if raw.startswith("+") else default_country)
            return phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            # parsing failed for this number; fall back to heuristic
            pass
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if raw.startswith("+"):
        return re.sub(r"[^\d+]", "", raw)
    return f"+1{digits}" if digits else ""


def aggregate_unique(list_of_dicts: List[Dict[str, Any]], key: str = "value") -> List[Dict[str, Any]]:
    seen = set(); out = []
    for d in list_of_dicts:
        v1 = d.get(key, "")
        if v1 and v1 not in seen:
            seen.add(v1); out.append(d)
    return out


def load_linkedin_csv(path: Optional[str]) -> List[Dict[str, Any]]:
    if _warn_missing(path, "Linkedin"):
        return []
    df = read_csv_with_optional_header(path, header_starts_with="First Name,Last Name,URL")
    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        first = _sget(row, "First Name")
        last = _sget(row, "Last Name")
        raw_full = " ".join([first, last]).strip()
        rec: Dict[str, Any] = {
            "full_name_raw": raw_full, "full_name": "",
            "first_name": "", "middle_name": "", "last_name": "",
            "maiden_name": "", "suffix": "", "suffix_professional": "",
            "emails": [], "phones": [], "addresses": [],
            "company": _sget(row, "Company"),
            "title": _sget(row, "Position"),
            "linkedin_url": _sget(row, "URL") if "linkedin.com" in _sget(row, "URL").lower() else "",
            "source": "linkedin", "source_row_id": str(i)
        }
        em = _sget(row, "Email Address")
        if em:
            rec["emails"].append({"value": em, "label": "work"})
        if any([first, last, em, rec["company"], rec["title"], rec["linkedin_url"]]):
            rows.append(rec)
    return rows


def load_gmail_csv(path: Optional[str]) -> List[Dict[str, Any]]:
    if _warn_missing(path, "GMail"):
        return []
    df = read_csv_with_optional_header(path)
    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        raw_full = " ".join([_sget(row, "First Name"), _sget(row, "Middle Name"), _sget(row, "Last Name")]).strip()
        rec: Dict[str, Any] = {
            "full_name_raw": raw_full, "full_name": "",
            "first_name": "", "middle_name": "", "last_name": "",
            "maiden_name": "", "suffix": _sget(row, "Name Suffix"), "suffix_professional": "",
            "emails": [], "phones": [], "addresses": [],
            "company": _sget(row, "Organization Name"),
            "title": _sget(row, "Organization Title"),
            "linkedin_url": "", "source": "gmail", "source_row_id": str(i)
        }
        for n in range(1, 5):
            v2 = _sget(row, f"E-mail {n} - Value")
            if v2:
                rec["emails"].append({"value": v2, "label": _sget(row, f"E-mail {n} - Type").lower()})
        for n in range(1, 5):
            v2 = _sget(row, f"Phone {n} - Value")
            if v2:
                rec["phones"].append({"value": v2, "label": _sget(row, f"Phone {n} - Label").lower()})
        for n in range(1, 4):
            addr = {
                "po_box": _sget(row, f"Address { n } - PO Box"),
                "extended": "",
                "street": _sget(row, f"Address { n } - Street") or _sget(row, f"Address { n } - Formatted"),
                "city": _sget(row, f"Address { n } - City"),
                "state": _sget(row, f"Address { n } - Region"),
                "postal_code": _sget(row, f"Address { n } - Postal Code"),
                "country": _sget(row, f"Address { n } - Country"),
                "label": _sget(row, f"Address { n } - Label")
            }
            if any(addr[k] for k in ["street", "city", "state", "postal_code", "country", "po_box"]):
                rec["addresses"].append(addr)
        rows.append(rec)
    return rows


def vcard_unescape(val: str) -> str:
    # RFC 6350 unescaping
    return (val.replace("\\n", "\n")
               .replace("\\N", "\n")
               .replace("\\;", ";")
               .replace("\\,", ",")
               .replace("\\\\", "\\"))


def parse_org(value: str) -> str:
    # 1) Unescape vCard escapes
    v3 = vcard_unescape(value or "").strip()
    # 2) Split on semicolons (structured components)
    parts = [p.strip() for p in v3.split(";")]
    # 3) Keep the first non-empty component as "company"
    for p in parts:
        if p:
            return p
    return ""


def load_vcards(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        logger.warning("VCF path missing: %s", path)
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    rows: List[Dict[str, Any]] = []
    for block in re.split(r"END:VCARD", content):
        if "BEGIN:VCARD" not in block: continue
        b = block + "END:VCARD"
        d: Dict[str, Any] = {
            "full_name_raw": "", "full_name": "",
            "first_name": "", "middle_name": "", "last_name": "",
            "maiden_name": "", "suffix": "", "suffix_professional": "",
            "emails": [], "phones": [], "addresses": [],
            "company": "", "title": "", "linkedin_url": "",
            "source": "mac_vcf", "source_row_id": None
        }
        m = re.search(r"^FN:(.+)$", b, flags=re.MULTILINE)
        if m: d["full_name_raw"] = m.group(1).strip()
        m = re.search(r"^N:(.+)$", b, flags=re.MULTILINE)
        if m:
            parts = m.group(1).split(";")
            d["last_name"] = parts[0].strip() if len(parts) > 0 else ""
            d["first_name"] = parts[1].strip() if len(parts) > 1 else ""
            d["middle_name"] = parts[2].strip() if len(parts) > 2 else ""
            d["suffix"] = parts[3].strip() if len(parts) > 3 else ""
            if not d["full_name_raw"]:
                d["full_name_raw"] = " ".join([d["first_name"], d["middle_name"], d["last_name"], d["suffix"]]).strip()
        for em in re.finditer(r"^EMAIL.*:(.+)$", b, flags=re.MULTILINE):
            label = ""
            label_m = re.search(r"EMAIL;TYPE=([^:]+):", em.group(0))
            if label_m: label = label_m.group(1)
            d["emails"].append({"value": em.group(1).strip(), "label": label})
        for tel in re.finditer(r"^TEL.*:(.+)$", b, flags=re.MULTILINE):
            label = ""
            label_m = re.search(r"TEL;TYPE=([^:]+):", tel.group(0))
            if label_m: label = label_m.group(1)
            d["phones"].append({"value": tel.group(1).strip(), "label": label})
        for adr in re.finditer(r"^ADR.*:(.+)$", b, flags=re.MULTILINE):
            label = ""
            label_m = re.search(r"ADR;TYPE=([^:]+):", adr.group(0))
            if label_m: label = label_m.group(1)
            comps = adr.group(1).split(";")
            addr = {
                "po_box": comps[0].strip() if len(comps) > 0 else "",
                "extended": comps[1].strip() if len(comps) > 1 else "",
                "street": comps[2].strip() if len(comps) > 2 else "",
                "city": comps[3].strip() if len(comps) > 3 else "",
                "state": comps[4].strip() if len(comps) > 4 else "",
                "postal_code": comps[5].strip() if len(comps) > 5 else "",
                "country": comps[6].strip() if len(comps) > 6 else "",
                "label": label
            }
            d["addresses"].append(addr)
        m = re.search(r"^ORG:(.+)$", b, flags=re.MULTILINE)
        if m: d["company"] = parse_org(m.group(1).strip())
        m = re.search(r"^TITLE:(.+)$", b, flags=re.MULTILINE)
        if m: d["title"] = m.group(1).strip()
        for u in re.findall(r"^URL:(.+)$", b, flags=re.MULTILINE):
            if "linkedin.com" in u.lower(): d["linkedin_url"] = u.strip(); break
        rows.append(d)
    return rows


def parse_address(addr: Optional[str]) -> Tuple[str, str, str, str]:
    m = re.search(r"(.*?)[,\s]+([^,]+?)[,\s]+([A-Za-z]{2})[,\s]+(\d{4,10})(?:[-\s]\d+)?$", addr or "")
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip().upper(), m.group(4).strip()
    return (addr or "").strip(), "", "", ""


def normalize_address_components(a: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k in ["street","city","state","postal_code","po_box","extended","country"]:
        out[k] = (out.get(k, "") or "").strip()
    street = out.get("street","")
    if street and (not out.get("city") or not out.get("state") or not out.get("postal_code")):
        s,c,st,pc = parse_address(street)
        if s and (c or st or pc):
            out["street"] = s
            out["city"] = out.get("city") or c
            out["state"] = out.get("state") or st
            out["postal_code"] = out.get("postal_code") or pc
    out["state"] = normalize_state(out.get("state",""))
    out["country"] = normalize_country_iso2(out.get("country",""))
    return out


def extract_first_from_email(email: Optional[str]) -> str:
    if not email or "@" not in email: return ""
    local = email.split("@")[0]
    m = re.match(r"([a-zA-Z]+)[.\-_][a-zA-Z]+", local)
    if m: return m.group(1)
    m = re.match(r"([a-zA-Z]+)\d+$", local)
    if m: return m.group(1)
    return ""

def address_keys_for_match(addresses: List[Dict[str, Any]]) -> set:
    """Return a set of (city, state, postal) keys that have at least 2 non-empty values."""
    keys = set()
    for x in addresses or []:
        city = (x.get("city","") or "").strip().lower()
        state = (x.get("state","") or "").strip().upper()
        postal = (x.get("postal_code","") or "").strip()
        non_empty = sum(bool(v4) for v4 in (city, state, postal))
        if non_empty >= 2:
            keys.add((city, state, postal))
    return keys

def edit_distance_leq1(a: Optional[str], b: Optional[str]) -> bool:
    a = (a or "").lower(); b = (b or "").lower()
    if a == b: return True
    if abs(len(a)-len(b)) > 1: return False
    if len(a) == len(b):
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        return mismatches <= 1
    if len(a) > len(b): a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]: i += 1; j += 1
        else:
            diffs += 1; j += 1
            if diffs > 1: return False
    return True

def choose_best_first_name(records: List[Tuple[int, Dict[str, Any]]]) -> Tuple[str, str]:
    # use defaultdict for float counts to avoid Counter typing issues
    counts: Dict[str, float] = defaultdict(float)
    casing: Dict[str, str] = {}
    for _, r in records:
        fn = r.get("first_name", "")
        if fn:
            w = 2 if r.get("source") == "linkedin" else 1
            counts[fn.lower()] += float(w)
            casing.setdefault(fn.lower(), fn)
        for e in r.get("emails", []):
            cand = extract_first_from_email(e.get("value", ""))
            if cand:
                counts[cand.lower()] += 1.5
                casing.setdefault(cand.lower(), cand.title())
    if not counts:
        return "", ""
    keys = list(counts.keys())
    merged: Dict[str, float] = {}
    for k in keys:
        if k in merged:
            continue
        merged[k] = counts[k]
        for j in keys:
            if j == k or j in merged:
                continue
            if edit_distance_leq1(k, j):
                merged[k] += counts[j]
                merged[j] = -1.0
    best_key = max((k for k, v5 in merged.items() if v5 >= 0), key=lambda k1: merged[k1])
    return casing.get(best_key, best_key.title()), best_key

def build(args: Namespace) -> Tuple[DataFrame, DataFrame]:
    gen = set([g.lower() for g in (args.keep_generational_suffixes or list(DEFAULT_GEN))])
    prof = set([p.lower() for p in (args.professional_suffixes or list(DEFAULT_PROF))])
    default_phone_country = args.default_phone_country or "US"

    # Load sources
    all_rows: List[Dict[str, Any]] = []
    all_rows += load_linkedin_csv(args.linkedin_csv)
    all_rows += load_gmail_csv(args.gmail_csv)
    all_rows += load_vcards(args.mac_vcf)

    # Normalize per row
    for r in all_rows:
        tmp_emails: List[Dict[str, str]] = []
        raw_name = r.get("full_name_raw", "")
        raw_name = strip_emails_from_text_and_capture(raw_name, tmp_emails)

        f, m, l, gen_s, profs, maiden, clean_full = strip_suffixes_and_parse_name(raw_name, gen, prof)

        # append any captured emails from name into the record's emails list
        if tmp_emails:
            r.setdefault("emails", [])
            # prefer not to duplicate
            existing = {e.get("value", "") for e in r["emails"]}
            for e in tmp_emails:
                if e["value"] and e["value"] not in existing:
                    r["emails"].append(e)

        # If no first/last after parsing, attempt to backfill from any email local part
        if not (f or l):
            # choose the first available email
            any_email = None
            for e in r.get("emails", []):
                if EMAIL_RE.match(e.get("value","")):
                    any_email = e["value"]; break
            if any_email:
                local = any_email.split("@",1)[0]
                gf, gl = guess_name_from_email_local(local)
                f = f or gf
                l = l or gl
                clean_full = " ".join([t for t in [f, m, l, gen_s] if t]).strip()

        r["first_name"], r["middle_name"], r["last_name"] = f, m, l
        r["maiden_name"] = maiden
        r["suffix"] = r.get("suffix","") or gen_s
        r["suffix_professional"] = "|".join(profs) if profs else ""
        r["full_name"] = clean_full

        # Emails
        emails = [{"value": clean_email_val(e.get("value","")), "label": e.get("label","")} for e in r.get("emails",[])]
        r["emails"] = aggregate_unique([e for e in emails if e["value"]])

        # --- NEW: sanitize first/middle/last if they contain emails ---
        r.setdefault("emails", [])

        # (1) Pull out any emails embedded in name fields
        for fld in ("first_name", "middle_name", "last_name"):
            old = r.get(fld, "")
            new = strip_emails_from_text_and_capture(old, r["emails"])
            if new != old:
                r[fld] = new

        # (2) If first/last are exactly email-like, promote to emails and clear the field(s)
        for fld in ("first_name", "last_name"):
            val = (r.get(fld, "") or "").strip()
            if val and is_email_like(val):
                if val not in {e.get("value", "") for e in r["emails"]}:
                    r["emails"].append({"value": val, "label": ""})
                r[fld] = ""

        # (3) If we still lack a real first/last, try to backfill from any email local-part
        if not (r.get("first_name") or r.get("last_name")):
            # choose the first proper email we have
            prim = next((e.get("value","") for e in r["emails"] if EMAIL_RE.match(e.get("value",""))), "")
            if prim:
                local = prim.split("@", 1)[0]
                f_guess, l_guess = guess_name_from_email_local(local)
                # if we already have a last_name from earlier parsing, honor it; else use l_guess
                if not r.get("last_name") and l_guess:
                    r["last_name"] = l_guess
                if not r.get("first_name") and f_guess:
                    r["first_name"] = f_guess

        # (4) Special reconcile: if we have last_name, and email local looks like first-initial+last, set first_name initial
        if r.get("last_name"):
            prim = next((e.get("value","") for e in r["emails"] if EMAIL_RE.match(e.get("value",""))), "")
            if prim and not r.get("first_name"):
                local = prim.split("@", 1)[0]
                first_initial = reconcile_name_from_email_and_last(local, r["last_name"])
                if first_initial:
                    r["first_name"] = first_initial

        # (5) Rebuild full_name after any fixes
        parts = [r.get("first_name",""), r.get("middle_name",""), r.get("last_name",""), r.get("suffix","")]
        r["full_name"] = " ".join([p for p in parts if p]).strip()

        # Phones
        phones = [{"value": normalize_phone(p.get("value",""), default_phone_country), "label": p.get("label","")} for p in r.get("phones",[])]
        r["phones"] = aggregate_unique([p for p in phones if p["value"]])

        # Addresses
        addrs = [normalize_address_components(a) for a in r.get("addresses", [])]
        seen = set(); deduped_addrs: List[Dict[str, Any]] = []
        for a in addrs:
            addr_key = (a.get("po_box", ""), a.get("extended", ""), a.get("street", ""), a.get("city", ""), a.get("state", ""), a.get("postal_code", ""), a.get("country", ""))
            if addr_key not in seen:
                seen.add(addr_key); deduped_addrs.append(a)
        r["addresses"] = deduped_addrs

    # Duplicate detection: bucket by last name
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, r in enumerate(all_rows):
        buckets[re.sub(r"\s+", "", (r.get('last_name','') or '').lower())].append((idx, r))

    # union-find
    parent: Dict[int, int] = {}
    use_nick = getattr(args, "enable_nickname_equivalence", True)

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b1: int) -> None:
        ra, rb = find(a), find(b1)
        if ra != rb: parent[rb] = ra

    def dup_signals(ra: Tuple[int, Dict[str, Any]], rb: Tuple[int, Dict[str, Any]]):
        a = ra[1]; b2 = rb[1]
        score = 0.0
        corroborators = 0

        # Name similarity (first-name similarity has limited weight)
        first_sim1 = seq_ratio(a.get("first_name", ""), b2.get("first_name", ""))
        if use_nick and nickname_equivalent(a.get("first_name",""), b2.get("first_name", "")):
            first_sim1 = max(first_sim1, 0.96)   # effectively “very strong” first-name match
        score += 0.7 * first_sim1

        # Generational suffix exact match bonus
        if a.get("suffix", "") and (a.get("suffix", "") or "").lower() == (b2.get("suffix", "") or "").lower():
            score += 0.1

        # Email corroborator (and expose exact overlap as a flag)
        a_em = {e["value"] for e in a.get("emails",[])}
        b_em = {e["value"] for e in b2.get("emails", [])}
        emails_overlap1 = bool(a_em & b_em)
        if emails_overlap1:
            score += 1.0
            corroborators += 1

        # Phone corroborator
        a_ph = {p["value"] for p in a.get("phones",[])}
        b_ph = {p["value"] for p in b2.get("phones", [])}
        if a_ph & b_ph:
            score += 1.0
            corroborators += 1

        # Address corroborator (>=2 components must match)
        a_addr = address_keys_for_match(a.get("addresses",[]))
        b_addr = address_keys_for_match(b2.get("addresses", []))
        if a_addr & b_addr:
            score += 0.5
            corroborators += 1

        # LinkedIn corroborator
        if a.get("linkedin_url") and a.get("linkedin_url")==b2.get("linkedin_url"):
            score += 0.8
            corroborators += 1

        return score, corroborators, first_sim1, emails_overlap1

    for key, items in buckets.items():
        n = len(items)
        for i in range(n):
            for j in range(i+1, n):
                s, corrs, first_sim, emails_overlap = dup_signals(items[i], items[j])
                high = args.merge_score_threshold
                relaxed = args.relaxed_merge_threshold

                either_nameless = not (items[i][1].get("last_name") and items[i][1].get("first_name")) \
                                  or not (items[j][1].get("last_name") and items[j][1].get("first_name"))

                # Normal decision (as before)
                ok = (s >= high) or (first_sim >= args.first_name_similarity_threshold and s >= relaxed)

                # Require corroborator for nameless/email-like-name cases (from Phase 2.3)
                if either_nameless and corrs == 0:
                    ok = False

                if is_linkedin_rec(items[i][1]) or is_linkedin_rec(items[j][1]):
                    if not emails_overlap:
                    # Require the same last + (exact first or nickname-equivalent) + the same generational suffix
                        a = items[i][1]; b = items[j][1]
                        last_eq = _norm(a.get("last_name","")) == _norm(b.get("last_name",""))
                        gen_eq  = _norm(a.get("suffix","")) == _norm(b.get("suffix",""))
                        first_eq = _norm(a.get("first_name","")) == _norm(b.get("first_name",""))
                        first_nick = ""
                        if use_nick: first_nick = nickname_equivalent(a.get("first_name",""), b.get("first_name",""))
                        if not (last_eq and (first_eq or first_nick) and gen_eq):
                            ok = False

                # Respect global require_corroborator if set in config
                if args.require_corroborator:
                    ok = ok and (corrs > 0)

                if ok:
                    union(items[i][0], items[j][0])

    # Build clusters
    clusters: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, r in enumerate(all_rows):
        clusters[find(idx)].append((idx, r))

    # Merge clusters
    merged_contacts: List[Dict[str, Any]] = []
    lineage_records: List[Dict[str, Any]] = []
    duplicate_merge_count = 0
    for root1, records in clusters.items():
        # choose canonical first name
        best_first, _ = choose_best_first_name(records)

        middle = last = maiden = suffix = prof_suffixes = full_name = ""
        for _, r in records:
            middle = middle or r.get("middle_name", "")
            last = last or r.get("last_name", "")
            maiden = maiden or r.get("maiden_name", "")
            suffix = suffix or r.get("suffix", "")
            prof_suffixes = prof_suffixes or r.get("suffix_professional", "")
            full_name = full_name or r.get("full_name", "")

        full_name_clean = " ".join([t for t in [best_first, middle, last, suffix] if t]).strip()

        emails = []
        phones = []
        addresses = []
        for _, r in records:
            emails.extend(r.get("emails", []))
            phones.extend(r.get("phones", []))
            addresses.extend(r.get("addresses", []))

        def uniq_values(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen1 = set(); out: List[Dict[str, Any]] = []
            for d in lst:
                v6 = d.get("value", "")
                if v6 and v6 not in seen1:
                    seen1.add(v6); out.append(d)
            return out

        emails = uniq_values(emails)
        phones = uniq_values(phones)

        seen_addr = set(); uniq_addrs: List[Dict[str, Any]] = []
        for a in addresses:
            addr_key = (a.get("po_box", ""), a.get("extended", ""), a.get("street", ""), a.get("city", ""), a.get("state", ""), a.get("postal_code", ""), a.get("country", ""))
            if addr_key not in seen_addr:
                seen_addr.add(addr_key); uniq_addrs.append(a)
        addresses = uniq_addrs

        # company/title preference: first LinkedIn with data
        li = [r for _, r in records if r.get("source")=="linkedin" and (r.get("company") or r.get("title"))]
        if li:
            company = str(li[0].get("company",""))
            title = str(li[0].get("title",""))
        else:
            companies=[r.get("company","") for _, r in records if r.get("company","")]
            titles=[r.get("title","") for _, r in records if r.get("title","")]
            company = Counter(companies).most_common(1)[0][0] if companies else ""
            title = Counter(titles).most_common(1)[0][0] if titles else ""

        linkedin_url = ""
        for _, r in records:
            if r.get("linkedin_url"):
                linkedin_url = str(r.get("linkedin_url",""))
                break

        key_material = "|".join([
            (best_first or "") + " " + (middle or "") + " " + (last or "") + " " + (suffix or ""),
            ";".join(sorted([e["value"] for e in emails])),
            ";".join(sorted([p["value"] for p in phones]))
        ]).strip()
        contact_id = deterministic_uuid(key_material if key_material else str(uuid.uuid4()))

        merged_contacts.append({
            "contact_id": contact_id,
            "full_name": full_name_clean,
            "first_name": best_first,
            "middle_name": middle,
            "last_name": last,
            "maiden_name": maiden,
            "suffix": suffix,
            "suffix_professional": prof_suffixes,
            "company": company,
            "title": title,
            "linkedin_url": linkedin_url,
            "emails": "|".join([f"{e['value']}::{e.get('label','')}" for e in emails]) if emails else "",
            "phones": "|".join([f"{p['value']}::{p.get('label','')}" for p in phones]) if phones else "",
            "addresses_json": json.dumps(addresses, ensure_ascii=False),
            "source_count": len(records)
        })

        for idx, r in records:
            lineage_records.append({
                "contact_id": contact_id,
                "source": r.get("source",""),
                "source_row_id": r.get("source_row_id",""),
                "source_full_name": r.get("full_name_raw",""),
                "source_company": r.get("company",""),
                "source_title": r.get("title",""),
                "source_emails": "|".join([e["value"] for e in r.get("emails", [])]),
                "source_phones": "|".join([p["value"] for p in r.get("phones", [])]),
                "source_addresses_json": json.dumps(r.get("addresses", []), ensure_ascii=False)
            })
        if len(records) > 1: duplicate_merge_count += (len(records)-1)

    contacts_df = pd.DataFrame(merged_contacts)
    lineage_df = pd.DataFrame(lineage_records)
    return contacts_df, lineage_df

def main() -> int:
    # Minimal: enable INFO for the whole process
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Consolidate contacts from multiple sources.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--linkedin-csv", type=str, default=None)
    parser.add_argument("--gmail-csv", type=str, default=None)
    parser.add_argument("--mac-vcf", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--default-phone-country", type=str, default=None)
    parser.add_argument("--first-name-similarity-threshold", type=float, default=0.88)
    parser.add_argument("--merge-score-threshold", type=float, default=1.2)
    parser.add_argument("--relaxed-merge-threshold", type=float, default=0.6)
    parser.add_argument("--require-corroborator", action="store_true")
    parser.add_argument("--keep-generational-suffixes", nargs="*", default=None)
    parser.add_argument("--professional-suffixes", nargs="*", default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Resolve parameters from CLI > config > defaults
    inputs: Dict[str, Any] = cfg.get("inputs", {})
    outputs: Dict[str, Any] = cfg.get("outputs", {})
    normalization: Dict[str, Any] = cfg.get("normalization", {})
    dedupe: Dict[str, Any] = cfg.get("dedupe", {})

    args.linkedin_csv = args.linkedin_csv or inputs.get("linkedin_csv")
    args.gmail_csv = args.gmail_csv or inputs.get("gmail_csv")
    args.mac_vcf = args.mac_vcf or inputs.get("mac_vcf")

    out_dir = Path(args.out_dir or outputs.get("dir") or os.getcwd())

    args.default_phone_country = args.default_phone_country or normalization.get("default_phone_country", "US")
    args.keep_generational_suffixes = args.keep_generational_suffixes or normalization.get("keep_generational_suffixes")
    args.professional_suffixes = args.professional_suffixes or normalization.get("professional_suffixes")
    args.first_name_similarity_threshold = args.first_name_similarity_threshold or dedupe.get("first_name_similarity_threshold", 0.88)
    args.merge_score_threshold = args.merge_score_threshold or dedupe.get("merge_score_threshold", 1.2)
    args.relaxed_merge_threshold = args.relaxed_merge_threshold or dedupe.get("relaxed_merge_threshold", 0.6)
    args.require_corroborator = args.require_corroborator or dedupe.get("require_corroborator", False)

    contacts_df, lineage_df = build(args)

    # Save with full quoting
    contacts_path = out_dir / "consolidated_contacts.csv"
    lineage_path = out_dir/ "consolidated_lineage.csv"
    contacts_df.to_csv(str(contacts_path), index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    lineage_df.to_csv(str(lineage_path), index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    logger.info("Saved: %s", contacts_path)
    logger.info("Saved: %s", lineage_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
