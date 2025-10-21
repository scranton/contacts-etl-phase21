import re
import unicodedata
import uuid
from difflib import SequenceMatcher
import os
from io import StringIO
from typing import List, Dict, Any, Optional

import pandas as pd
import logging

# Optional external libs: import once and expose safe wrappers
try:
    from email_validator import validate_email, EmailNotValidError  # type: ignore
    HAS_EMAIL_VALIDATOR = True
except (ImportError, ModuleNotFoundError):
    validate_email = None  # type: ignore
    class _DummyEmailNotValidError(Exception): ...
    EmailNotValidError = _DummyEmailNotValidError  # type: ignore
    HAS_EMAIL_VALIDATOR = False

try:
    import phonenumbers  # type: ignore
    NumberParseException = getattr(phonenumbers, "NumberParseException", Exception)
    HAS_PHONENUMBERS = True
except (ImportError, ModuleNotFoundError):
    phonenumbers = None  # type: ignore
    NumberParseException = Exception
    HAS_PHONENUMBERS = False

# Country/state mappings (subset, extend as needed)
ISO2 = {
    "us":"US","usa":"US","united states":"US","united states of america":"US","u.s.":"US","u.s.a.":"US","america":"US",
    "canada":"CA","ca":"CA","mexico":"MX","mx":"MX","united kingdom":"GB","uk":"GB","u.k.":"GB","great britain":"GB",
    "england":"GB","scotland":"GB","wales":"GB","northern ireland":"GB","ireland":"IE","republic of ireland":"IE",
    "germany":"DE","deutschland":"DE","de":"DE","france":"FR","fr":"FR","italy":"IT","it":"IT","spain":"ES","es":"ES",
    "portugal":"PT","pt":"PT","netherlands":"NL","holland":"NL","nl":"NL","belgium":"BE","be":"BE","switzerland":"CH","ch":"CH",
    "austria":"AT","at":"AT","australia":"AU","au":"AU","new zealand":"NZ","nz":"NZ","india":"IN","in":"IN",
    "china":"CN","cn":"CN","people's republic of china":"CN","prc":"CN","japan":"JP","jp":"JP","south korea":"KR","republic of korea":"KR","kr":"KR",
    "brazil":"BR","br":"BR","argentina":"AR","ar":"AR","south africa":"ZA","za":"ZA","sweden":"SE","se":"SE","norway":"NO","no":"NO",
    "denmark":"DK","dk":"DK","finland":"FI","fi":"FI","czech republic":"CZ","czechia":"CZ","cz":"CZ","poland":"PL","pl":"PL",
    "singapore":"SG","sg":"SG","hong kong":"HK","hk":"HK","israel":"IL","il":"IL","united arab emirates":"AE","uae":"AE","ae":"AE"
}
STATE_ABBR = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
    "connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
    "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
    "maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
    "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
    "west virginia":"WV","wisconsin":"WI","wyoming":"WY","district of columbia":"DC","dc":"DC"
}
PARTICLES = {"da","de","del","della","der","di","la","le","van","von","den","ten","ter","du","st","st.","san","mac","mc","o","d","l"}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

def validate_email_safe(raw: str, check_deliverability: bool = False) -> Optional[str]:
    """
    Return normalized email on success, or empty string on invalid.
    Uses email_validator when available, otherwise falls back to regex.
    """
    e = (raw or "").strip()
    if not e:
        return ""
    if HAS_EMAIL_VALIDATOR:
        try:
            return validate_email(e, check_deliverability=check_deliverability).normalized
        except EmailNotValidError:
            return ""
    e = e.replace(" ", "").lower()
    return e if EMAIL_RE.match(e) else ""

def is_valid_phone_safe(e164_or_raw: str) -> bool:
    """
    Quick validity check: prefer phonenumbers if available, otherwise simple heuristic.
    Accepts either already-formatted E164 or raw input.
    """
    s = (e164_or_raw or "").strip()
    if not s:
        return False
    if HAS_PHONENUMBERS:
        try:
            pn = phonenumbers.parse(s, None if s.startswith("+") else "US")
            return phonenumbers.is_possible_number(pn) and phonenumbers.is_valid_number(pn)
        except NumberParseException:
            return False
    # fallback: must start with '+' and contain at least 11 digits total
    digits = re.sub(r"\D", "", s)
    return s.startswith("+") and len(digits) >= 11

def format_phone_e164_safe(raw: str, default_country: str = "US") -> str:
    """
    Return an E.164-like formatted phone or empty string.
    Uses `phonenumbers` when available, otherwise falls back to simple digit heuristics.
    Does *not* assert validity — caller can use `is_valid_phone_safe`.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    formatted = ""
    if HAS_PHONENUMBERS:
        try:
            pn = phonenumbers.parse(s, None if s.startswith("+") else default_country)
            formatted = phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException:
            logging.getLogger(__name__).debug("phonenumbers.parse failed for %s", s)
            formatted = ""
    if not formatted:
        digits = re.sub(r"\D", "", s)
        if len(digits) == 10:
            formatted = f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            formatted = f"+{digits}"
        elif s.startswith("+"):
            formatted = re.sub(r"[^\d+]", "", s)
        else:
            formatted = f"+1{digits}" if digits else ""
    return formatted

def _sget(row: Any, key: str) -> str:
    """Safe string retrieval from a mapping/row and strip whitespace."""
    try:
        return str(row.get(key, "") or "").strip()
    except (AttributeError, KeyError, TypeError):
        try:
            return str(row[key] if key in row else "").strip()
        except (KeyError, TypeError, AttributeError):
            return ""

def _warn_missing(path: Optional[str], label: str) -> bool:
    """Log a warning if path is missing; returns True if missing."""
    if not path or not os.path.exists(path):
        logging.getLogger(__name__).warning("%s path missing: %s", label, path)
        return True
    return False

def read_csv_with_optional_header(path: Optional[str], header_starts_with: Optional[str] = None) -> pd.DataFrame:
    """Read CSV but allow noisy prefix by scanning for a header line that starts with header_starts_with."""
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
    """Deduplicate list-of-dicts preserving order by `key`."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in lst:
        v = d.get(key, "")
        if v and v not in seen:
            seen.add(v)
            out.append(d)
    return out

def _norm(s: Optional[str]) -> str:
    """Normalize text: strip, unicode-normalize, collapse whitespace and lowercase."""
    s = (s or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).lower()


def normalize_state(s:str)->str:
    s = (s or "").strip()
    if not s: return ""
    if len(s)==2 and s.isalpha(): return s.upper()
    return STATE_ABBR.get(s.lower(), s.upper())

def normalize_country_iso2(c:str)->str:
    c = (c or "").strip()
    if not c: return ""
    return ISO2.get(c.lower(), c.upper() if len(c)==2 else c)

def normalize_prof_token(tok:str)->str:
    return re.sub(r"[^a-z0-9]", "", (tok or "").lower())

def parse_name_multi_last(name_str:str):
    """Parse first/middle/last considering multi-element surnames and apostrophes."""
    if not name_str: return "","",""
    tokens = name_str.split()
    if len(tokens) == 1: return tokens[0], "", ""
    last_parts = [tokens[-1]]
    i = len(tokens) - 2
    while i >= 1:
        t = tokens[i]
        t_clean = (t or "").lower().strip(".")
        if t_clean in PARTICLES or (t_clean in {"o","d","l"} and i+1 < len(tokens) and "'" in tokens[i+1]):
            last_parts.insert(0, t); i -= 1; continue
        if t and t[0].islower() and tokens[i+1][0].isupper():
            last_parts.insert(0, t); i -= 1; continue
        break
    first = tokens[0]
    middle = " ".join(tokens[1: i+1]) if i >= 1 else ""
    last = " ".join(last_parts)
    return first, middle, last

def strip_suffixes_and_parse_name(full_name:str, gen_suffixes:set, prof_suffixes:set):
    """Return (first, middle, last, gen_suffix, prof_suffixes_list, maiden, full_name_clean)."""
    if not full_name or str(full_name).strip() == "":
        return "", "", "", "", [], "", ""
    name = str(full_name).strip()
    maiden = ""
    paren = re.search(r"\(([^)]+)\)", name)
    if paren:
        maiden = paren.group(1).strip()
        name = (name[:paren.start()] + name[paren.end():]).strip()
    parts = [p.strip() for p in re.split(r"[,\u2013\u2014-]+", name) if p.strip()]
    kept_parts = []
    gen_suffix = ""
    prof_out = []
    def is_prof_token(token:str)->bool:
        t = normalize_prof_token(token)
        return (t in prof_suffixes) or (t.endswith("spc6"))
    for p in parts:
        tokens = p.split()
        if len(tokens) == 1:
            tok = tokens[0]
            if normalize_prof_token(tok) in gen_suffixes:
                gen_suffix = p; continue
            if is_prof_token(tok):
                prof_out.append(p); continue
            kept_parts.append(p); continue
        trailing = []
        while tokens and is_prof_token(tokens[-1]):
            trailing.append(tokens.pop())
        if trailing: prof_out.extend(trailing[::-1])
        if len(tokens) == 1 and normalize_prof_token(tokens[0]) in gen_suffixes:
            gen_suffix = tokens[0]; tokens = []
        if tokens: kept_parts.append(" ".join(tokens))
    base = " ".join(kept_parts).strip()
    first, middle, last = parse_name_multi_last(base)
    full_name_clean = " ".join([t for t in [first, middle, last, gen_suffix] if t]).strip()
    return first, middle, last, gen_suffix, prof_out, maiden, full_name_clean

def deterministic_uuid(namespace_str:str)->str:
    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(ns, namespace_str))

def seq_ratio(a:str,b:str)->float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
