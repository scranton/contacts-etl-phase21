from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .models import Address, ContactRecord, Email, Phone

logger = logging.getLogger(__name__)

try:
    from email_validator import EmailNotValidError, validate_email  # type: ignore

    HAS_EMAIL_VALIDATOR = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    validate_email = None  # type: ignore
    EmailNotValidError = Exception  # type: ignore
    HAS_EMAIL_VALIDATOR = False

try:
    import phonenumbers  # type: ignore

    NumberParseException = getattr(phonenumbers, "NumberParseException", Exception)
    HAS_PHONENUMBERS = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    phonenumbers = None  # type: ignore
    NumberParseException = Exception
    HAS_PHONENUMBERS = False

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

ISO2 = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "america": "US",
    "canada": "CA",
    "ca": "CA",
    "mexico": "MX",
    "mx": "MX",
    "united kingdom": "GB",
    "uk": "GB",
    "u.k.": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "ireland": "IE",
    "republic of ireland": "IE",
    "germany": "DE",
    "deutschland": "DE",
    "de": "DE",
    "france": "FR",
    "fr": "FR",
    "italy": "IT",
    "it": "IT",
    "spain": "ES",
    "es": "ES",
    "portugal": "PT",
    "pt": "PT",
    "netherlands": "NL",
    "holland": "NL",
    "nl": "NL",
    "belgium": "BE",
    "be": "BE",
    "switzerland": "CH",
    "ch": "CH",
    "austria": "AT",
    "at": "AT",
    "australia": "AU",
    "au": "AU",
    "new zealand": "NZ",
    "nz": "NZ",
    "india": "IN",
    "in": "IN",
    "china": "CN",
    "cn": "CN",
    "people's republic of china": "CN",
    "prc": "CN",
    "japan": "JP",
    "jp": "JP",
    "south korea": "KR",
    "republic of korea": "KR",
    "kr": "KR",
    "brazil": "BR",
    "br": "BR",
    "argentina": "AR",
    "ar": "AR",
    "south africa": "ZA",
    "za": "ZA",
    "sweden": "SE",
    "se": "SE",
    "norway": "NO",
    "no": "NO",
    "denmark": "DK",
    "dk": "DK",
    "finland": "FI",
    "fi": "FI",
    "czech republic": "CZ",
    "czechia": "CZ",
    "cz": "CZ",
    "poland": "PL",
    "pl": "PL",
    "singapore": "SG",
    "sg": "SG",
    "hong kong": "HK",
    "hk": "HK",
    "israel": "IL",
    "il": "IL",
    "united arab emirates": "AE",
    "uae": "AE",
    "ae": "AE",
}

STATE_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
    "dc": "DC",
}

PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "der",
    "di",
    "la",
    "le",
    "van",
    "von",
    "den",
    "ten",
    "ter",
    "du",
    "st",
    "st.",
    "san",
    "mac",
    "mc",
    "o",
    "d",
    "l",
}


@dataclass
class NormalizationSettings:
    keep_generational_suffixes: Set[str] = field(default_factory=set)
    professional_suffixes: Set[str] = field(default_factory=set)
    default_phone_country: str = "US"

    @classmethod
    def from_args(
        cls,
        keep_generational_suffixes: Optional[Iterable[str]],
        professional_suffixes: Optional[Iterable[str]],
        default_phone_country: str = "US",
    ) -> "NormalizationSettings":
        return cls(
            keep_generational_suffixes=set(s.lower() for s in (keep_generational_suffixes or [])),
            professional_suffixes=set(s.lower() for s in (professional_suffixes or [])),
            default_phone_country=default_phone_country or "US",
        )


def _norm(text: Optional[str]) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).lower()


def normalize_state(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if len(v) == 2 and v.isalpha():
        return v.upper()
    return STATE_ABBR.get(v.lower(), v.upper())


def normalize_country_iso2(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    return ISO2.get(v.lower(), v.upper() if len(v) == 2 else v)


def validate_email_safe(raw: str, check_deliverability: bool = False) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return ""
    if HAS_EMAIL_VALIDATOR:
        try:
            return validate_email(candidate, check_deliverability=check_deliverability).normalized  # type: ignore[arg-type]
        except EmailNotValidError:
            return ""
    candidate = candidate.replace(" ", "").lower()
    return candidate if EMAIL_RE.match(candidate) else ""


def is_valid_phone_safe(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return False
    if HAS_PHONENUMBERS:
        try:
            parsed = phonenumbers.parse(s, None if s.startswith("+") else "US")  # type: ignore[arg-type]
            return phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed)
        except NumberParseException:
            return False
    digits = re.sub(r"\D", "", s)
    return s.startswith("+") and len(digits) >= 11


def format_phone_e164_safe(value: str, default_country: str = "US") -> str:
    s = (value or "").strip()
    if not s:
        return ""
    formatted = ""
    if HAS_PHONENUMBERS:
        try:
            parsed = phonenumbers.parse(s, None if s.startswith("+") else default_country)  # type: ignore[arg-type]
            formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException:
            logger.debug("phonenumbers.parse failed for %s", s)
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


def read_csv_with_optional_header(
    path: Optional[str], header_starts_with: Optional[str] = None
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not header_starts_with:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.read().splitlines()
    header_idx: Optional[int] = None
    for index, line in enumerate(lines[:100]):
        if line.strip().startswith(header_starts_with):
            header_idx = index
            break
    if header_idx is None:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_csv(StringIO("\n".join(lines[header_idx:])), dtype=str, keep_default_na=False)


def safe_get(row: Any, key: str) -> str:
    try:
        return str(row.get(key, "") or "").strip()
    except (AttributeError, KeyError, TypeError):
        try:
            return str(row[key] if key in row else "").strip()
        except (KeyError, TypeError, AttributeError):
            return ""


def warn_missing(path: Optional[str], label: str) -> bool:
    if not path or not os.path.exists(path):
        logger.warning("%s path missing: %s", label, path)
        return True
    return False


def uniq_list_of_dicts(
    values: Sequence[Dict[str, Any]], key: str = "value"
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    results: List[Dict[str, Any]] = []
    for entry in values:
        val = str(entry.get(key, "") or "")
        if val and val not in seen:
            seen.add(val)
            results.append(dict(entry))
    return results


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def normalize_prof_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (token or "").lower())


def parse_name_multi_last(name_str: str) -> Tuple[str, str, str]:
    if not name_str:
        return "", "", ""
    tokens = name_str.split()
    if len(tokens) == 1:
        return tokens[0], "", ""
    last_parts = [tokens[-1]]
    idx = len(tokens) - 2
    while idx >= 1:
        token = tokens[idx]
        token_clean = (token or "").lower().strip(".")
        if token_clean in PARTICLES or (
            token_clean in {"o", "d", "l"} and idx + 1 < len(tokens) and "'" in tokens[idx + 1]
        ):
            last_parts.insert(0, token)
            idx -= 1
            continue
        if token and token[0].islower() and tokens[idx + 1][0].isupper():
            last_parts.insert(0, token)
            idx -= 1
            continue
        break
    first = tokens[0]
    middle = " ".join(tokens[1 : idx + 1]) if idx >= 1 else ""
    last = " ".join(last_parts)
    return first, middle, last


def strip_suffixes_and_parse_name(
    full_name: str, gen_suffixes: Set[str], prof_suffixes: Set[str]
) -> Tuple[str, str, str, str, List[str], str, str]:
    if not full_name or str(full_name).strip() == "":
        return "", "", "", "", [], "", ""
    name = str(full_name).strip()
    maiden = ""
    paren = re.search(r"\(([^)]+)\)", name)
    paren_text = ""
    if paren:
        paren_text = paren.group(1).strip()
        name = (name[: paren.start()] + name[paren.end() :]).strip()
    parts = [p.strip() for p in re.split(r"[,\u2013\u2014-]+", name) if p.strip()]
    kept_parts: List[str] = []
    gen_suffix = ""
    professional: List[str] = []

    def is_prof_token(token: str) -> bool:
        normalized = normalize_prof_token(token)
        return (normalized in prof_suffixes) or normalized.endswith("spc6")

    for part in parts:
        tokens = part.split()
        if len(tokens) == 1:
            token = tokens[0]
            if normalize_prof_token(token) in gen_suffixes:
                gen_suffix = part
                continue
            if is_prof_token(token):
                professional.append(part)
                continue
            kept_parts.append(part)
            continue
        trailing: List[str] = []
        while tokens and is_prof_token(tokens[-1]):
            trailing.append(tokens.pop())
        if trailing:
            professional.extend(reversed(trailing))
        while tokens and normalize_prof_token(tokens[-1]) in gen_suffixes:
            gen_suffix = tokens.pop()
        if tokens:
            kept_parts.append(" ".join(tokens))

    if paren_text:
        maiden_tokens: List[str] = []
        for token in [t.strip() for t in re.split(r"[,/&;]+", paren_text) if t.strip()]:
            if is_prof_token(token):
                professional.append(token)
            else:
                maiden_tokens.append(token)
        if maiden_tokens:
            maiden = " ".join(maiden_tokens)

    base = " ".join(kept_parts).strip()
    first, middle, last = parse_name_multi_last(base)
    full_name_clean = " ".join([part for part in (first, middle, last, gen_suffix) if part]).strip()
    return first, middle, last, gen_suffix, professional, maiden, full_name_clean


def normalize_email_collection(
    values: Sequence[Email], check_deliverability: bool = False
) -> List[Email]:
    out: List[Email] = []
    seen: Set[str] = set()
    for entry in values:
        normalized_value = validate_email_safe(
            entry.value, check_deliverability=check_deliverability
        )
        if not normalized_value:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        out.append(Email(value=normalized_value, label=entry.label))
    return out


def normalize_phone_collection(values: Sequence[Phone], default_country: str) -> List[Phone]:
    out: List[Phone] = []
    seen: Set[str] = set()
    for entry in values:
        formatted = format_phone_e164_safe(entry.value, default_country=default_country)
        if not formatted:
            continue
        if not is_valid_phone_safe(formatted):
            continue
        if formatted in seen:
            continue
        seen.add(formatted)
        out.append(Phone(value=formatted, label=entry.label))
    return out


def normalize_address(address: Address) -> Address:
    street = address.street
    city = address.city
    state = address.state
    postal_code = address.postal_code

    if street and (not city or not state or not postal_code):
        city_guess, state_guess, postal_guess = "", "", ""
        match = re.search(
            r"(.*?)[,\s]+([^,]+?)[,\s]+([A-Za-z]{2})[,\s]+(\d{4,10})(?:[-\s]\d+)?$", street
        )
        if match:
            street = match.group(1).strip()
            city_guess = match.group(2).strip()
            state_guess = match.group(3).strip()
            postal_guess = match.group(4).strip()
        city = city or city_guess
        state = state or state_guess
        postal_code = postal_code or postal_guess

    return Address(
        po_box=address.po_box,
        extended=address.extended,
        street=street.strip(),
        city=city.strip(),
        state=normalize_state(state),
        postal_code=postal_code.strip(),
        country=normalize_country_iso2(address.country),
        label=address.label,
    )


def normalize_address_collection(values: Sequence[Address]) -> List[Address]:
    seen: Set[str] = set()
    normalized: List[Address] = []
    for entry in values:
        addr = normalize_address(entry)
        key = json.dumps(addr.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(addr)
    return normalized


def strip_emails_from_text_and_capture(text: str, accumulator: List[Email]) -> str:
    if not text:
        return ""
    found = re.findall(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    for email in found:
        accumulator.append(Email(value=email, label=""))
    cleaned = text
    for email in found:
        cleaned = cleaned.replace(email, "").strip()
    return cleaned


def guess_name_from_email_local(local: str) -> Tuple[str, str]:
    parts = [part for part in re.split(r"[._-]+", local) if part]
    first = parts[0].title() if parts else ""
    last = parts[1].title() if len(parts) > 1 else ""
    return first, last


def reconcile_name_from_email_and_last(local: str, last: str) -> str:
    local_lower = (local or "").lower()
    last_lower = (last or "").lower()
    if last_lower and local_lower.endswith(last_lower) and len(local_lower) > len(last_lower):
        prefix = local_lower[: -len(last_lower)]
        if 1 <= len(prefix) <= 2:
            return prefix[0].upper()
    return ""


def nickname_root(name: str) -> str:
    return _VARIANT_TO_ROOT.get(_norm(name), _norm(name))


def nickname_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return nickname_root(a) == nickname_root(b)


_NICKMAP = {
    "william": {"william", "will", "bill", "billy", "liam"},
    "robert": {"robert", "rob", "bob", "bobby", "robby"},
    "richard": {"richard", "rich", "rick", "ricky", "dick"},
    "edward": {"edward", "ed", "eddie", "ted", "teddy", "ned"},
    "margaret": {"margaret", "meg", "maggie", "peggy"},
    "elizabeth": {"elizabeth", "liz", "beth", "lizzy", "eliza", "liza", "betsy"},
    "katherine": {"katherine", "kathy", "kate", "katie", "cathy", "cait"},
    "alexander": {"alexander", "alex", "sasha"},
    "james": {"james", "jim", "jimmy", "jamie"},
    "john": {"john", "jack", "johnny"},
    "jonathan": {"jonathan", "jon", "john"},
    "joseph": {"joseph", "joe", "joey"},
    "matthew": {"matthew", "matt"},
    "michael": {"michael", "mike", "mikey"},
    "jeffrey": {"jeffrey", "jeff"},
    "andrew": {"andrew", "andy", "drew"},
    "steven": {"steven", "steve", "stevie"},
    "christopher": {"christopher", "chris", "kit"},
    "patrick": {"patrick", "pat", "paddy"},
    "nicholas": {"nicholas", "nick", "nicky"},
    "catherine": {"catherine", "cathy", "kate", "katie"},
    "francis": {"francis", "frank", "frankie"},
}

_VARIANT_TO_ROOT: Dict[str, str] = {}
for root, variants in _NICKMAP.items():
    for variant in variants:
        _VARIANT_TO_ROOT[_norm(variant)] = root


def choose_best_first_name(records: Sequence[ContactRecord]) -> Tuple[str, str]:
    counts: Dict[str, float] = {}
    casing: Dict[str, str] = {}
    for record in records:
        if record.first_name:
            weight = 2.0 if record.source.lower() == "linkedin" else 1.0
            key = record.first_name.lower()
            counts[key] = counts.get(key, 0.0) + weight
            casing.setdefault(key, record.first_name)
        for email in record.emails:
            local = email.value.split("@", 1)[0] if "@" in email.value else ""
            first_guess = guess_name_from_email_local(local)[0]
            if first_guess:
                key = first_guess.lower()
                counts[key] = counts.get(key, 0.0) + 1.5
                casing.setdefault(key, first_guess.title())
    if not counts:
        return "", ""
    merged: Dict[str, float] = {}
    for key in counts:
        if key in merged:
            continue
        merged[key] = counts[key]
        for other in counts:
            if other == key or other in merged:
                continue
            if seq_ratio(key, other) >= 0.9:
                merged[key] += counts[other]
                merged[other] = -1.0
    best_key = max((k for k, score in merged.items() if score >= 0), key=lambda k: merged[k])
    return casing.get(best_key, best_key.title()), best_key


def normalize_contact_record(
    record: ContactRecord, settings: NormalizationSettings
) -> ContactRecord:
    gen_suffixes = settings.keep_generational_suffixes or set()
    prof_suffixes = settings.professional_suffixes or set()

    tmp_emails: List[Email] = []
    raw_name = strip_emails_from_text_and_capture(
        record.full_name_raw or record.full_name, tmp_emails
    )
    first, middle, last, gen_suffix, prof, maiden, full_name_clean = strip_suffixes_and_parse_name(
        raw_name, gen_suffixes, prof_suffixes
    )

    if tmp_emails:
        existing_values = {email.value for email in record.emails}
        record.emails.extend(
            [email for email in tmp_emails if email.value and email.value not in existing_values]
        )

    if not (first or last):
        primary_email = next(
            (email.value for email in record.emails if EMAIL_RE.match(email.value)), ""
        )
        if primary_email:
            local = primary_email.split("@", 1)[0]
            f_guess, l_guess = guess_name_from_email_local(local)
            first = first or f_guess
            last = last or l_guess
            full_name_clean = " ".join(
                part for part in [first, middle, last, gen_suffix] if part
            ).strip()

    record.first_name = first or record.first_name
    record.middle_name = middle or record.middle_name
    record.last_name = last or record.last_name
    record.maiden_name = maiden or record.maiden_name
    record.suffix = record.suffix or gen_suffix
    record.suffix_professional = (
        record.suffix_professional or "|".join(prof) if prof else record.suffix_professional
    )
    record.full_name = full_name_clean or record.full_name

    cleaned_emails: List[Email] = []
    for field_name in ("first_name", "middle_name", "last_name"):
        val = getattr(record, field_name)
        new_val = strip_emails_from_text_and_capture(val, record.emails)
        if new_val != val:
            setattr(record, field_name, new_val)
    for field_name in ("first_name", "last_name"):
        val = getattr(record, field_name).strip()
        if val and EMAIL_RE.match(val):
            record.emails.append(Email(value=val, label=""))
            setattr(record, field_name, "")

    if not (record.first_name or record.last_name):
        primary_email = next(
            (email.value for email in record.emails if EMAIL_RE.match(email.value)), ""
        )
        if primary_email:
            local = primary_email.split("@", 1)[0]
            f_guess, l_guess = guess_name_from_email_local(local)
            if not record.last_name and l_guess:
                record.last_name = l_guess
            if not record.first_name and f_guess:
                record.first_name = f_guess
    if record.last_name:
        primary_email = next(
            (email.value for email in record.emails if EMAIL_RE.match(email.value)), ""
        )
        if primary_email and not record.first_name:
            local = primary_email.split("@", 1)[0]
            initial = reconcile_name_from_email_and_last(local, record.last_name)
            if initial:
                record.first_name = initial

    record.full_name = " ".join(
        part
        for part in [record.first_name, record.middle_name, record.last_name, record.suffix]
        if part
    ).strip()

    record.emails = normalize_email_collection(record.emails)
    record.phones = normalize_phone_collection(record.phones, settings.default_phone_country)
    record.addresses = normalize_address_collection(record.addresses)

    return record


def address_keys_for_match(addresses: Sequence[Dict[str, Any]]) -> set[tuple[str, str, str]]:
    keys = set()
    for raw in addresses or []:
        city = (raw.get("city", "") or "").strip().lower()
        state = (raw.get("state", "") or "").strip().upper()
        postal = (raw.get("postal_code", "") or "").strip()
        non_empty = sum(bool(value) for value in (city, state, postal))
        if non_empty >= 2:
            keys.add((city, state, postal))
    return keys


def normalize_text_key(value: str) -> str:
    return _norm(value)
