from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .normalization import safe_get

MARTIAL_KWS = [r"tai\s*chi", r"wu\s*an", r"wu\s*dao", r"kung\s*fu", r"shaolin", r"martial\s*arts"]
NUTCRACKER_KWS = [r"nutcracker", r"\bcherub(s)?\b", r"jose\s*mateo", r"ballet"]


def _any_kw(text: str, patterns: Iterable[str]) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def _extract_domain_set(emails_field: str) -> Set[str]:
    domains: Set[str] = set()
    if not emails_field:
        return domains
    parts = [part for part in emails_field.split("|") if part.strip()]
    for part in parts:
        email = part.split("::")[0].strip().lower()
        if "@" in email:
            domains.add(email.split("@")[1])
    return domains


@dataclass
class TaggingSettings:
    prior_companies: List[str]
    prior_domains: List[str]
    local_cities: List[str]

    def normalized_companies(self) -> List[str]:
        return [company.strip().lower() for company in self.prior_companies]

    def normalized_domains(self) -> List[str]:
        return [domain.strip().lower() for domain in self.prior_domains]

    def normalized_cities(self) -> List[str]:
        return [city.strip().lower() for city in self.local_cities]


class TagEngine:
    def __init__(self, settings: TaggingSettings):
        self.settings = settings

    def tag_record(self, row: Dict[str, str]) -> tuple[Set[str], str]:
        tags: Set[str] = set()
        text_blob = " ".join(
            [
                safe_get(row, "company"),
                safe_get(row, "title"),
                safe_get(row, "linkedin_url"),
                safe_get(row, "notes_blob"),
            ]
        ).lower()

        if _any_kw(text_blob, MARTIAL_KWS):
            tags.add("martial_arts")
        if _any_kw(text_blob, NUTCRACKER_KWS):
            tags.add("nutcracker_performance")

        company = safe_get(row, "company").lower()
        if company and any(prior in company for prior in self.settings.normalized_companies()):
            tags.add("work_colleague")

        domains = _extract_domain_set(safe_get(row, "emails"))
        if any(
            any(prior in domain for prior in self.settings.normalized_domains())
            for domain in domains
        ):
            tags.add("work_colleague")

        try:
            addresses = json.loads(row.get("addresses_json", "") or "[]")
        except Exception:
            addresses = []
        local_cities = self.settings.normalized_cities()
        for address in addresses:
            city = safe_get(address, "city").lower()
            state = safe_get(address, "state").lower()
            if state == "ma" and (
                city in local_cities or any(city == lc or lc in city for lc in local_cities)
            ):
                tags.add("local_south_shore")
                break

        if "martial_arts" in tags or "nutcracker_performance" in tags:
            primary = "personal"
        elif "work_colleague" in tags or safe_get(row, "linkedin_url"):
            primary = "professional"
        elif "local_south_shore" in tags:
            primary = "local_referral"
        else:
            primary = "uncategorized"
        return tags, primary

    @staticmethod
    def compute_referral_priority(
        row: Dict[str, str],
        confidence_weight: float = 0.6,
        tag_weights: Dict[str, int] | None = None,
    ) -> int:
        if tag_weights is None:
            tag_weights = {
                "martial_arts": 30,
                "nutcracker_performance": 25,
                "work_colleague": 20,
                "local_south_shore": 10,
            }
        try:
            confidence = float(row.get("confidence_score", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        tags = set((row.get("tags") or "").split("|")) if row.get("tags") else set()
        score = confidence * confidence_weight
        score += sum(tag_weights.get(tag, 0) for tag in tags)
        return int(min(100, round(score, 0)))


__all__ = ["TaggingSettings", "TagEngine"]
