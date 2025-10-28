from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .models import ContactRecord
from .normalization import address_keys_for_match, nickname_equivalent, seq_ratio


@dataclass
class MergeSignals:
    score: float
    corroborators: int
    first_similarity: float
    emails_overlap: bool

    @property
    def has_corroborator(self) -> bool:
        return self.corroborators > 0


class MergeEvaluator:
    def __init__(self, nickname_equivalence: bool = True):
        self.nickname_equivalence = nickname_equivalence

    def compute(self, a: ContactRecord, b: ContactRecord) -> MergeSignals:
        score = 0.0
        corroborators = 0

        first_similarity = seq_ratio(a.first_name, b.first_name)
        if self.nickname_equivalence and nickname_equivalent(a.first_name, b.first_name):
            first_similarity = max(first_similarity, 0.96)
        score += 0.7 * first_similarity

        if a.suffix and a.suffix.lower() == b.suffix.lower():
            score += 0.1

        emails_a = {email.value for email in a.emails}
        emails_b = {email.value for email in b.emails}
        emails_overlap = bool(emails_a & emails_b)
        if emails_overlap:
            score += 1.0
            corroborators += 1

        phones_a = {phone.value for phone in a.phones}
        phones_b = {phone.value for phone in b.phones}
        if phones_a & phones_b:
            score += 1.0
            corroborators += 1

        addr_a = address_keys_for_match([address.to_dict() for address in a.addresses])
        addr_b = address_keys_for_match([address.to_dict() for address in b.addresses])
        if addr_a & addr_b:
            score += 0.5
            corroborators += 1

        if a.linkedin_url and a.linkedin_url == b.linkedin_url:
            score += 0.8
            corroborators += 1

        return MergeSignals(
            score=score,
            corroborators=corroborators,
            first_similarity=first_similarity,
            emails_overlap=emails_overlap,
        )

    @staticmethod
    def either_nameless(records: Iterable[ContactRecord]) -> bool:
        return any(not (record.first_name and record.last_name) for record in records)
