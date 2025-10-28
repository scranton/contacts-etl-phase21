from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Email:
    value: str
    label: str = ""

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Email":
        return Email(
            value=str(payload.get("value", "") or "").strip(),
            label=str(payload.get("label", "") or "").strip(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {"value": self.value, "label": self.label}


@dataclass(frozen=True)
class Phone:
    value: str
    label: str = ""

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Phone":
        return Phone(
            value=str(payload.get("value", "") or "").strip(),
            label=str(payload.get("label", "") or "").strip(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {"value": self.value, "label": self.label}


@dataclass(frozen=True)
class Address:
    po_box: str = ""
    extended: str = ""
    street: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    label: str = ""

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Address":
        return Address(
            po_box=str(payload.get("po_box", "") or "").strip(),
            extended=str(payload.get("extended", "") or "").strip(),
            street=str(payload.get("street", "") or "").strip(),
            city=str(payload.get("city", "") or "").strip(),
            state=str(payload.get("state", "") or "").strip(),
            postal_code=str(payload.get("postal_code", "") or "").strip(),
            country=str(payload.get("country", "") or "").strip(),
            label=str(payload.get("label", "") or "").strip(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "po_box": self.po_box,
            "extended": self.extended,
            "street": self.street,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
            "label": self.label,
        }


@dataclass
class ContactRecord:
    contact_id: str = ""
    full_name_raw: str = ""
    full_name: str = ""
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    maiden_name: str = ""
    suffix: str = ""
    suffix_professional: str = ""
    company: str = ""
    title: str = ""
    linkedin_url: str = ""
    source: str = ""
    source_row_id: str = ""
    emails: List[Email] = field(default_factory=list)
    phones: List[Phone] = field(default_factory=list)
    addresses: List[Address] = field(default_factory=list)
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _ensure_email_list(values: Sequence[Any]) -> List[Email]:
        return [
            value if isinstance(value, Email) else Email.from_mapping(value) for value in values
        ]

    @staticmethod
    def _ensure_phone_list(values: Sequence[Any]) -> List[Phone]:
        return [
            value if isinstance(value, Phone) else Phone.from_mapping(value) for value in values
        ]

    @staticmethod
    def _ensure_address_list(values: Sequence[Any]) -> List[Address]:
        return [
            value if isinstance(value, Address) else Address.from_mapping(value) for value in values
        ]

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "ContactRecord":
        return cls(
            contact_id=str(payload.get("contact_id", "") or "").strip(),
            full_name_raw=str(payload.get("full_name_raw", "") or "").strip(),
            full_name=str(payload.get("full_name", "") or "").strip(),
            first_name=str(payload.get("first_name", "") or "").strip(),
            middle_name=str(payload.get("middle_name", "") or "").strip(),
            last_name=str(payload.get("last_name", "") or "").strip(),
            maiden_name=str(payload.get("maiden_name", "") or "").strip(),
            suffix=str(payload.get("suffix", "") or "").strip(),
            suffix_professional=str(payload.get("suffix_professional", "") or "").strip(),
            company=str(payload.get("company", "") or "").strip(),
            title=str(payload.get("title", "") or "").strip(),
            linkedin_url=str(payload.get("linkedin_url", "") or "").strip(),
            source=str(payload.get("source", "") or "").strip(),
            source_row_id=str(payload.get("source_row_id", "") or "").strip(),
            emails=cls._ensure_email_list(payload.get("emails", []) or []),
            phones=cls._ensure_phone_list(payload.get("phones", []) or []),
            addresses=cls._ensure_address_list(payload.get("addresses", []) or []),
            notes=str(payload.get("notes", "") or "").strip(),
            extra=dict(payload.get("extra", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "full_name_raw": self.full_name_raw,
            "full_name": self.full_name,
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "maiden_name": self.maiden_name,
            "suffix": self.suffix,
            "suffix_professional": self.suffix_professional,
            "company": self.company,
            "title": self.title,
            "linkedin_url": self.linkedin_url,
            "source": self.source,
            "source_row_id": self.source_row_id,
            "emails": [email.to_dict() for email in self.emails],
            "phones": [phone.to_dict() for phone in self.phones],
            "addresses": [address.to_dict() for address in self.addresses],
            "notes": self.notes,
            "extra": dict(self.extra),
        }

    def replace(self, **changes: Any) -> "ContactRecord":
        return replace(self, **changes)


@dataclass(frozen=True)
class LineageEntry:
    contact_id: str
    source: str
    source_row_id: str
    source_full_name: str
    source_company: str
    source_title: str
    source_emails: str
    source_phones: str
    source_addresses_json: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "contact_id": self.contact_id,
            "source": self.source,
            "source_row_id": self.source_row_id,
            "source_full_name": self.source_full_name,
            "source_company": self.source_company,
            "source_title": self.source_title,
            "source_emails": self.source_emails,
            "source_phones": self.source_phones,
            "source_addresses_json": self.source_addresses_json,
        }
