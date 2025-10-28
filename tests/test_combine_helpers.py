import csv
from types import SimpleNamespace

import pandas as pd
import pytest

from contacts_etl import combine_contacts as cc
from contacts_etl import tag_contacts as tag
from contacts_etl.common import (
    ContactRecord,
    read_csv_with_optional_header,
    safe_get,
    warn_missing,
)
from contacts_etl.models import Address, Phone
from contacts_etl.normalization import format_phone_e164_safe, is_valid_phone_safe


def test_safe_get_and_warn_missing(tmp_path):
    row = {"A": "  value  ", "B": None}
    assert safe_get(row, "A") == "value"
    assert safe_get(row, "B") == ""
    missing = tmp_path / "nope.csv"
    assert warn_missing(str(missing), "Test") is True


def test_read_csv_with_optional_header(tmp_path):
    content = "noise line 1\nnoise line 2\nFirst Name,Last Name,URL\nJohn,Doe,https://linkedin.com/in/jdoe\n"
    path = tmp_path / "lin.csv"
    path.write_text(content, encoding="utf-8")
    df = read_csv_with_optional_header(str(path), header_starts_with="First Name,Last Name,URL")
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["First Name"] == "John"


def test_format_phone_fallback():
    assert format_phone_e164_safe("(415) 555-2671") == "+14155552671"
    assert format_phone_e164_safe("1-415-555-2671") == "+14155552671"
    assert format_phone_e164_safe("+44 20 7946 0958").startswith("+44")
    assert is_valid_phone_safe("+14155552671") is True


def test_load_vcards_assigns_row_ids(tmp_path):
    content = "\n".join(
        [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "FN:John Doe",
            "N:Doe;John;;;",
            "END:VCARD",
            "BEGIN:VCARD",
            "VERSION:3.0",
            "FN:Jane Smith",
            "N:Smith;Jane;;;",
            "END:VCARD",
            "",
        ]
    )
    path = tmp_path / "test.vcf"
    path.write_text(content, encoding="utf-8")
    rows = cc._load_vcards(str(path))
    assert [record.source_row_id for record in rows] == ["0", "1"]


def test_tag_contacts_merges_vcf_notes(tmp_path):
    contacts_df = pd.DataFrame(
        [
            {
                "contact_id": "cid-1",
                "full_name": "John Doe",
                "first_name": "John",
                "middle_name": "",
                "last_name": "Doe",
                "maiden_name": "",
                "suffix": "",
                "suffix_professional": "",
                "company": "Acme",
                "title": "Engineer",
                "linkedin_url": "",
                "emails": "",
                "phones": "",
                "addresses_json": "[]",
                "source_count": "1",
            }
        ]
    )
    contacts_csv = tmp_path / "contacts.csv"
    contacts_df.to_csv(contacts_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    lineage_df = pd.DataFrame(
        [
            {
                "contact_id": "cid-1",
                "source": "mac_vcf",
                "source_row_id": "0",
                "source_full_name": "",
                "source_company": "",
                "source_title": "",
                "source_emails": "",
                "source_phones": "",
                "source_addresses_json": "[]",
            }
        ]
    )
    lineage_csv = tmp_path / "lineage.csv"
    lineage_df.to_csv(lineage_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    vcf_content = "\n".join(
        [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "FN:John Doe",
            "N:Doe;John;;;",
            "NOTE:Prefers email intros",
            "END:VCARD",
            "",
        ]
    )
    vcf_path = tmp_path / "mac.vcf"
    vcf_path.write_text(vcf_content, encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"outputs:",
                f'  dir: "{tmp_path}"',
            ]
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        config=str(config_path),
        contacts_csv=str(contacts_csv),
        lineage_csv=str(lineage_csv),
        gmail_csv=None,
        mac_vcf=str(vcf_path),
        out_dir=str(tmp_path),
    )

    tag.build(args)
    tagged = pd.read_csv(tmp_path / "tagged_contacts.csv", dtype=str, keep_default_na=False)
    assert tagged.loc[0, "notes_blob"] == "Prefers email intros"


def test_build_respects_nickname_equivalence(monkeypatch):
    base_record = ContactRecord(
        source="gmail", source_row_id="0", first_name="Bill", last_name="Doe"
    )
    other_record = ContactRecord(
        source="gmail", source_row_id="1", first_name="William", last_name="Doe"
    )

    monkeypatch.setattr(cc, "_load_sources", lambda config: [base_record, other_record])

    base_args = SimpleNamespace(
        config=None,
        linkedin_csv=None,
        gmail_csv=None,
        mac_vcf=None,
        out_dir=None,
        default_phone_country="US",
        first_name_similarity_threshold=0.88,
        merge_score_threshold=1.2,
        relaxed_merge_threshold=0.6,
        require_corroborator=False,
        keep_generational_suffixes=None,
        professional_suffixes=None,
        enable_nickname_equivalence=True,
    )

    contacts_enabled, _ = cc.build(base_args)
    assert len(contacts_enabled) == 1

    base_args.enable_nickname_equivalence = False
    contacts_disabled, _ = cc.build(base_args)
    assert len(contacts_disabled) == 2


def test_build_keeps_distinct_household_members(monkeypatch):
    alex = ContactRecord(source="gmail", source_row_id="0", first_name="Alex", last_name="Resident")
    alex.phones = [Phone(value="+15550000001", label="home")]
    alex.addresses = [
        Address(
            street="123 Elm St", city="Sampleville", state="MA", postal_code="02144", country="US"
        )
    ]

    riley = ContactRecord(
        source="gmail", source_row_id="1", first_name="Riley", last_name="Resident"
    )
    riley.phones = [Phone(value="+15550000002", label="home")]
    riley.addresses = [
        Address(
            street="123 Elm St", city="Sampleville", state="MA", postal_code="02144", country="US"
        )
    ]

    monkeypatch.setattr(cc, "_load_sources", lambda config: [alex, riley])

    base_args = SimpleNamespace(
        config=None,
        linkedin_csv=None,
        gmail_csv=None,
        mac_vcf=None,
        out_dir=None,
        default_phone_country="US",
        first_name_similarity_threshold=0.88,
        merge_score_threshold=1.2,
        relaxed_merge_threshold=0.6,
        require_corroborator=False,
        keep_generational_suffixes=None,
        professional_suffixes=None,
        enable_nickname_equivalence=True,
    )

    contacts_df, lineage_df = cc.build(base_args)
    assert len(contacts_df) == 2
    assert set(contacts_df["first_name"]) == {"Alex", "Riley"}
    assert set(contacts_df["source_count"].astype(int)) == {1}
    assert set(contacts_df["source_row_count"].astype(int)) == {1}


if __name__ == "__main__":
    pytest.main(["-q"])
