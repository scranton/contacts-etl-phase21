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
from contacts_etl.models import Address, Email, Phone
from contacts_etl.normalization import (
    NormalizationSettings,
    format_phone_e164_safe,
    is_valid_phone_safe,
    normalize_contact_record,
)


def test_safe_get_and_warn_missing(tmp_path):
    row = {"A": "  value  ", "B": None}
    assert safe_get(row, "A") == "value"
    assert safe_get(row, "B") == ""
    missing = tmp_path / "nope.csv"
    assert warn_missing(str(missing), "Test") is True


def test_read_csv_with_optional_header(tmp_path):
    content_lines = [
        "noise line 1",
        "noise line 2",
        "First Name,Last Name,URL",
        "John,Doe,https://linkedin.com/in/jdoe",
        "",
    ]
    content = "\n".join(content_lines)
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


def test_load_vcards_reads_nickname(tmp_path):
    content = "\n".join(
        [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "FN:John Doe",
            "N:Doe;John;;;",
            "NICKNAME:Johnny",
            "END:VCARD",
            "",
        ]
    )
    path = tmp_path / "nick.vcf"
    path.write_text(content, encoding="utf-8")
    rows = cc._load_vcards(str(path))
    assert rows[0].nickname == "Johnny"


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
            ["outputs:", f'  dir: "{tmp_path}"']
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


def test_load_gmail_reads_nickname(tmp_path):
    df = pd.DataFrame(
        [
            {
                "First Name": "John",
                "Nickname": "Johnny",
                "E-mail 1 - Value": "john@example.com",
            }
        ]
    )
    path = tmp_path / "gmail.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    rows = cc._load_gmail_csv(str(path))
    assert rows[0].nickname == "Johnny"


def test_build_exposes_nickname(monkeypatch):
    record = ContactRecord(
        source="gmail",
        source_row_id="0",
        first_name="John",
        last_name="Example",
        nickname="Johnny",
        emails=[Email(value="john@example.com", label="home")],
    )

    monkeypatch.setattr(cc, "_load_sources", lambda config: [record])

    args = SimpleNamespace(
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

    contacts_df, _ = cc.build(args)
    assert list(contacts_df["nickname"]) == ["Johnny"]


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


def test_build_matches_on_explicit_nickname(monkeypatch):
    primary = ContactRecord(
        source="gmail", source_row_id="0", first_name="William", last_name="Example"
    )
    secondary = ContactRecord(
        source="gmail", source_row_id="1", first_name="", last_name="Example", nickname="Billy"
    )

    monkeypatch.setattr(cc, "_load_sources", lambda config: [primary, secondary])

    args = SimpleNamespace(
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

    contacts_df, _ = cc.build(args)
    assert len(contacts_df) == 1


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


def test_merge_prefers_linkedin_metadata(monkeypatch):
    linkedin_record = ContactRecord(
        source="linkedin",
        source_row_id="1",
        first_name="Jordan",
        last_name="Example",
        company="Future Corp",
        title="Principal Engineer",
        linkedin_url="https://linkedin.com/in/jordan-example",
    )
    gmail_record = ContactRecord(
        source="gmail",
        source_row_id="2",
        first_name="Jordan",
        last_name="Example",
        company="Old Employer",
        title="Developer",
        linkedin_url="",
    )

    monkeypatch.setattr(cc, "_load_sources", lambda config: [linkedin_record, gmail_record])

    args = SimpleNamespace(
        config=None,
        linkedin_csv=None,
        gmail_csv=None,
        mac_vcf=None,
        out_dir=None,
        default_phone_country="US",
        first_name_similarity_threshold=0.0,
        merge_score_threshold=0.0,
        relaxed_merge_threshold=0.0,
        require_corroborator=False,
        keep_generational_suffixes=None,
        professional_suffixes=None,
        enable_nickname_equivalence=True,
    )

    contacts_df, _ = cc.build(args)
    assert len(contacts_df) == 1
    merged = contacts_df.iloc[0]
    assert merged["company"] == "Future Corp"
    assert merged["title"] == "Principal Engineer"
    assert merged["linkedin_url"] == "https://linkedin.com/in/jordan-example"


def test_normalize_email_dedup_preserves_best_label():
    record = ContactRecord(
        emails=[
            Email(value="primary@example.com", label=""),
            Email(value="primary@example.com", label="work"),
            Email(value="SECONDARY@example.com", label="HOME"),
            Email(value="secondary@example.com", label=""),
        ]
    )
    settings = NormalizationSettings(default_phone_country="US")
    normalized = normalize_contact_record(record, settings)
    assert normalized.emails == [
        Email(value="primary@example.com", label="work"),
        Email(value="SECONDARY@example.com", label="home"),
        Email(value="secondary@example.com", label=""),
    ]


def test_load_vcards_filters_pref_and_internet_labels(tmp_path):
    vcf = "\n".join(
        [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "FN:Casey Example",
            "N:Example;Casey;;;",
            "EMAIL;TYPE=INTERNET;TYPE=WORK;TYPE=pref:casey.work@example.com",
            "EMAIL;TYPE=INTERNET:casey.other@example.com",
            "TEL;TYPE=CELL;TYPE=pref:+1-555-000-0003",
            "TEL;TYPE=VOICE:+1-555-000-0004",
            "END:VCARD",
            "",
        ]
    )
    path = tmp_path / "labels.vcf"
    path.write_text(vcf, encoding="utf-8")

    records = cc._load_vcards(str(path))
    assert len(records) == 1
    entry = records[0]
    assert entry.emails == [
        Email(value="casey.work@example.com", label="work"),
        Email(value="casey.other@example.com", label=""),
    ]
    assert entry.phones == [
        Phone(value="+1-555-000-0003", label="cell"),
        Phone(value="+1-555-000-0004", label="voice"),
    ]


def test_address_dedup_keeps_label():
    record = ContactRecord(
        addresses=[
            Address(
                street="1 Maple Street",
                city="Sampletown",
                state="MA",
                postal_code="02144",
                country="US",
                label="home",
            ),
            Address(
                street="1 Maple Street",
                city="Sampletown",
                state="MA",
                postal_code="02144",
                country="US",
                label="",
            ),
        ]
    )
    settings = NormalizationSettings(default_phone_country="US")
    normalized = normalize_contact_record(record, settings)
    assert normalized.addresses == [
        Address(
            po_box="",
            extended="",
            street="1 Maple Street",
            city="Sampletown",
            state="MA",
            postal_code="02144",
            country="US",
            label="home",
        )
    ]


if __name__ == "__main__":
    pytest.main(["-q"])
