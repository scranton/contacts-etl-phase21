import csv
from types import SimpleNamespace

import pandas as pd
import pytest

from contacts_etl import combine_contacts as cc
from contacts_etl import tag_contacts as tag


def test_sget_and_warn_missing(tmp_path):
    row = {"A": "  value  ", "B": None}
    assert cc._sget(row, "A") == "value"
    assert cc._sget(row, "B") == ""
    # warn_missing returns True when missing
    missing = tmp_path / "nope.csv"
    assert cc._warn_missing(str(missing), "Test") is True


def test_read_csv_with_optional_header(tmp_path):
    content = "noise line 1\nnoise line 2\nFirst Name,Last Name,URL\nJohn,Doe,https://linkedin.com/in/jdoe\n"
    p = tmp_path / "lin.csv"
    p.write_text(content, encoding="utf-8")
    df = cc.read_csv_with_optional_header(str(p), header_starts_with="First Name,Last Name,URL")
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["First Name"] == "John"


def test_uniq_list_of_dicts():
    lst = [{"value": "a"}, {"value": "b"}, {"value": "a"}, {"value": ""}]
    out = cc.uniq_list_of_dicts(lst)
    assert out == [{"value": "a"}, {"value": "b"}]


def test_normalize_phone_fallback():
    # Without phonenumbers installed, should normalize heuristically
    assert cc.normalize_phone("(415) 555-2671") == "+14155552671"
    assert cc.normalize_phone("1-415-555-2671") == "+14155552671"
    assert cc.normalize_phone("+44 20 7946 0958") == "+442079460958"


def test_load_vcards_assigns_row_ids(tmp_path):
    content = "\n".join([
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
    ])
    path = tmp_path / "test.vcf"
    path.write_text(content, encoding="utf-8")
    rows = cc.load_vcards(str(path))
    assert [r["source_row_id"] for r in rows] == ["0", "1"]


def test_tag_contacts_merges_vcf_notes(tmp_path):
    contacts_df = pd.DataFrame([{
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
    }])
    contacts_csv = tmp_path / "contacts.csv"
    contacts_df.to_csv(contacts_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    lineage_df = pd.DataFrame([{
        "contact_id": "cid-1",
        "source": "mac_vcf",
        "source_row_id": "0",
        "source_full_name": "",
        "source_company": "",
        "source_title": "",
        "source_emails": "",
        "source_phones": "",
        "source_addresses_json": "[]",
    }])
    lineage_csv = tmp_path / "lineage.csv"
    lineage_df.to_csv(lineage_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    vcf_content = "\n".join([
        "BEGIN:VCARD",
        "VERSION:3.0",
        "FN:John Doe",
        "N:Doe;John;;;",
        "NOTE:Prefers email intros",
        "END:VCARD",
        "",
    ])
    vcf_path = tmp_path / "mac.vcf"
    vcf_path.write_text(vcf_content, encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"outputs:\n  dir: \"{tmp_path}\"\n", encoding="utf-8")

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
    def stub_records():
        return [
            {
                "full_name_raw": "Bill Doe",
                "full_name": "",
                "first_name": "Bill",
                "middle_name": "",
                "last_name": "Doe",
                "maiden_name": "",
                "suffix": "",
                "suffix_professional": "",
                "emails": [],
                "phones": [],
                "addresses": [],
                "company": "",
                "title": "",
                "linkedin_url": "",
                "source": "gmail",
                "source_row_id": "0",
            },
            {
                "full_name_raw": "William Doe",
                "full_name": "",
                "first_name": "William",
                "middle_name": "",
                "last_name": "Doe",
                "maiden_name": "",
                "suffix": "",
                "suffix_professional": "",
                "emails": [],
                "phones": [],
                "addresses": [],
                "company": "",
                "title": "",
                "linkedin_url": "",
                "source": "gmail",
                "source_row_id": "1",
            },
        ]

    monkeypatch.setattr(cc, "load_linkedin_csv", lambda _: [])
    monkeypatch.setattr(cc, "load_vcards", lambda _: [])
    monkeypatch.setattr(cc, "load_gmail_csv", lambda _: stub_records())

    base_args = {
        "linkedin_csv": None,
        "gmail_csv": None,
        "mac_vcf": None,
        "out_dir": None,
        "default_phone_country": "US",
        "first_name_similarity_threshold": 0.88,
        "merge_score_threshold": 1.2,
        "relaxed_merge_threshold": 0.6,
        "require_corroborator": False,
        "keep_generational_suffixes": None,
        "professional_suffixes": None,
    }

    args_enabled = SimpleNamespace(enable_nickname_equivalence=True, **base_args)
    contacts_enabled, _ = cc.build(args_enabled)
    assert len(contacts_enabled) == 1

    args_disabled = SimpleNamespace(enable_nickname_equivalence=False, **base_args)
    contacts_disabled, _ = cc.build(args_disabled)
    assert len(contacts_disabled) == 2


if __name__ == "__main__":
    pytest.main(["-q"])
