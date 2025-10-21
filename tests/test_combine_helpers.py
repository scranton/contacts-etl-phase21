import os
import tempfile
from pathlib import Path
from io import StringIO

import pytest
import pandas as pd

from contacts_etl import combine_contacts as cc


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


if __name__ == "__main__":
    pytest.main(["-q"])
