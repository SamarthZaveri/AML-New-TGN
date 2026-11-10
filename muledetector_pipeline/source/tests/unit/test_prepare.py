"""One-time preparation passes the canonical inputs and protects existing data."""
import os
import sys
from types import SimpleNamespace

import pytest

from preprocessing import prepare


def test_prepare_uses_the_exact_baseline_and_lookup_builders(tmp_path, monkeypatch):
    import preprocessing
    transactions, accounts = tmp_path / "transactions.csv", tmp_path / "accounts.csv"
    transactions.touch()
    accounts.touch()
    output = tmp_path / "static"
    calls = []
    monkeypatch.setenv("AML_TRANSACTIONS_PATH", "before")
    monkeypatch.setenv("AML_ACCOUNTS_PATH", "before")
    monkeypatch.setattr(preprocessing, "spark_static_pipeline", SimpleNamespace(
        main=lambda: calls.append(("baseline", list(sys.argv)))), raising=False)
    monkeypatch.setattr(preprocessing, "build_account_metadata_lookup", SimpleNamespace(
        main=lambda: calls.append(("lookup", list(sys.argv)))), raising=False)
    monkeypatch.setattr(sys, "argv", ["prepare", "--transactions-path", str(transactions),
                                      "--accounts-path", str(accounts), "--output-dir", str(output)])
    prepare.main()
    assert [name for name, command in calls] == ["baseline", "lookup"]
    assert calls[0][1][1:] == ["--full", "--end-timestamp", "2022/09/01 00:30",
                              "--output-dir", str(output), "--write-account-events"]
    assert calls[1][1][1:] == ["--output-dir", str(tmp_path / "account_entity_lookup.parquet")]
    assert os.environ["AML_TRANSACTIONS_PATH"] == str(transactions.resolve())
    assert os.environ["AML_ACCOUNTS_PATH"] == str(accounts.resolve())


def test_prepare_rejects_existing_outputs(tmp_path, monkeypatch):
    transactions, accounts, output = tmp_path / "transactions.csv", tmp_path / "accounts.csv", tmp_path / "static"
    transactions.touch()
    accounts.touch()
    output.mkdir()
    monkeypatch.setattr(sys, "argv", ["prepare", "--transactions-path", str(transactions),
                                      "--accounts-path", str(accounts), "--output-dir", str(output)])
    with pytest.raises(FileExistsError):
        prepare.main()
