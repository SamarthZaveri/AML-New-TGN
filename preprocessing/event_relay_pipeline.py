"""
MuleDetector — Event Replay Pipeline

Reads the raw HI-Medium CSV files and produces a chronologically sorted,
per-transaction event stream as Parquet. This is the data source for:
  - Stage 1: Chain Tracer (needs per-event timestamps and amounts)
  - Stage 2: TGN backbone (needs chronological event replay)

The existing Spark pipeline aggregates per-pair statistics and loses
per-event ordering — this pipeline preserves it.

Usage:
    python -m preprocessing.event_replay_pipeline
    python -m preprocessing.event_replay_pipeline --max-rows 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_ACCOUNTS_FILE,
    RAW_TRANSACTIONS_FILE,
    EVENT_STREAM_DIR,
    EVENT_STREAM_FILE,
)


# Raw transaction CSV columns (IBM HI-Medium format).
# The file has two columns named "Account"; pandas reads the second as
# "Account.1" when header=0.
TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def read_raw_transactions(
    transactions_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Read the raw transaction CSV with explicit column names.

    The source file has two 'Account' columns.  We read with header=0 and
    pandas automatically disambiguates the duplicate as 'Account.1'.
    """
    if not transactions_path.exists():
        raise FileNotFoundError(
            f"Transaction CSV not found: {transactions_path}\n"
            "Update RAW_TRANSACTIONS_FILE in config.py to your local path."
        )

    df = pd.read_csv(
        transactions_path,
        header=0,
        dtype={
            "From Bank": np.int64,
            "To Bank": np.int64,
            "Amount Received": np.float64,
            "Amount Paid": np.float64,
            "Is Laundering": np.int32,
        },
        nrows=max_rows,
    )

    # Validate expected columns are present
    for col in TRANSACTION_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in transaction CSV. "
                f"Found columns: {list(df.columns)}"
            )

    return df


def read_raw_accounts(accounts_path: Path) -> pd.DataFrame:
    """Read the raw accounts metadata CSV."""
    if not accounts_path.exists():
        raise FileNotFoundError(
            f"Accounts CSV not found: {accounts_path}\n"
            "Update RAW_ACCOUNTS_FILE in config.py to your local path."
        )

    df = pd.read_csv(
        accounts_path,
        dtype={"Bank ID": np.int64},
    )

    for col in ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in accounts CSV. "
                f"Found columns: {list(df.columns)}"
            )

    return df


def build_event_stream(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform raw transactions into a chronologically sorted event stream.

    Adds canonical account IDs (Bank::Account) and joins entity metadata
    for both sender and receiver.
    """
    # --- Parse timestamps ---
    transactions["timestamp"] = pd.to_datetime(
        transactions["Timestamp"],
        format="%Y/%m/%d %H:%M",
    )

    # --- Build canonical account IDs ---
    # account_id = "<Bank ID>::<Account Number>" prevents same account number
    # at different banks from merging.
    transactions["sender_account_id"] = (
        transactions["From Bank"].astype(str)
        + "::"
        + transactions["Account"].astype(str)
    )
    transactions["receiver_account_id"] = (
        transactions["To Bank"].astype(str)
        + "::"
        + transactions["Account.1"].astype(str)
    )

    # --- Join sender metadata ---
    sender_meta = accounts[["Bank ID", "Account Number", "Entity ID"]].rename(
        columns={
            "Bank ID": "sender_join_bank",
            "Account Number": "sender_join_account",
            "Entity ID": "sender_entity_id",
        }
    )
    transactions = transactions.merge(
        sender_meta,
        left_on=["From Bank", "Account"],
        right_on=["sender_join_bank", "sender_join_account"],
        how="left",
    ).drop(columns=["sender_join_bank", "sender_join_account"])

    # --- Join receiver metadata ---
    receiver_meta = accounts[["Bank ID", "Account Number", "Entity ID"]].rename(
        columns={
            "Bank ID": "receiver_join_bank",
            "Account Number": "receiver_join_account",
            "Entity ID": "receiver_entity_id",
        }
    )
    transactions = transactions.merge(
        receiver_meta,
        left_on=["To Bank", "Account.1"],
        right_on=["receiver_join_bank", "receiver_join_account"],
        how="left",
    ).drop(columns=["receiver_join_bank", "receiver_join_account"])

    # --- Sort chronologically ---
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    # --- Build final event stream ---
    events = pd.DataFrame(
        {
            "event_idx": np.arange(len(transactions), dtype=np.int64),
            "timestamp": transactions["timestamp"],
            "timestamp_unix": transactions["timestamp"]
            .astype(np.int64)
            .values
            // 10**9,  # seconds since epoch
            "sender_account_id": transactions["sender_account_id"],
            "receiver_account_id": transactions["receiver_account_id"],
            "from_bank": transactions["From Bank"],
            "to_bank": transactions["To Bank"],
            "amount_paid": transactions["Amount Paid"],
            "amount_received": transactions["Amount Received"],
            "payment_currency": transactions["Payment Currency"],
            "receiving_currency": transactions["Receiving Currency"],
            "payment_format": transactions["Payment Format"],
            "sender_entity_id": transactions["sender_entity_id"],
            "receiver_entity_id": transactions["receiver_entity_id"],
            "is_laundering": transactions["Is Laundering"],
        }
    )

    # Validate no NaT timestamps
    nat_count = events["timestamp"].isna().sum()
    if nat_count > 0:
        print(
            f"WARNING: {nat_count} transactions have unparseable timestamps "
            "and will be dropped."
        )
        events = events.dropna(subset=["timestamp"]).reset_index(drop=True)
        events["event_idx"] = np.arange(len(events), dtype=np.int64)

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chronological event stream from raw HI-Medium CSVs."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of transaction rows read (for testing).",
    )
    parser.add_argument(
        "--transactions-file",
        type=Path,
        default=RAW_TRANSACTIONS_FILE,
        help="Path to HI-Medium_Trans.csv",
    )
    parser.add_argument(
        "--accounts-file",
        type=Path,
        default=RAW_ACCOUNTS_FILE,
        help="Path to HI-Medium_accounts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVENT_STREAM_DIR,
        help="Output directory for events.parquet",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MuleDetector — Event Replay Pipeline")
    print("=" * 60)

    print(f"\nReading transactions from: {args.transactions_file}")
    transactions = read_raw_transactions(
        args.transactions_file,
        max_rows=args.max_rows,
    )
    print(f"  Loaded {len(transactions):,} transactions")

    print(f"Reading accounts from: {args.accounts_file}")
    accounts = read_raw_accounts(args.accounts_file)
    print(f"  Loaded {len(accounts):,} accounts")

    print("\nBuilding chronological event stream...")
    events = build_event_stream(transactions, accounts)

    print(f"  Total events: {len(events):,}")
    print(f"  Time range: {events['timestamp'].min()} → {events['timestamp'].max()}")
    print(f"  Unique sender accounts: {events['sender_account_id'].nunique():,}")
    print(f"  Unique receiver accounts: {events['receiver_account_id'].nunique():,}")

    laundering_count = int(events["is_laundering"].sum())
    print(
        f"  Laundering events: {laundering_count:,} "
        f"({100 * laundering_count / len(events):.2f}%)"
    )

    # Save
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "events.parquet"

    events.to_parquet(output_file, index=False, engine="pyarrow")
    print(f"\nSaved event stream to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
