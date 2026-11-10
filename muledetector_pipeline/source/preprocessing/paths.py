"""Portable input/output paths; no developer-specific filesystem locations."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS_FILE = Path(os.environ.get("AML_ACCOUNTS_PATH", PROJECT_ROOT / "data/raw/HI-Medium_accounts.csv"))
TRANSACTIONS_FILE = Path(os.environ.get("AML_TRANSACTIONS_PATH", PROJECT_ROOT / "data/raw/HI-Medium_Trans.csv"))
PROCESSED_DATA_DIR = Path(os.environ.get("AML_PROCESSED_DATA_DIR", PROJECT_ROOT / "data/processed"))
