"""
MuleDetector — Stage 1: Relay Checker

Checks accounts-per-bank and accounts-per-entity clustering distributions
to determine whether Bank/Entity relay is worth building into Stage 2.

Per ARCHITECTURE.md:
    "Near-singleton clustering means relay has nothing meaningful to relay
     through — drop it rather than build it on faith."

This settles PRD.md §7 open question:
    "Does entity-relay earn its place in the architecture, or is it
     near-singleton (most entities own ~1 account) and safe to drop?"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SPARK_STATIC_DIR


def load_metadata_edges(
    input_dir: Path = SPARK_STATIC_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load account-to-bank and entity-to-account edges from static parquets."""
    a2b = pd.read_parquet(input_dir / "edge_account_to_bank.parquet")
    e2a = pd.read_parquet(input_dir / "edge_entity_to_account.parquet")
    return a2b, e2a


def analyze_clustering(
    account_to_bank: pd.DataFrame,
    entity_to_account: pd.DataFrame,
) -> dict:
    """
    Analyze accounts-per-bank and accounts-per-entity distributions.

    Returns statistics and a recommendation on whether relay is worthwhile.
    """
    # --- Bank clustering ---
    bank_sizes = (
        account_to_bank.groupby("bank_id")["account_id"]
        .nunique()
        .values
    )

    # --- Entity clustering ---
    entity_sizes = (
        entity_to_account.groupby("entity_id")["account_id"]
        .nunique()
        .values
    )

    stats = {
        "bank": {
            "count": len(bank_sizes),
            "mean_accounts": float(np.mean(bank_sizes)),
            "median_accounts": float(np.median(bank_sizes)),
            "max_accounts": int(np.max(bank_sizes)),
            "std_accounts": float(np.std(bank_sizes)),
            "singleton_fraction": float(np.mean(bank_sizes == 1)),
            "p90_accounts": float(np.percentile(bank_sizes, 90)),
            "p99_accounts": float(np.percentile(bank_sizes, 99)),
        },
        "entity": {
            "count": len(entity_sizes),
            "mean_accounts": float(np.mean(entity_sizes)),
            "median_accounts": float(np.median(entity_sizes)),
            "max_accounts": int(np.max(entity_sizes)),
            "std_accounts": float(np.std(entity_sizes)),
            "singleton_fraction": float(np.mean(entity_sizes == 1)),
            "p90_accounts": float(np.percentile(entity_sizes, 90)),
            "p99_accounts": float(np.percentile(entity_sizes, 99)),
        },
    }

    # --- Determine if relay is worthwhile ---
    # Relay is useful when there's a heavy tail (a handful of hub
    # banks/entities with many accounts).  Near-singleton means drop it.
    bank_has_heavy_tail = (
        stats["bank"]["p99_accounts"] > 5  # some banks have many accounts
        and stats["bank"]["singleton_fraction"] < 0.9
    )
    entity_has_heavy_tail = (
        stats["entity"]["p99_accounts"] > 3
        and stats["entity"]["singleton_fraction"] < 0.9
    )

    stats["bank_relay_recommended"] = bank_has_heavy_tail
    stats["entity_relay_recommended"] = entity_has_heavy_tail

    return stats


def print_relay_report(stats: dict) -> None:
    """Print a human-readable relay check report."""
    print("\n" + "=" * 60)
    print("Relay Mechanism Check — Bank/Entity Clustering Analysis")
    print("=" * 60)

    for node_type in ["bank", "entity"]:
        s = stats[node_type]
        print(f"\n  {node_type.upper()} clustering:")
        print(f"    Total {node_type}s: {s['count']:,}")
        print(f"    Mean accounts per {node_type}: {s['mean_accounts']:.2f}")
        print(f"    Median accounts per {node_type}: {s['median_accounts']:.1f}")
        print(f"    Max accounts per {node_type}: {s['max_accounts']:,}")
        print(f"    P90 accounts per {node_type}: {s['p90_accounts']:.1f}")
        print(f"    P99 accounts per {node_type}: {s['p99_accounts']:.1f}")
        print(f"    Singleton fraction (1 account): {s['singleton_fraction']:.3f}")

    print(f"\n  Relay recommendation:")
    print(
        f"    Bank relay:   {'✓ RECOMMENDED (heavy tail detected)' if stats['bank_relay_recommended'] else '✗ NOT RECOMMENDED (near-singleton)'}"
    )
    print(
        f"    Entity relay: {'✓ RECOMMENDED (heavy tail detected)' if stats['entity_relay_recommended'] else '✗ NOT RECOMMENDED (near-singleton)'}"
    )


def run_relay_check(input_dir: Path = SPARK_STATIC_DIR) -> dict:
    """Run the complete relay check and print results."""
    account_to_bank, entity_to_account = load_metadata_edges(input_dir)
    stats = analyze_clustering(account_to_bank, entity_to_account)
    print_relay_report(stats)
    return stats
