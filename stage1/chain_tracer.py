"""
MuleDetector — Stage 1: Chain Tracer

Temporal motif walk algorithm: from a seed transaction, walk forward through
the receiver's outgoing edges where each next transaction's timestamp is
strictly after the previous one and within a configurable max time gap.

This is NOT a GNN layer — it is an explicit, separate algorithm for
extracting temporal chains (see PRD.md §4 for why more GNN layers cannot
substitute for this).

References:
    - Paranjape et al. (temporal motifs)
    - LAS-GNN (2025) — closest existing work, cited explicitly
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MAX_GAP_HOURS, MAX_CHAIN_LENGTH, RANDOM_SEED_SAMPLE_SIZE


class TracedChain(NamedTuple):
    """One traced temporal chain."""

    seed_event_idx: int
    hops: list[dict]  # list of {event_idx, sender, receiver, timestamp, amount_paid, amount_received}
    length: int


def build_outgoing_index(events: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Build an index: sender_account_id → array of event indices, pre-sorted
    by timestamp.

    Since events are already chronologically sorted, we just group by sender.
    """
    print("  Building outgoing transaction index...")
    grouped = events.groupby("sender_account_id")["event_idx"].apply(
        lambda x: x.values
    )
    return grouped.to_dict()


def trace_chain(
    seed_idx: int,
    events: pd.DataFrame,
    outgoing_index: dict[str, np.ndarray],
    max_gap_seconds: float,
    max_chain_length: int,
) -> TracedChain:
    """
    Trace a single temporal chain starting from a seed transaction.

    Algorithm (from ARCHITECTURE.md):
        1. Start at seed transaction (sender → receiver, time t).
        2. Look at receiver's outgoing transactions.
        3. Keep only those with time > t AND time <= t + max_gap.
        4. Take the earliest qualifying one.
        5. Repeat until no qualifying next hop or max chain length reached.
    """
    # Pre-extract arrays for fast indexing
    timestamps = events["timestamp_unix"].values
    senders = events["sender_account_id"].values
    receivers = events["receiver_account_id"].values
    amounts_paid = events["amount_paid"].values
    amounts_received = events["amount_received"].values

    hops = []

    current_idx = seed_idx
    current_time = timestamps[current_idx]

    for _ in range(max_chain_length):
        hop_info = {
            "event_idx": int(current_idx),
            "sender": senders[current_idx],
            "receiver": receivers[current_idx],
            "timestamp_unix": float(current_time),
            "amount_paid": float(amounts_paid[current_idx]),
            "amount_received": float(amounts_received[current_idx]),
        }
        hops.append(hop_info)

        # Look at the receiver's outgoing transactions
        next_sender = receivers[current_idx]
        outgoing = outgoing_index.get(next_sender)

        if outgoing is None or len(outgoing) == 0:
            break

        # Find qualifying next hops:
        #   time > current_time AND time <= current_time + max_gap
        outgoing_times = timestamps[outgoing]
        qualifying_mask = (outgoing_times > current_time) & (
            outgoing_times <= current_time + max_gap_seconds
        )

        qualifying_indices = outgoing[qualifying_mask]

        if len(qualifying_indices) == 0:
            break

        # Take the earliest qualifying one
        qualifying_times = timestamps[qualifying_indices]
        earliest_pos = np.argmin(qualifying_times)
        next_idx = qualifying_indices[earliest_pos]

        current_idx = next_idx
        current_time = timestamps[next_idx]

    return TracedChain(
        seed_event_idx=seed_idx,
        hops=hops,
        length=len(hops),
    )


def trace_chains_from_seeds(
    seed_indices: np.ndarray,
    events: pd.DataFrame,
    outgoing_index: dict[str, np.ndarray],
    max_gap_hours: float = MAX_GAP_HOURS,
    max_chain_length: int = MAX_CHAIN_LENGTH,
    desc: str = "Tracing chains",
) -> list[TracedChain]:
    """
    Trace temporal chains from a set of seed transaction indices.
    """
    max_gap_seconds = max_gap_hours * 3600.0

    chains = []
    for seed_idx in tqdm(seed_indices, desc=desc, unit="chain"):
        chain = trace_chain(
            seed_idx=seed_idx,
            events=events,
            outgoing_index=outgoing_index,
            max_gap_seconds=max_gap_seconds,
            max_chain_length=max_chain_length,
        )
        chains.append(chain)

    return chains


def run_laundering_seeded(
    events: pd.DataFrame,
    outgoing_index: dict[str, np.ndarray],
    max_gap_hours: float = MAX_GAP_HOURS,
    max_chain_length: int = MAX_CHAIN_LENGTH,
) -> list[TracedChain]:
    """
    Trace chains seeded from Is Laundering == 1 transactions.
    This is the primary signal for chain detection.
    """
    laundering_mask = events["is_laundering"] == 1
    seed_indices = events.loc[laundering_mask, "event_idx"].values

    print(f"\n  Laundering-seeded tracing: {len(seed_indices):,} seeds")

    return trace_chains_from_seeds(
        seed_indices=seed_indices,
        events=events,
        outgoing_index=outgoing_index,
        max_gap_hours=max_gap_hours,
        max_chain_length=max_chain_length,
        desc="Laundering-seeded chains",
    )


def run_random_seeded(
    events: pd.DataFrame,
    outgoing_index: dict[str, np.ndarray],
    max_gap_hours: float = MAX_GAP_HOURS,
    max_chain_length: int = MAX_CHAIN_LENGTH,
    sample_size: int = RANDOM_SEED_SAMPLE_SIZE,
    random_state: int = 42,
) -> list[TracedChain]:
    """
    Trace chains seeded from a random sample of all transactions.
    This is the required sanity-check baseline: laundering-seeded chains
    should be longer/denser than random-seeded chains.
    """
    rng = np.random.RandomState(random_state)
    n = min(sample_size, len(events))
    seed_indices = rng.choice(events["event_idx"].values, size=n, replace=False)

    print(f"\n  Random-seeded tracing: {len(seed_indices):,} seeds")

    return trace_chains_from_seeds(
        seed_indices=seed_indices,
        events=events,
        outgoing_index=outgoing_index,
        max_gap_hours=max_gap_hours,
        max_chain_length=max_chain_length,
        desc="Random-seeded chains",
    )


def compare_chain_distributions(
    laundering_chains: list[TracedChain],
    random_chains: list[TracedChain],
) -> dict:
    """
    Compare laundering-seeded vs random-seeded chain length distributions.
    This is a required sanity check per ARCHITECTURE.md.
    """
    l_lengths = np.array([c.length for c in laundering_chains])
    r_lengths = np.array([c.length for c in random_chains])

    stats = {
        "laundering_count": len(l_lengths),
        "random_count": len(r_lengths),
        "laundering_mean_length": float(np.mean(l_lengths)) if len(l_lengths) > 0 else 0,
        "laundering_median_length": float(np.median(l_lengths)) if len(l_lengths) > 0 else 0,
        "laundering_max_length": int(np.max(l_lengths)) if len(l_lengths) > 0 else 0,
        "laundering_std_length": float(np.std(l_lengths)) if len(l_lengths) > 0 else 0,
        "random_mean_length": float(np.mean(r_lengths)) if len(r_lengths) > 0 else 0,
        "random_median_length": float(np.median(r_lengths)) if len(r_lengths) > 0 else 0,
        "random_max_length": int(np.max(r_lengths)) if len(r_lengths) > 0 else 0,
        "random_std_length": float(np.std(r_lengths)) if len(r_lengths) > 0 else 0,
        # Fraction of chains with length > 1 (non-trivial chains)
        "laundering_nontrivial_fraction": float(
            np.mean(l_lengths > 1)
        ) if len(l_lengths) > 0 else 0,
        "random_nontrivial_fraction": float(
            np.mean(r_lengths > 1)
        ) if len(r_lengths) > 0 else 0,
    }

    print("\n" + "=" * 60)
    print("Chain Distribution Comparison (Laundering vs Random Seeds)")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Laundering':>12} {'Random':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Seed count':<35} {stats['laundering_count']:>12,} {stats['random_count']:>12,}")
    print(f"  {'Mean chain length':<35} {stats['laundering_mean_length']:>12.2f} {stats['random_mean_length']:>12.2f}")
    print(f"  {'Median chain length':<35} {stats['laundering_median_length']:>12.1f} {stats['random_median_length']:>12.1f}")
    print(f"  {'Max chain length':<35} {stats['laundering_max_length']:>12} {stats['random_max_length']:>12}")
    print(f"  {'Std chain length':<35} {stats['laundering_std_length']:>12.2f} {stats['random_std_length']:>12.2f}")
    print(f"  {'Non-trivial fraction (len > 1)':<35} {stats['laundering_nontrivial_fraction']:>12.3f} {stats['random_nontrivial_fraction']:>12.3f}")

    if stats["laundering_mean_length"] <= stats["random_mean_length"]:
        print(
            "\n  ⚠ WARNING: Laundering-seeded chains are NOT longer than "
            "random-seeded chains. The chain-tracing premise may need "
            "rethinking before Stage 2 architecture is finalized."
        )
    else:
        ratio = stats["laundering_mean_length"] / max(stats["random_mean_length"], 1e-6)
        print(
            f"\n  ✓ Laundering-seeded chains are {ratio:.1f}x longer on average "
            "than random-seeded chains."
        )

    return stats
