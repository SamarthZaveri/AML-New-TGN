"""
MuleDetector — Stage 1: Conservation Scorer

For each traced chain, at each hop, computes a conservation score comparing
amount-in to amount-out with configurable tolerance (fees, currency spread,
partial pass-through mean exact equality is the wrong target).

The chain-level score and per-account aggregation feed Stage 2 as an
auxiliary training signal.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONSERVATION_TOLERANCE, CONSERVATION_AGG
from stage1.chain_tracer import TracedChain


def compute_hop_conservation(
    amount_in: float,
    amount_out: float,
    tolerance: float = CONSERVATION_TOLERANCE,
) -> float:
    """
    Conservation score for a single hop.

    Score = 1.0 when amounts match perfectly, decreasing toward 0.0 as
    the gap grows.  A tolerance band accounts for fees, currency conversion
    spread, and partial pass-through.

    Formula:
        gap = |amount_in - amount_out|
        max_amount = max(amount_in, amount_out, 1e-8)
        relative_gap = max(0, gap / max_amount - tolerance)
        score = exp(-relative_gap)

    Returns a value in (0, 1].
    """
    gap = abs(amount_in - amount_out)
    max_amount = max(amount_in, amount_out, 1e-8)
    relative_gap = max(0.0, gap / max_amount - tolerance)
    return float(np.exp(-relative_gap))


def score_chain(
    chain: TracedChain,
    aggregation: str = CONSERVATION_AGG,
    tolerance: float = CONSERVATION_TOLERANCE,
) -> tuple[float, list[float]]:
    """
    Score an entire chain's flow conservation.

    For chains with > 1 hop, each consecutive pair of hops forms a
    conservation check:
        hop[i].amount_received ≈ hop[i+1].amount_paid

    Returns:
        chain_score: aggregated chain-level conservation score
        hop_scores: per-hop-pair conservation scores
    """
    if chain.length <= 1:
        # Single-hop chains have no conservation to check
        return 1.0, []

    hop_scores = []
    for i in range(chain.length - 1):
        # Amount flowing into the intermediate node = what it received
        amount_in = chain.hops[i]["amount_received"]
        # Amount flowing out = what it paid in the next hop
        amount_out = chain.hops[i + 1]["amount_paid"]

        score = compute_hop_conservation(amount_in, amount_out, tolerance)
        hop_scores.append(score)

    if len(hop_scores) == 0:
        return 1.0, []

    if aggregation == "product":
        chain_score = float(np.prod(hop_scores))
    elif aggregation == "mean":
        chain_score = float(np.mean(hop_scores))
    else:
        raise ValueError(
            f"Unknown conservation aggregation '{aggregation}'. "
            "Use 'product' or 'mean'."
        )

    return chain_score, hop_scores


def aggregate_to_accounts(
    chains: list[TracedChain],
    tolerance: float = CONSERVATION_TOLERANCE,
    aggregation: str = CONSERVATION_AGG,
) -> pd.DataFrame:
    """
    Aggregate chain tracer + conservation scorer output to per-account signals.

    Per ARCHITECTURE.md output contract:
        - in_chain (bool): whether the account appears in any traced chain
        - chain_length (int): max chain length across all chains involving
          this account
        - conservation_score (float): chain-level score, aggregated to
          account level using the same configurable strategy

    These feed Stage 2 as an auxiliary training signal.
    """
    account_chains: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for chain in chains:
        chain_score, _ = score_chain(chain, aggregation, tolerance)

        # Every account in the chain gets tagged
        seen_in_chain = set()
        for hop in chain.hops:
            sender = hop["sender"]
            receiver = hop["receiver"]

            if sender not in seen_in_chain:
                account_chains[sender].append((chain.length, chain_score))
                seen_in_chain.add(sender)

            if receiver not in seen_in_chain:
                account_chains[receiver].append((chain.length, chain_score))
                seen_in_chain.add(receiver)

    # Build per-account aggregation
    records = []
    for account_id, chain_info in account_chains.items():
        lengths = [c[0] for c in chain_info]
        scores = [c[1] for c in chain_info]

        records.append(
            {
                "account_id": account_id,
                "in_chain": True,
                "chain_length": int(np.max(lengths)),
                "conservation_score": float(np.mean(scores)),
                "num_chains": len(chain_info),
            }
        )

    df = pd.DataFrame(records)

    print(f"\n  Conservation scoring summary:")
    print(f"    Accounts in chains: {len(df):,}")
    if len(df) > 0:
        print(f"    Mean chain length: {df['chain_length'].mean():.2f}")
        print(f"    Mean conservation score: {df['conservation_score'].mean():.4f}")
        print(
            f"    Conservation score distribution: "
            f"min={df['conservation_score'].min():.4f}, "
            f"median={df['conservation_score'].median():.4f}, "
            f"max={df['conservation_score'].max():.4f}"
        )

    return df
