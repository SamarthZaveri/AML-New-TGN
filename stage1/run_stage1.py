"""
MuleDetector — Stage 1: Orchestrator

Runs the complete Stage 1 pipeline:
    1. Load raw event stream
    2. Run chain tracer (laundering-seeded + random baseline)
    3. Run conservation scorer
    4. Run relay checker
    5. Compare chain distributions (required sanity check)
    6. Save per-account chain signals for Stage 2

Usage:
    python -m stage1.run_stage1
    python -m stage1.run_stage1 --events-file data/processed/event_stream/events.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EVENT_STREAM_FILE,
    STAGE1_OUTPUT_DIR,
    CHAIN_SIGNALS_FILE,
    SPARK_STATIC_DIR,
    MAX_GAP_HOURS,
    MAX_CHAIN_LENGTH,
    RANDOM_SEED_SAMPLE_SIZE,
    CONSERVATION_TOLERANCE,
    CONSERVATION_AGG,
)
from stage1.chain_tracer import (
    build_outgoing_index,
    run_laundering_seeded,
    run_random_seeded,
    compare_chain_distributions,
)
from stage1.conservation_scorer import aggregate_to_accounts
from stage1.relay_checker import run_relay_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MuleDetector Stage 1: Chain Tracer + Conservation Scorer"
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=EVENT_STREAM_FILE,
        help="Path to the chronological events.parquet",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=SPARK_STATIC_DIR,
        help="Path to Spark static pipeline output (for relay check)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGE1_OUTPUT_DIR,
        help="Output directory for Stage 1 artifacts",
    )
    parser.add_argument(
        "--max-gap-hours",
        type=float,
        default=MAX_GAP_HOURS,
        help=f"Max time gap between consecutive hops in hours (default: {MAX_GAP_HOURS})",
    )
    parser.add_argument(
        "--max-chain-length",
        type=int,
        default=MAX_CHAIN_LENGTH,
        help=f"Max hops per chain (default: {MAX_CHAIN_LENGTH})",
    )
    parser.add_argument(
        "--skip-relay-check",
        action="store_true",
        help="Skip the Bank/Entity relay clustering check",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MuleDetector — Stage 1: Chain Tracer + Conservation Scorer")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load event stream
    # ---------------------------------------------------------------
    print(f"\n1. Loading event stream from: {args.events_file}")
    if not args.events_file.exists():
        print(
            f"   ERROR: Event stream file not found: {args.events_file}\n"
            "   Run the event replay pipeline first:\n"
            "     python -m preprocessing.event_replay_pipeline"
        )
        sys.exit(1)

    events = pd.read_parquet(args.events_file)
    print(f"   Loaded {len(events):,} events")

    # ---------------------------------------------------------------
    # 2. Build outgoing index
    # ---------------------------------------------------------------
    print("\n2. Building outgoing transaction index...")
    outgoing_index = build_outgoing_index(events)
    print(f"   Indexed {len(outgoing_index):,} sender accounts")

    # ---------------------------------------------------------------
    # 3. Run chain tracer — laundering seeds
    # ---------------------------------------------------------------
    print("\n3. Running chain tracer (laundering-seeded)...")
    laundering_chains = run_laundering_seeded(
        events=events,
        outgoing_index=outgoing_index,
        max_gap_hours=args.max_gap_hours,
        max_chain_length=args.max_chain_length,
    )
    print(f"   Traced {len(laundering_chains):,} laundering-seeded chains")

    # ---------------------------------------------------------------
    # 4. Run chain tracer — random baseline
    # ---------------------------------------------------------------
    print("\n4. Running chain tracer (random-seeded baseline)...")
    random_chains = run_random_seeded(
        events=events,
        outgoing_index=outgoing_index,
        max_gap_hours=args.max_gap_hours,
        max_chain_length=args.max_chain_length,
        sample_size=RANDOM_SEED_SAMPLE_SIZE,
    )
    print(f"   Traced {len(random_chains):,} random-seeded chains")

    # ---------------------------------------------------------------
    # 5. Compare distributions (required sanity check)
    # ---------------------------------------------------------------
    print("\n5. Comparing chain distributions...")
    comparison_stats = compare_chain_distributions(laundering_chains, random_chains)

    # ---------------------------------------------------------------
    # 6. Conservation scoring + account aggregation
    # ---------------------------------------------------------------
    print("\n6. Running conservation scoring on laundering-seeded chains...")
    chain_signals = aggregate_to_accounts(
        chains=laundering_chains,
        tolerance=CONSERVATION_TOLERANCE,
        aggregation=CONSERVATION_AGG,
    )

    # ---------------------------------------------------------------
    # 7. Relay check (optional)
    # ---------------------------------------------------------------
    relay_stats = None
    if not args.skip_relay_check:
        print("\n7. Running Bank/Entity relay check...")
        try:
            relay_stats = run_relay_check(args.static_dir)
        except FileNotFoundError as e:
            print(f"   Skipping relay check: {e}")
    else:
        print("\n7. Relay check skipped (--skip-relay-check)")

    # ---------------------------------------------------------------
    # 8. Save outputs
    # ---------------------------------------------------------------
    print("\n8. Saving Stage 1 outputs...")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save chain signals
    chain_signals.to_parquet(
        output_dir / "chain_signals.parquet",
        index=False,
        engine="pyarrow",
    )
    print(f"   Saved chain signals: {output_dir / 'chain_signals.parquet'}")
    print(f"   Accounts with chain signals: {len(chain_signals):,}")

    # Save comparison stats
    all_stats = {
        "chain_comparison": comparison_stats,
        "config": {
            "max_gap_hours": args.max_gap_hours,
            "max_chain_length": args.max_chain_length,
            "conservation_tolerance": CONSERVATION_TOLERANCE,
            "conservation_aggregation": CONSERVATION_AGG,
        },
    }
    if relay_stats is not None:
        all_stats["relay_check"] = relay_stats

    stats_file = output_dir / "stage1_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"   Saved statistics: {stats_file}")

    print("\n" + "=" * 60)
    print("Stage 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
