"""Offline suspicious-network reconstruction from completed transaction history."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset


EVENT_COLUMNS = [
    "sender_account_id",
    "receiver_account_id",
    "timestamp_unix",
    "amount_paid",
    "amount_received",
]
SCORE_ALIASES = ("risk_score", "mule_score", "score", "probability")


@dataclass(frozen=True)
class NetworkConfig:
    red_threshold: float = 0.80
    yellow_threshold: float = 0.55
    context_min_score: float = 0.15
    max_flow_gap_hours: float = 72.0
    amount_ratio_low: float = 0.70
    amount_ratio_high: float = 1.30
    max_paths_per_intermediary: int = 200
    max_candidate_events: int = 5_000_000
    max_context_accounts: int = 50_000


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        return parent

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def read_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(path)

    if "account_id" not in scores:
        raise ValueError("Account-score file must contain account_id")

    score_column = next((name for name in SCORE_ALIASES if name in scores), None)
    if score_column is None:
        raise ValueError(f"Account-score file must contain one of {SCORE_ALIASES}")

    scores = scores[["account_id", score_column]].rename(
        columns={score_column: "risk_score"}
    ).copy()

    scores["account_id"] = scores["account_id"].astype(str)
    scores["risk_score"] = pd.to_numeric(scores["risk_score"], errors="coerce")

    if scores.risk_score.isna().any() or (~scores.risk_score.between(0.0, 1.0)).any():
        raise ValueError("risk_score must be finite and in [0, 1]")

    return scores.groupby("account_id", as_index=False).risk_score.max()


def read_candidate_events(
    events_file: Path,
    allowed_accounts: set[str],
    max_candidate_events: int,
) -> pd.DataFrame:
    retained = []
    count = 0

    dataset = pa_dataset.dataset(events_file, format="parquet")

    for batch in dataset.scanner(
        columns=EVENT_COLUMNS,
        batch_size=262_144,
    ).to_batches():
        frame = batch.to_pandas()
        frame["sender_account_id"] = frame["sender_account_id"].astype(str)
        frame["receiver_account_id"] = frame["receiver_account_id"].astype(str)

        frame = frame.loc[
            frame.sender_account_id.isin(allowed_accounts)
            & frame.receiver_account_id.isin(allowed_accounts)
        ]

        count += len(frame)

        if count > max_candidate_events:
            raise ValueError(
                f"{count:,}+ candidate events exceed "
                f"--max-candidate-events={max_candidate_events:,}. "
                "Lower --max-context-accounts or raise the cap deliberately."
            )

        if not frame.empty:
            retained.append(frame)

    events = (
        pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )

    if events.empty:
        return events.assign(event_id=pd.Series(dtype=np.int64))

    for column in ("timestamp_unix", "amount_paid", "amount_received"):
        events[column] = pd.to_numeric(events[column], errors="coerce")

    if events[["timestamp_unix", "amount_paid", "amount_received"]].isna().any().any():
        raise ValueError("Candidate events contain non-numeric timestamp or amount values")

    if (events[["amount_paid", "amount_received"]] < 0).any().any():
        raise ValueError("Candidate events contain negative amounts")

    events = events.sort_values(
        ["timestamp_unix", "sender_account_id", "receiver_account_id"],
        kind="stable",
    )

    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events.reset_index(drop=True)


def aggregate_edges(events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["source_account_id", "target_account_id"])

    edges = (
        events.groupby(
            ["sender_account_id", "receiver_account_id"],
            as_index=False,
        )
        .agg(
            transaction_count=("event_id", "size"),
            first_timestamp_unix=("timestamp_unix", "min"),
            last_timestamp_unix=("timestamp_unix", "max"),
            amount_paid_total=("amount_paid", "sum"),
            amount_received_total=("amount_received", "sum"),
        )
        .rename(
            columns={
                "sender_account_id": "source_account_id",
                "receiver_account_id": "target_account_id",
            }
        )
    )

    lookup = scores.set_index("account_id").risk_score

    edges["source_risk_score"] = edges.source_account_id.map(lookup)
    edges["target_risk_score"] = edges.target_account_id.map(lookup)
    edges["edge_risk_score"] = np.sqrt(
        edges.source_risk_score * edges.target_risk_score
    )

    return edges.sort_values(
        ["edge_risk_score", "transaction_count"],
        ascending=False,
    ).reset_index(drop=True)


def extract_flow_paths(
    events: pd.DataFrame,
    scores: pd.DataFrame,
    config: NetworkConfig,
) -> pd.DataFrame:
    columns = [
        "source_account_id",
        "intermediary_account_id",
        "target_account_id",
        "in_event_id",
        "out_event_id",
        "in_timestamp_unix",
        "out_timestamp_unix",
        "gap_hours",
        "in_amount",
        "out_amount",
        "amount_ratio",
        "path_risk_score",
    ]

    if events.empty:
        return pd.DataFrame(columns=columns)

    score_lookup = scores.set_index("account_id").risk_score
    inbound = {
        account: frame
        for account, frame in events.groupby("receiver_account_id", sort=False)
    }
    outbound = {
        account: frame
        for account, frame in events.groupby("sender_account_id", sort=False)
    }

    rows = []
    max_gap_seconds = config.max_flow_gap_hours * 3600.0

    for intermediary in sorted(set(inbound).intersection(outbound)):
        incoming = inbound[intermediary]
        outgoing = outbound[intermediary]
        outgoing_times = outgoing.timestamp_unix.to_numpy()

        found = 0

        for incoming_event in incoming.itertuples(index=False):
            index = int(
                np.searchsorted(
                    outgoing_times,
                    incoming_event.timestamp_unix,
                    side="right",
                )
            )

            if index >= len(outgoing):
                continue

            outgoing_event = outgoing.iloc[index]
            gap = float(outgoing_event.timestamp_unix - incoming_event.timestamp_unix)

            if gap > max_gap_seconds or incoming_event.amount_received <= 0:
                continue

            amount_ratio = float(
                outgoing_event.amount_paid / incoming_event.amount_received
            )

            if not config.amount_ratio_low <= amount_ratio <= config.amount_ratio_high:
                continue

            path_risk = float(
                np.mean(
                    [
                        score_lookup[incoming_event.sender_account_id],
                        score_lookup[intermediary],
                        score_lookup[outgoing_event.receiver_account_id],
                    ]
                )
            )

            rows.append(
                (
                    incoming_event.sender_account_id,
                    intermediary,
                    outgoing_event.receiver_account_id,
                    incoming_event.event_id,
                    outgoing_event.event_id,
                    incoming_event.timestamp_unix,
                    outgoing_event.timestamp_unix,
                    gap / 3600.0,
                    incoming_event.amount_received,
                    outgoing_event.amount_paid,
                    amount_ratio,
                    path_risk,
                )
            )

            found += 1

            if found >= config.max_paths_per_intermediary:
                break

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["path_risk_score", "gap_hours"],
        ascending=[False, True],
        ignore_index=True,
    )


def build_components(
    edges: pd.DataFrame,
    scores: pd.DataFrame,
    paths: pd.DataFrame,
    config: NetworkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = scores.rename(columns={"account_id": "node_id"}).copy()

    if edges.empty:
        nodes["component_id"] = [
            f"network_{index:06d}"
            for index in range(len(nodes))
        ]
        return nodes, pd.DataFrame()

    union_find = UnionFind(nodes.node_id.tolist())

    for edge in edges.itertuples(index=False):
        union_find.union(edge.source_account_id, edge.target_account_id)

    roots = {node: union_find.find(node) for node in nodes.node_id}
    root_ids = {
        root: f"network_{index:06d}"
        for index, root in enumerate(sorted(set(roots.values())))
    }

    nodes["component_id"] = nodes.node_id.map(
        lambda node: root_ids[roots[node]]
    )

    component_lookup = nodes.set_index("node_id").component_id
    edges["component_id"] = edges.source_account_id.map(component_lookup)

    red_counts = (
        (nodes.risk_score >= config.red_threshold)
        .groupby(nodes.component_id)
        .sum()
        .rename("red_accounts")
    )

    yellow_counts = (
        (
            (nodes.risk_score >= config.yellow_threshold)
            & (nodes.risk_score < config.red_threshold)
        )
        .groupby(nodes.component_id)
        .sum()
        .rename("yellow_accounts")
    )

    summary = (
        nodes.groupby("component_id", as_index=False)
        .agg(
            account_count=("node_id", "size"),
            risk_score_max=("risk_score", "max"),
            risk_score_mean=("risk_score", "mean"),
        )
        .merge(red_counts, on="component_id")
        .merge(yellow_counts, on="component_id")
    )

    edge_summary = (
        edges.groupby("component_id", as_index=False)
        .agg(
            edge_count=("source_account_id", "size"),
            transaction_count=("transaction_count", "sum"),
            suspicious_value=("amount_paid_total", "sum"),
        )
    )

    path_counts = (
        pd.Series(dtype="int64", name="flow_path_count")
        if paths.empty
        else paths.intermediary_account_id.map(component_lookup)
        .value_counts()
        .rename("flow_path_count")
    )

    summary = (
        summary.merge(edge_summary, on="component_id", how="left")
        .merge(
            path_counts,
            left_on="component_id",
            right_index=True,
            how="left",
        )
        .fillna(
            {
                "edge_count": 0,
                "transaction_count": 0,
                "suspicious_value": 0,
                "flow_path_count": 0,
            }
        )
    )

    possible_edges = (
        summary.account_count * (summary.account_count - 1)
    ).clip(lower=1)

    density = summary.edge_count / possible_edges

    summary["network_score"] = np.clip(
        0.45 * summary.risk_score_max
        + 0.20 * summary.risk_score_mean
        + 0.15 * np.minimum(
            1.0,
            np.log1p(summary.flow_path_count) / math.log(11.0),
        )
        + 0.10 * np.minimum(
            1.0,
            np.log1p(summary.suspicious_value) / math.log(1_000_001.0),
        )
        + 0.10 * np.minimum(1.0, density * 5.0),
        0.0,
        1.0,
    )

    summary["network_bin"] = np.select(
        [
            summary.network_score >= config.red_threshold,
            summary.network_score >= config.yellow_threshold,
        ],
        ["red", "yellow"],
        default="green",
    )

    return (
        nodes,
        summary.sort_values(
            "network_score",
            ascending=False,
            ignore_index=True,
        ),
    )


def build_networks(
    events_file: Path,
    score_file: Path,
    output_dir: Path,
    config: NetworkConfig,
) -> dict[str, int]:
    if not (
        0
        <= config.context_min_score
        <= config.yellow_threshold
        <= config.red_threshold
        <= 1
    ):
        raise ValueError(
            "Require 0 <= context-min-score <= yellow-threshold <= red-threshold <= 1"
        )

    scores = read_scores(score_file)

    scores["alert_bin"] = np.select(
        [
            scores.risk_score >= config.red_threshold,
            scores.risk_score >= config.yellow_threshold,
        ],
        ["red", "yellow"],
        default="green",
    )

    selected = scores.loc[
        scores.risk_score >= config.context_min_score
    ].copy()

    if selected.empty:
        raise ValueError("No accounts meet --context-min-score")

    if config.max_context_accounts > 0:
        selected = selected.nlargest(
            config.max_context_accounts,
            "risk_score",
        ).copy()

    events = read_candidate_events(
        events_file,
        set(selected.account_id),
        config.max_candidate_events,
    )

    edges = aggregate_edges(events, selected)
    paths = extract_flow_paths(events, selected, config)
    nodes, networks = build_components(edges, selected, paths, config)

    node_components = nodes.set_index("node_id").component_id

    if not paths.empty:
        paths["component_id"] = paths.intermediary_account_id.map(
            node_components
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    nodes.to_parquet(output_dir / "network_nodes.parquet", index=False)
    edges.to_parquet(output_dir / "network_edges.parquet", index=False)
    paths.to_parquet(output_dir / "flow_paths.parquet", index=False)
    networks.to_parquet(output_dir / "networks.parquet", index=False)

    manifest = {
        "account_scores": str(score_file),
        "events": str(events_file),
        "config": config.__dict__,
        "selected_accounts": int(len(selected)),
        "candidate_events": int(len(events)),
        "network_count": int(len(networks)),
        "flow_path_count": int(len(paths)),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=__doc__
    )

    argument_parser.add_argument(
        "--events-file",
        type=Path,
        required=True,
    )
    argument_parser.add_argument(
        "--account-scores",
        type=Path,
        required=True,
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    argument_parser.add_argument(
        "--red-threshold",
        type=float,
        default=0.80,
    )
    argument_parser.add_argument(
        "--yellow-threshold",
        type=float,
        default=0.55,
    )
    argument_parser.add_argument(
        "--context-min-score",
        type=float,
        default=0.15,
    )
    argument_parser.add_argument(
        "--max-flow-gap-hours",
        type=float,
        default=72.0,
    )
    argument_parser.add_argument(
        "--amount-ratio-low",
        type=float,
        default=0.70,
    )
    argument_parser.add_argument(
        "--amount-ratio-high",
        type=float,
        default=1.30,
    )
    argument_parser.add_argument(
        "--max-paths-per-intermediary",
        type=int,
        default=200,
    )
    argument_parser.add_argument(
        "--max-candidate-events",
        type=int,
        default=5_000_000,
    )
    argument_parser.add_argument(
        "--max-context-accounts",
        type=int,
        default=50_000,
        help="Keep only highest-risk eligible accounts; 0 disables the cap",
    )

    return argument_parser


def main() -> None:
    args = parser().parse_args()

    config = NetworkConfig(
        **{
            name: getattr(args, name)
            for name in NetworkConfig.__dataclass_fields__
        }
    )

    if (
        config.max_flow_gap_hours <= 0
        or config.amount_ratio_low <= 0
        or config.amount_ratio_high < config.amount_ratio_low
        or config.max_context_accounts < 0
    ):
        raise SystemExit("Invalid post-processing settings")

    result = build_networks(
        args.events_file,
        args.account_scores,
        args.output_dir,
        config,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()