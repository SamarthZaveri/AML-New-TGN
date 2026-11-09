"""
MuleDetector — Centralized Configuration

All paths, hyperparameters, feature column lists, and model settings.
Update RAW_TRANSACTIONS_FILE and RAW_ACCOUNTS_FILE to your local paths
once you have downloaded the HI-Medium dataset.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (ml_ly_latest/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Raw IBM HI-Medium CSV files
# Update these paths once the CSVs are on your machine.
# ---------------------------------------------------------------------------
RAW_TRANSACTIONS_FILE = Path(
    r"C:\Users\Samarth\Desktop\GNN\HI-Medium_Trans.csv"
)
RAW_ACCOUNTS_FILE = Path(
    r"C:\Users\Samarth\Desktop\GNN\HI-Medium_accounts.csv"
)

# ---------------------------------------------------------------------------
# Processed data directories
# ---------------------------------------------------------------------------
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Existing Spark pipeline outputs (cold-start features + labels)
SPARK_STATIC_DIR = PROCESSED_DATA_DIR / "spark_static_full"

# New event replay pipeline output
EVENT_STREAM_DIR = PROCESSED_DATA_DIR / "event_stream"
EVENT_STREAM_FILE = EVENT_STREAM_DIR / "events.parquet"

# Stage 1 outputs
STAGE1_OUTPUT_DIR = PROCESSED_DATA_DIR / "stage1_output"
CHAIN_SIGNALS_FILE = STAGE1_OUTPUT_DIR / "chain_signals.parquet"

# Stage 2 outputs (model checkpoints, evaluation results)
STAGE2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2"
CHECKPOINT_DIR = STAGE2_OUTPUT_DIR / "checkpoints"
RESULTS_DIR = STAGE2_OUTPUT_DIR / "results"

# ---------------------------------------------------------------------------
# Feature column contracts (must match DATA_CONTRACT.md exactly)
# ---------------------------------------------------------------------------
ACCOUNT_FEATURE_COLUMNS = [
    "outgoing_txn_count",
    "incoming_txn_count",
    "outgoing_unique_counterparties",
    "incoming_unique_counterparties",
    "repeat_counterparty_ratio",
    "self_transfer_ratio",
    "outgoing_total_paid",
    "incoming_total_received",
    "outgoing_avg_paid",
    "incoming_avg_received",
    "outgoing_amount_std",
    "incoming_amount_std",
    "payment_currency_diversity",
    "receiving_currency_diversity",
    "currency_switch_ratio",
    "payment_format_diversity",
    "unique_connected_banks",
    "cross_bank_txn_ratio",
    "odd_hour_txn_ratio",
    "weekend_txn_ratio",
    "active_span_hours",
    "hours_since_last_txn",
    "max_txns_per_hour",
    "avg_inter_txn_gap_hours",
]

BANK_FEATURE_COLUMNS = [
    "bank_account_count",
    "outgoing_txn_count",
    "incoming_txn_count",
    "outgoing_total_paid",
    "incoming_total_received",
    "outgoing_avg_paid",
    "incoming_avg_received",
    "unique_sender_accounts",
    "unique_receiver_accounts",
    "counterparty_bank_count",
    "currency_diversity",
    "payment_format_diversity",
    "cross_bank_txn_ratio",
    "odd_hour_txn_ratio",
    "weekend_txn_ratio",
]

ENTITY_FEATURE_COLUMNS = [
    "entity_type_sole_proprietorship",
    "entity_type_partnership",
    "entity_type_corporation",
    "entity_type_country",
    "entity_type_other",
    "owned_account_count",
    "connected_bank_count",
    "outgoing_txn_count",
    "incoming_txn_count",
    "outgoing_total_paid",
    "incoming_total_received",
    "unique_counterparty_accounts",
    "cross_bank_txn_ratio",
    "odd_hour_txn_ratio",
    "payment_format_diversity",
]

EDGE_FEATURE_COLUMNS = [
    "edge_txn_count",
    "edge_total_paid",
    "edge_avg_paid",
    "edge_amount_std",
    "edge_is_repeated",
    "edge_recency_hours",
    "edge_currency_switch_ratio",
    "edge_payment_format_diversity",
    "edge_inter_txn_gap_hours",
]

# Feature dimension constants (derived from column lists)
ACCOUNT_FEAT_DIM = len(ACCOUNT_FEATURE_COLUMNS)   # 24
BANK_FEAT_DIM = len(BANK_FEATURE_COLUMNS)         # 15
ENTITY_FEAT_DIM = len(ENTITY_FEATURE_COLUMNS)     # 15
EDGE_FEAT_DIM = len(EDGE_FEATURE_COLUMNS)         # 9

# ---------------------------------------------------------------------------
# Stage 1 — Chain Tracer + Conservation Scorer
# ---------------------------------------------------------------------------
# Temporal motif walk
MAX_GAP_HOURS = 72          # max time gap between consecutive hops
MAX_CHAIN_LENGTH = 20       # max hops per chain (bounds runtime on dense accounts)
RANDOM_SEED_SAMPLE_SIZE = 10_000  # number of random-seed transactions for baseline

# Conservation scorer
CONSERVATION_TOLERANCE = 0.05     # 5% tolerance for fees/conversion spread
CONSERVATION_AGG = "product"      # "product" or "mean" for chain-level aggregation

# ---------------------------------------------------------------------------
# Stage 2 — TGN + TGAT Model
# ---------------------------------------------------------------------------
# Architecture
HIDDEN_DIM = 128            # hidden / memory dimension for all node types
TIME_DIM = 64               # Time2Vec encoding dimension
NUM_ATTENTION_HEADS = 4     # TGAT multi-head attention
DROPOUT = 0.1

# Training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
BATCH_SIZE = 600            # events per temporal mini-batch
PATIENCE = 10               # early stopping patience (epochs)

# Loss weights
LAMBDA_CONSERVATION = 0.1   # conservation constraint weight (sweep this)
LAMBDA_CHAIN_AUX = 0.05     # auxiliary chain-signal loss weight

# Focal loss (for class imbalance)
FOCAL_LOSS_GAMMA = 2.0      # set to 0 for standard weighted BCE
# Class weight is computed from data, not hardcoded.

# Relay mechanism
ENABLE_RELAY = True         # bank/entity heterogeneous message passing
RELAY_INTERVAL = 5          # relay every N event batches

# Evaluation
NUM_SEEDS = 3               # minimum seeds for variance reporting

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
