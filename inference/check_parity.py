"""Prove inference/features.py reproduces the training inputs exactly (DATA_CONTRACT §6).

Rebuilds the HI-Medium training dataset (a few minutes), then compares its transaction and
static node features with the inference transforms driven by the frozen checkpoint state.
"""
import torch  # before pandas (Windows WinError 1114)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from stage2.transaction_dataset import TransactionDataset
from inference.features import EVENT_COLUMNS, edge_features, node_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('inference/model/best.pt'))
    p.add_argument('--rows', type=int, default=200_000)
    a = p.parse_args()
    pre = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['dataset_metadata']['preprocessing']
    ds = TransactionDataset(config.EVENT_STREAM_FILE, config.SPARK_STATIC_DIR, config.CHAIN_SIGNALS_FILE, 10000,
                            split_mode='chronological')
    for key in pre:
        if ds.preprocessing[key] != pre[key]:
            raise SystemExit(f'FAIL: frozen preprocessing[{key!r}] differs from the training dataset')
    ev = pd.read_parquet(config.EVENT_STREAM_FILE, columns=EVENT_COLUMNS)
    ev = ev.iloc[np.argsort(ev.timestamp_unix.to_numpy(np.float64), kind='stable')].reset_index(drop=True)
    ev = ev.iloc[:a.rows]
    mine, _ = edge_features(ev, pre)
    diff = np.abs(mine - ds.event_edge_features[:a.rows].numpy()).max()
    print(f'transaction features: max abs diff over {a.rows:,} events = {diff:.2e}')
    worst = diff
    static = pd.read_parquet(config.SPARK_STATIC_DIR/'node_account_features.parquet').set_index('account_id')
    ids = ds.account_index[:a.rows]
    raw = static.loc[ids, config.ACCOUNT_FEATURE_COLUMNS].to_numpy(np.float64)
    diff = np.abs(node_features(raw, pre, 'account') - ds.account_features[:a.rows].numpy()).max()
    print(f'account features: max abs diff over {len(ids):,} accounts = {diff:.2e}')
    worst = max(worst, diff)
    print('PASS' if worst < 1e-5 else 'FAIL')


if __name__ == '__main__':
    main()
