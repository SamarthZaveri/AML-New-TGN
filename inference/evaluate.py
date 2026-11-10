"""Offline metrics for one inference run at the frozen thresholds (DATA_CONTRACT §9). Nothing is tuned here."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def metrics(scores, labels, threshold):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {'rows': int(len(labels)), 'positives': int(labels.sum()), 'note': 'needs both classes'}
    flag = scores >= threshold
    tp = int((flag & (labels == 1)).sum()); fp = int((flag & (labels == 0)).sum()); fn = int((~flag & (labels == 1)).sum())
    precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
    return {'rows': int(len(labels)), 'positives': int(labels.sum()), 'positive_rate': float(labels.mean()),
            'auprc': float(average_precision_score(labels, scores)), 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/max(precision+recall, 1e-12), 'tp': tp, 'fp': fp, 'fn': fn, 'threshold': threshold}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='inference/outputs/run_id=<R>')
    a = p.parse_args()
    manifest = json.loads((a.run_dir/'manifest.json').read_text())
    thresholds = manifest['thresholds']
    tx = pd.concat(pd.read_parquet(f) for f in sorted((a.run_dir/'transaction_scores').glob('*.parquet')))
    labels = pd.read_parquet(a.run_dir/'eval'/'labels_transactions.parquet')
    tx = tx.merge(labels, on='event_idx', how='left', validate='one_to_one')
    if tx.is_laundering.isna().any():
        raise ValueError('Scored transactions without labels')
    # Account label: ever in a laundering transaction within this replay.
    bad = pd.concat([tx.loc[tx.is_laundering == 1, 'sender_account_id'], tx.loc[tx.is_laundering == 1, 'receiver_account_id']])
    acc = pd.read_parquet(a.run_dir/'account_scores_latest.parquet')
    acc['label'] = acc.account_id.isin(set(bad)).astype(int)
    result = {'run_dir': str(a.run_dir), 'warmup_records': manifest['warmup_records']}
    for name, mask in (('after_warmup', ~tx.in_warmup), ('all', np.ones(len(tx), bool))):
        result[f'transactions_{name}'] = metrics(tx.transaction_risk_score[mask], tx.is_laundering[mask], thresholds['transaction'])
    for name, mask in (('after_warmup', ~acc.in_warmup), ('all', np.ones(len(acc), bool))):
        result[f'accounts_{name}'] = metrics(acc.risk_score[mask], acc.label[mask], thresholds['account'])
    (a.run_dir/'eval'/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
