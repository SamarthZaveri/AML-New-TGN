"""Frozen seed-42 weights, different static-feature inputs: which can streaming inference rely on? (no retraining)

Replays HI-Medium exactly like training evaluation (stage2.train_transactions.evaluate) and reports
validation/test metrics at the frozen thresholds for each variant:
  full        static features as trained (full history; leaky; reference only)
  no_time     hours_since_last_txn and active_span_hours set to the training median
  neutral     every account, bank and entity feature set to the training median (no static signal;
              what streaming inference sees with --static-features neutral)
Also writes inference/model/neutral_features.json, the median vectors the consumer uses.
Choose the variant on validation; test is reported once, for the record.
"""
import torch  # before pandas (Windows WinError 1114)
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import config
from stage2.checkpointing import StopRequest
from stage2.transaction_dataset import TransactionDataset
from stage2.transaction_model import TransactionAccountTGN
from stage2.train_transactions import evaluate

TIME_COLUMNS = ('hours_since_last_txn', 'active_span_hours')


def training_medians(dataset):
    """Median of the transformed training-time features: per account over training accounts,
    per bank/entity over hubs linked to training accounts (the rows their scalers were fitted on)."""
    train = dataset.train_mask
    medians = {'account': dataset.account_features[train].median(0).values}
    for kind, mapping in (('bank', dataset.account_bank_map), ('entity', dataset.account_entity_map)):
        ids = mapping[train]
        ids = torch.unique(ids[ids >= 0])
        medians[kind] = getattr(dataset, kind + '_features')[ids].median(0).values
    return medians


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('inference/model/best.pt'))
    p.add_argument('--variants', nargs='+', default=['full', 'no_time', 'neutral'],
                   choices=['full', 'no_time', 'neutral'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, default=Path('inference/model/ablation.json'))
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    dataset = TransactionDataset(config.EVENT_STREAM_FILE, config.SPARK_STATIC_DIR, config.CHAIN_SIGNALS_FILE, 10000,
                                 split_mode='chronological')
    if dataset.fingerprint != checkpoint['fingerprint']:
        raise SystemExit('Dataset differs from the one the checkpoint was trained on')
    medians = training_medians(dataset)
    neutral = {kind: value.tolist() for kind, value in medians.items()}
    a.checkpoint.with_name('neutral_features.json').write_text(json.dumps(neutral, indent=2))
    model = TransactionAccountTGN(mode='separate', **checkpoint['settings']['model']).to(a.device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint['weights'][name])
    args = SimpleNamespace(device=a.device, relay_interval=1, merge_alpha=.5, mode='separate')
    original = {kind: getattr(dataset, kind + '_features').clone() for kind in ('account', 'bank', 'entity')}
    columns = list(config.ACCOUNT_FEATURE_COLUMNS)
    results = json.loads(a.output.read_text()) if a.output.exists() else {}
    with StopRequest() as stop:
        for variant in a.variants:
            for kind in original:
                setattr(dataset, kind + '_features', original[kind].clone())
            if variant == 'no_time':
                for name in TIME_COLUMNS:
                    i = columns.index(name)
                    dataset.account_features[:, i] = medians['account'][i]
            elif variant == 'neutral':
                for kind in original:
                    getattr(dataset, kind + '_features')[:] = medians[kind]
            row = {}
            for window in ('val', 'test'):
                metrics, _ = evaluate(model, dataset, args, stop, window, checkpoint['thresholds'])
                row[window] = {head: {k: metrics[head]['at_validation_threshold'][k]
                                      for k in ('auprc', 'precision', 'recall', 'f1', 'predicted_positive_rate')}
                               for head in ('account_base', 'transaction')}
                print(variant, window, json.dumps({h: round(v['auprc'], 4) for h, v in row[window].items()}), flush=True)
            results[variant] = row
            a.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
