"""Account metrics. `auprc` is sklearn average precision (not trapezoidal PR area)."""
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)


def _arrays(logits, labels):
    scores = torch.sigmoid(logits.detach()).cpu().numpy()
    y = labels.detach().cpu().numpy()
    if not len(y) or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Metrics require nonempty, finite predictions and binary labels')
    if len(np.unique(y)) != 2:
        raise ValueError('Metrics require both classes')
    return scores, y


def find_optimal_threshold(logits, labels):
    scores, y = _arrays(logits, labels)
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    # Highest threshold among exact ties: deterministic and fewer alerts.
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def compute_metrics(logits, labels, threshold=.5):
    scores, y = _arrays(logits, labels)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'auprc': float(average_precision_score(y, scores)),
            'positive_rate': float(y.mean()), 'predicted_positive_rate': float(predicted.mean()),
            'threshold': float(threshold), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'score_quantiles': np.quantile(scores, [0, .1, .5, .9, .99, 1]).tolist(),
            'positive_score_mean': float(scores[y == 1].mean()),
            'negative_score_mean': float(scores[y == 0].mean())}


def print_metrics(metrics, prefix=''):
    print(f"{prefix}: AP={metrics['auprc']:.5f} F1={metrics['f1']:.5f} "
          f"precision={metrics['precision']:.5f} recall={metrics['recall']:.5f} "
          f"predicted_positive={metrics['predicted_positive_rate']:.3%} threshold={metrics['threshold']:.5f}")


def aggregate_seed_metrics(rows):
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'std': float(np.std([r[key] for r in rows]))}
            for key in ('precision', 'recall', 'f1', 'auprc')}


def print_seed_summary(metrics, n):
    print(f'{n} seeds: {metrics}')
