"""Frozen HI-Medium preprocessing applied to streamed inputs (DATA_CONTRACT §6). Nothing is refitted."""
import numpy as np
import pandas as pd
from stage2.preprocessing import transform

EVENT_COLUMNS = ['event_idx', 'timestamp_unix', 'sender_account_id', 'receiver_account_id', 'from_bank', 'to_bank',
                 'amount_paid', 'amount_received', 'payment_currency', 'receiving_currency', 'payment_format']
CATEGORICAL = ('payment_currency', 'receiving_currency', 'payment_format')


def edge_features(events, preprocessing):
    """Mirror of TransactionDataset._events, transform-only. Returns float32 [n, 49]; checked by check_parity.py."""
    amounts = events[['amount_paid', 'amount_received']].to_numpy(np.float64)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError('Amounts must be finite and nonnegative')
    utc = pd.to_datetime(events.timestamp_unix, unit='s', utc=True)
    hour = utc.dt.hour.to_numpy() + utc.dt.minute.to_numpy()/60
    dow = utc.dt.dayofweek.to_numpy()
    extra = np.column_stack([
        (events.sender_account_id.to_numpy() == events.receiver_account_id.to_numpy()).astype(float),
        (events.from_bank.astype(str).to_numpy() != events.to_bank.astype(str).to_numpy()).astype(float),
        (events.payment_currency.astype(str).to_numpy() != events.receiving_currency.astype(str).to_numpy()).astype(float),
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
    parts = [transform(amounts, preprocessing['edge']), transform(extra, preprocessing['transaction_numeric'])]
    unseen = {}
    for name in CATEGORICAL:
        vocab = preprocessing['transaction_vocabularies'][name]
        mapping = {value: i+1 for i, value in enumerate(vocab)}  # 0 = unseen
        ids = events[name].astype(str).map(mapping).fillna(0).to_numpy(np.int64)
        unseen[name] = int((ids == 0).sum())
        parts.append(np.eye(len(vocab)+1, dtype=np.float32)[ids])
    return np.concatenate(parts, axis=1).astype(np.float32), unseen


def node_features(raw, preprocessing, kind):
    """raw: float64 [n, dim] in config column order -> float32 model inputs."""
    return transform(raw, preprocessing[kind])
