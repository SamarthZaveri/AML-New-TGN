"""Chronological transaction supervision and training-fitted transaction attributes."""
import numpy as np
import pandas as pd
import torch
from stage2.dataset import TemporalEventDataset
from stage2.preprocessing import fit_transform

class TransactionDataset(TemporalEventDataset):
    def _events(self, path):
        super()._events(path)
        columns = ['event_idx', 'timestamp_unix', 'is_laundering', 'payment_currency',
                   'receiving_currency', 'payment_format', 'from_bank', 'to_bank']
        ev = pd.read_parquet(path, columns=columns)
        if ev[columns].isna().any().any():
            raise ValueError('Missing transaction labels/attributes; rebuild the event stream')
        if not ev.is_laundering.isin([0, 1]).all() or ev.event_idx.duplicated().any():
            raise ValueError('Require binary transaction labels and unique event_idx')
        order = np.argsort(ev.timestamp_unix.to_numpy(np.float64), kind='stable')
        ev = ev.iloc[order].reset_index(drop=True)
        self.event_transaction_labels = torch.tensor(ev.is_laundering.to_numpy(), dtype=torch.long)
        self.event_original_ids = torch.tensor(ev.event_idx.to_numpy(np.int64))
        n = len(ev)
        end = self.event_windows['train'][1]
        train = np.arange(n) < end
        # All features are available on the transaction; no label is an input.
        utc = pd.to_datetime(ev.timestamp_unix, unit='s', utc=True)
        hour = utc.dt.hour.to_numpy() + utc.dt.minute.to_numpy()/60
        dow = utc.dt.dayofweek.to_numpy()
        src, dst = self.event_source_ids.numpy(), self.event_dest_ids.numpy()
        extra = np.column_stack([
            (src == dst).astype(float),
            (ev.from_bank.astype(str) != ev.to_bank.astype(str)).astype(float),
            (ev.payment_currency.astype(str) != ev.receiving_currency.astype(str)).astype(float),
            np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
        extra, state = fit_transform(extra, train)
        self.preprocessing['transaction_numeric'] = state
        categorical, vocabs = [], {}
        for name in ('payment_currency', 'receiving_currency', 'payment_format'):
            values = ev[name].astype(str)
            vocab = sorted(values[train].unique().tolist())
            mapping = {value: i+1 for i, value in enumerate(vocab)}  # 0 = unseen
            ids = values.map(mapping).fillna(0).to_numpy(np.int64)
            categorical.append(np.eye(len(vocab)+1, dtype=np.float32)[ids])
            vocabs[name] = vocab
        self.preprocessing['transaction_vocabularies'] = vocabs
        # Refit amounts using only the chronological prefix, without endpoint-mask filtering.
        amounts = np.column_stack([self.event_amounts_paid.numpy(), self.event_amounts_received.numpy()])
        amounts, state = fit_transform(amounts, train, force_log=True, clip=6.)
        self.preprocessing['edge'] = {**state, 'columns': ['amount_paid', 'amount_received']}
        self.event_edge_features = torch.from_numpy(np.concatenate([amounts, extra, *categorical], axis=1).astype(np.float32))
        self.edge_feat_dim = self.event_edge_features.size(1)

    def get_batch(self, batch_idx, device, split='all'):
        result = super().get_batch(batch_idx, device, split)
        start, end = (0, len(self.event_timestamps)) if split == 'all' else self.event_windows[split]
        a = start + batch_idx*self.batch_size
        b = min(a+self.batch_size, end)
        result['transaction_labels'] = self.event_transaction_labels[a:b].to(device)
        result['original_event_ids'] = self.event_original_ids[a:b]
        return result

    def transaction_pos_weight(self, cap):
        a,b = self.event_windows['train']; y = self.event_transaction_labels[a:b]
        positives = int(y.sum()); negatives = len(y)-positives
        if not positives or not negatives: raise ValueError('Training transactions need both classes')
        return min(negatives/positives, cap)
