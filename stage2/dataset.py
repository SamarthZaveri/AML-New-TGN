"""Validated event stream with optional event-time windows and held-out account labels.

Chronological mode restricts event replay but assumes static inputs existed at the
cutoff; this must be checked before interpreting results as prospective.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from stage2.preprocessing import fit_transform
from config import (EVENT_STREAM_FILE, SPARK_STATIC_DIR, CHAIN_SIGNALS_FILE,
                    ACCOUNT_FEATURE_COLUMNS, BANK_FEATURE_COLUMNS, ENTITY_FEATURE_COLUMNS, BATCH_SIZE)


def _unique(df, key):
    df = df.copy()
    if df[key].isna().any():
        raise ValueError(f'Missing {key}')
    df[key] = df[key].astype(str)
    if df[key].duplicated().any():
        raise ValueError(f'Duplicate {key}; resolve duplicate/conflicting rows upstream')
    return df.reset_index(drop=True)


def _sha(path):
    """Hash a regular file or a Spark/partitioned Parquet directory.

    File digests retain the v3 format. Directory digests include sorted
    relative paths and content digests, so relocation is harmless but a
    renamed, added, removed or changed part invalidates recovery state.
    """
    path = Path(path)
    if path.is_dir():
        parts = sorted((p for p in path.rglob('*') if p.is_file()),
                       key=lambda p: p.relative_to(path).as_posix())
        if not parts:
            raise ValueError(f'Empty dataset directory: {path}')
        h = hashlib.sha256(b'MuleDetector-directory-v1\0')
        for part in parts:
            name = part.relative_to(path).as_posix().encode('utf-8')
            h.update(len(name).to_bytes(8, 'big'))
            h.update(name)
            h.update(bytes.fromhex(_sha(part)))
        return h.hexdigest()
    # Do not conceal genuine missing-file or file-permission errors.
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class TemporalEventDataset:
    def __init__(self, events_file=EVENT_STREAM_FILE, static_dir=SPARK_STATIC_DIR,
                 chain_signals_file=CHAIN_SIGNALS_FILE, batch_size=BATCH_SIZE,
                 test_size=.30, val_fraction=.50, random_state=42, split_mode='account'):
        if split_mode not in ('account', 'chronological'):
            raise ValueError('split_mode must be account or chronological')
        self.split_mode = split_mode
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        static_dir, events_file, chain_signals_file = map(Path, (static_dir, events_file, chain_signals_file))
        names = ['account_labels.parquet', 'node_account_features.parquet', 'node_bank_features.parquet',
                 'node_entity_features.parquet', 'edge_account_to_bank.parquet', 'edge_entity_to_account.parquet']
        print('Validating dataset identity (SHA-256 of input files)...', flush=True)
        sources = {name: _sha(static_dir/name) for name in names}
        sources['events'] = _sha(events_file)
        sources['chain_signals'] = _sha(chain_signals_file) if chain_signals_file.exists() else None
        labels = _unique(pd.read_parquet(static_dir/'account_labels.parquet'), 'account_id')
        if not labels.account_label.isin([0, 1]).all():
            raise ValueError('account_label must be 0 or 1')
        self.account_index = pd.Index(labels.account_id)
        self.account_labels = torch.tensor(labels.account_label.to_numpy(), dtype=torch.long)
        self.num_accounts = len(labels)
        y = self.account_labels.numpy()
        if min((y == 0).sum(), (y == 1).sum()) < 10:
            raise ValueError('Need at least 10 accounts per class for a meaningful stratified split')
        tr, temp = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state)
        va, te = train_test_split(temp, test_size=val_fraction, stratify=y[temp], random_state=random_state)
        for split, ids in [('train', tr), ('val', va), ('test', te)]:
            mask = torch.zeros(len(y), dtype=torch.bool)
            mask[ids] = True
            if len(np.unique(y[ids])) != 2:
                raise ValueError(f'{split} must contain both classes')
            setattr(self, split+'_mask', mask)
        print(f'Accounts: {len(y):,}; positives: {y.sum():,}; train/val/test: {len(tr):,}/{len(va):,}/{len(te):,}')
        accounts = _unique(pd.read_parquet(static_dir/'node_account_features.parquet'), 'account_id')
        banks = _unique(pd.read_parquet(static_dir/'node_bank_features.parquet'), 'bank_id')
        entities = _unique(pd.read_parquet(static_dir/'node_entity_features.parquet'), 'entity_id')
        self.bank_index, self.entity_index = pd.Index(banks.bank_id), pd.Index(entities.entity_id)
        self.num_banks, self.num_entities = len(banks), len(entities)
        self.account_bank_map = self._mapping(static_dir/'edge_account_to_bank.parquet', 'bank_id', self.bank_index)
        self.account_entity_map = self._mapping(static_dir/'edge_entity_to_account.parquet', 'entity_id', self.entity_index)
        ordered = labels[['account_id']].merge(accounts, on='account_id', how='left', validate='one_to_one', indicator=True)
        missing = int((ordered['_merge'] == 'left_only').sum())
        if missing:
            raise ValueError(f'{missing} labeled accounts have no static feature row')
        self.preprocessing = {}
        train = self.train_mask.numpy()
        for kind, frame, columns, fit_mask in [
                ('account', ordered, ACCOUNT_FEATURE_COLUMNS, train),
                ('bank', banks, BANK_FEATURE_COLUMNS, self._hub_train_mask(self.account_bank_map, self.num_banks)),
                ('entity', entities, ENTITY_FEATURE_COLUMNS, self._hub_train_mask(self.account_entity_map, self.num_entities))]:
            if not len(frame):
                raise ValueError(f'No {kind} features')
            values, state = fit_transform(frame[list(columns)].to_numpy(), fit_mask)
            self.preprocessing[kind] = {**state, 'columns': list(columns)}
            setattr(self, kind+'_features', torch.from_numpy(values))
        self._chains(chain_signals_file)
        self._events(events_file)
        self.metadata = {'schema': 3, 'sources': sources, 'batch_size': self.batch_size,
                         'split_seed': random_state, 'test_size': test_size, 'val_fraction': val_fraction,
                         'split_mode': self.split_mode, 'event_windows': self.event_windows,
                         'num_accounts': self.num_accounts, 'num_banks': self.num_banks,
                         'num_entities': self.num_entities, 'events': len(self.event_timestamps),
                         'preprocessing': self.preprocessing}
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def _mapping(self, path, hub_col, hub_index):
        edges = pd.read_parquet(path)[['account_id', hub_col]].drop_duplicates().copy()
        if edges.isna().any().any():
            raise ValueError(f'Missing metadata IDs in {path.name}')
        edges = edges.astype(str)
        if edges.account_id.duplicated().any():
            raise ValueError(f'Multiple {hub_col} relationships per account; dense relay requires one. Do not discard ownership edges.')
        a = self.account_index.get_indexer(edges.account_id)
        h = hub_index.get_indexer(edges[hub_col])
        if (a < 0).any() or (h < 0).any():
            raise ValueError(f'Unknown metadata IDs in {path.name}')
        result = np.full(self.num_accounts, -1, dtype=np.int64)
        result[a] = h
        return torch.from_numpy(result)

    def _hub_train_mask(self, mapping, n):
        mask = np.zeros(n, dtype=bool)
        ids = mapping[self.train_mask].numpy()
        mask[ids[ids >= 0]] = True
        return mask

    def _chains(self, path):
        n = self.num_accounts
        membership, scores = np.zeros(n, np.float32), np.zeros(n, np.float32)
        known, scored = np.zeros(n, bool), np.zeros(n, bool)
        self.has_chain_signals = path.exists()
        if self.has_chain_signals:
            cs = _unique(pd.read_parquet(path), 'account_id')
            ids = self.account_index.get_indexer(cs.account_id)
            if (ids < 0).any():
                raise ValueError('Unknown accounts in chain signals')
            mem = cs.in_chain.to_numpy(float) if 'in_chain' in cs else np.ones(len(cs))
            if not np.isin(mem, [0, 1]).all():
                raise ValueError('in_chain must be explicitly 0/1 for listed accounts')
            membership[ids], known[ids] = mem, True
            if 'conservation_score' in cs:
                score = cs.conservation_score.to_numpy(float)
                valid = np.isfinite(score) & (mem == 1)
                if ((score[valid] < 0) | (score[valid] > 1)).any():
                    raise ValueError('Chain scores must be in [0,1]')
                scores[ids[valid]], scored[ids[valid]] = score[valid], True
        self.chain_membership = torch.from_numpy(membership)
        self.chain_scores = torch.from_numpy(scores)
        self.chain_known_mask, self.chain_score_mask = torch.from_numpy(known), torch.from_numpy(scored)
        training_known = membership[known & self.train_mask.numpy()]
        self.chain_has_binary_labels = len(np.unique(training_known)) == 2
        print(f'Chain labels: {known.sum():,} known, {n-known.sum():,} unknown; binary task: {self.chain_has_binary_labels}')

    def _events(self, path):
        cols = ['sender_account_id', 'receiver_account_id', 'timestamp_unix', 'amount_paid', 'amount_received']
        ev = pd.read_parquet(path, columns=cols)
        if ev[cols].isna().any().any():
            raise ValueError('Missing event values')
        s = self.account_index.get_indexer(ev.sender_account_id.astype(str))
        r = self.account_index.get_indexer(ev.receiver_account_id.astype(str))
        if (s < 0).any() or (r < 0).any():
            raise ValueError('Unknown event accounts; repair mapping instead of silently dropping events')
        ts = ev.timestamp_unix.to_numpy(np.float64)
        amounts = ev[['amount_paid', 'amount_received']].to_numpy(np.float64)
        del ev
        if not len(ts) or not np.isfinite(ts).all() or not np.isfinite(amounts).all() or (amounts < 0).any():
            raise ValueError('Events must be nonempty with finite timestamps and nonnegative finite amounts')
        order = np.argsort(ts, kind='stable')
        ts, s, r, amounts = ts[order], s[order], r[order], amounts[order]
        self.time_origin = float(ts[0])
        hours = (ts - ts[0])/3600.
        # First subtract in float64, and retain float64 relative clocks.
        self.event_timestamps = torch.from_numpy(hours)
        self.event_source_ids, self.event_dest_ids = torch.from_numpy(s), torch.from_numpy(r)
        n = len(ts)
        if self.split_mode == 'chronological':
            # Do not split simultaneous transactions across boundaries.
            train_end = int(np.searchsorted(ts, ts[int(n*.70)], side='left'))
            val_end = int(np.searchsorted(ts, ts[int(n*.85)], side='left'))
            if not 0 < train_end < val_end < n:
                raise ValueError('Chronological windows require distinct timestamps and nonempty 70/15/15 segments')
            self.event_windows = {'train': (0, train_end), 'val': (train_end, val_end),
                                  'test': (val_end, n)}
            print(f'Chronological events: train={train_end:,}, val={val_end-train_end:,}, test={n-val_end:,}; '
                  f'cutoffs Unix seconds: {ts[train_end]:.3f}, {ts[val_end]:.3f}')
        else:
            self.event_windows = {'all': (0, n)}
        train = self.train_mask.numpy()
        fit = train[s] & train[r]
        if self.split_mode == 'chronological':
            fit &= np.arange(n) < self.event_windows['train'][1]
        edge, state = fit_transform(amounts, fit, force_log=True, clip=6.)
        self.preprocessing['edge'] = {**state, 'columns': ['amount_paid', 'amount_received']}
        self.preprocessing['time'] = {'origin_unix_seconds': self.time_origin, 'unit': 'hours'}
        if amounts.max() > np.finfo(np.float32).max:
            raise ValueError('Amounts exceed float32 range')
        self.event_amounts_paid = torch.from_numpy(amounts[:, 0].astype(np.float32))
        self.event_amounts_received = torch.from_numpy(amounts[:, 1].astype(np.float32))
        self.event_edge_features = torch.from_numpy(edge)
        self.edge_feat_dim = 2
        print(f'Events: {len(ts):,}; span: {hours[-1]:.2f} hours; verify source timestamps are seconds')

    def num_batches(self, split='all'):
        if self.split_mode == 'chronological' and split == 'all':
            return (len(self.event_timestamps) + self.batch_size-1)//self.batch_size
        if split not in self.event_windows:
            raise ValueError(f'Unknown event window {split!r}; available: {list(self.event_windows)}')
        start, end = self.event_windows[split]
        return (end-start + self.batch_size-1)//self.batch_size

    def get_batch(self, batch_idx, device, split='all'):
        if not 0 <= batch_idx < self.num_batches(split):
            raise IndexError(batch_idx)
        if self.split_mode == 'chronological' and split == 'all':
            start, end = 0, len(self.event_timestamps)
        else:
            start, end = self.event_windows[split]
        a = start + batch_idx*self.batch_size
        b = min(a+self.batch_size, end)
        return {key: getattr(self, 'event_'+key)[a:b].to(device) for key in
                ('source_ids', 'dest_ids', 'edge_features', 'timestamps', 'amounts_paid', 'amounts_received')}

    def compute_pos_weight(self, max_pos_weight=50.):
        y = self.account_labels[self.train_mask]
        value = min(float((y == 0).sum()/y.sum()), max_pos_weight)
        print(f'Training-account pos_weight: {value:.3f}')
        return value
