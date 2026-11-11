"""Score Docker-pipeline batches with the frozen Stage 2 model and grow live networks (DATA_CONTRACT.md).

Per batch k: score its transactions with node features through k-1, update the live network,
then upsert package k's features. State + cursor are saved together, so a restart resumes cleanly.
"""
import torch  # import before pandas: on Windows a later torch import can fail with WinError 1114
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from stage2.transaction_model import TransactionAccountTGN
from inference.features import edge_features, node_features
from inference.live_network import LiveConfig, LiveNetwork, atomic_write_text

CONTRACT_VERSION = '1.3'
HOURS_SINCE = config.ACCOUNT_FEATURE_COLUMNS.index('hours_since_last_txn')
CHECKPOINT_SHA256 = '37d44ad8a62f01b7fe5b5dbf45580d99a0898938af2512c0a99fcd977714d084'
MICRO_BATCH = 10_000
FEATURE_TABLES = {'account': ('account_features', 'account_id', config.ACCOUNT_FEATURE_COLUMNS),
                  'bank': ('bank_features', 'bank_id', config.BANK_FEATURE_COLUMNS),
                  'entity': ('entity_features', 'entity_id', config.ENTITY_FEATURE_COLUMNS)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Registry:
    """Stable external id -> tensor row, assigned on first sight, never reused within a run."""
    def __init__(self, kind, capacity):
        self.kind, self.capacity, self.rows, self.ids = kind, capacity, {}, []

    def ensure(self, ids):
        out = np.empty(len(ids), dtype=np.int64)
        for i, key in enumerate(ids):
            row = self.rows.get(key)
            if row is None:
                if len(self.ids) >= self.capacity:
                    raise RuntimeError(f'{self.kind} capacity {self.capacity:,} exceeded; raise --{self.kind}-margin')
                row = self.rows[key] = len(self.ids)
                self.ids.append(key)
            out[i] = row
        return out


def read_replay(path):
    """Replay CSV in Kafka publish order: row i is offset i (stable Timestamp sort, as producer.py)."""
    raw = pd.read_csv(path, dtype={'Account': str, 'Account.1': str, 'From Bank': np.int64, 'To Bank': np.int64})
    raw = raw.sort_values('Timestamp', kind='stable').reset_index(drop=True)
    ts = pd.to_datetime(raw['Timestamp'], format='%Y/%m/%d %H:%M', utc=True)
    events = pd.DataFrame({
        'event_idx': np.arange(len(raw), dtype=np.int64),
        'timestamp_unix': ts.astype('int64').to_numpy()//10**9,
        'sender_account_id': raw['From Bank'].astype(str) + '::' + raw['Account'],
        'receiver_account_id': raw['To Bank'].astype(str) + '::' + raw['Account.1'],
        'from_bank': raw['From Bank'].astype(str), 'to_bank': raw['To Bank'].astype(str),
        'amount_paid': raw['Amount Paid'].astype(np.float64), 'amount_received': raw['Amount Received'].astype(np.float64),
        'payment_currency': raw['Payment Currency'].astype(str), 'receiving_currency': raw['Receiving Currency'].astype(str),
        'payment_format': raw['Payment Format'].astype(str)})
    # Offline labels never enter the model; kept only for inference/evaluate.py.
    labels = pd.DataFrame({'event_idx': events.event_idx, 'is_laundering': raw['Is Laundering'].astype(np.int32)}) \
        if 'Is Laundering' in raw else None
    return events, labels


class Consumer:
    def __init__(self, args):
        self.args = args
        data = args.pipeline_data
        self.updates_root = data/'streaming'/'model_updates'
        self.metrics_root = data/'streaming'/'audit'/'metrics'
        sys.path.insert(0, str(args.pipeline_source))
        from streaming.operations.model_update_feed import ready_batches
        self.ready_batches = ready_batches
        if sha256(args.checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint SHA-256 differs from DATA_CONTRACT.md §1; refusing to run')
        self.checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        self.pre = self.checkpoint['dataset_metadata']['preprocessing']
        self.thresholds = self.checkpoint['thresholds']
        if self.checkpoint['settings']['model']['edge_feat_dim'] != 49:
            raise ValueError('Frozen model must use 49 transaction features')
        for kind, (_, _, columns) in FEATURE_TABLES.items():
            if self.pre[kind]['columns'] != list(columns):
                raise ValueError(f'{kind} feature order differs from the frozen preprocessing')
        lookup = pd.read_parquet(args.entity_lookup or data/'processed'/'account_entity_lookup.parquet',
                                 columns=['account_id', 'entity_id'])
        self.entity_lookup = dict(zip(lookup.account_id.astype(str), lookup.entity_id.astype(str)))

    # ---- run setup / state ------------------------------------------------
    def start_run(self, run_id):
        a = self.args
        self.run_id = run_id
        self.out = a.output_root/f'run_id={run_id}'
        self.state_dir = self.out/'state'
        self.events, labels = read_replay(a.pipeline_data/'streaming'/'replay_windows'/f'{run_id}.csv')
        if labels is not None:
            (self.out/'eval').mkdir(parents=True, exist_ok=True)
            labels.to_parquet(self.out/'eval'/'labels_transactions.parquet', index=False)
        accounts = pd.unique(np.concatenate([self.events.sender_account_id, self.events.receiver_account_id]))
        banks = pd.unique(np.concatenate([self.events.from_bank, self.events.to_bank]))
        # One entity per account, so the account count bounds the entity count.
        caps = {'account': len(accounts) + a.account_margin, 'bank': len(banks) + a.bank_margin,
                'entity': len(accounts) + a.account_margin}
        margs = dict(self.checkpoint['settings']['model'],
                     num_account_nodes=caps['account'], num_bank_nodes=caps['bank'], num_entity_nodes=caps['entity'])
        self.model = TransactionAccountTGN(mode='separate', **margs).to(a.device)
        params = dict(self.model.named_parameters())
        if set(params) != set(self.checkpoint['weights']):
            raise ValueError('Checkpoint weights do not match the model definition')
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(self.checkpoint['weights'][name])
        self.model.eval()
        self.registry = {kind: Registry(kind, cap) for kind, cap in caps.items()}
        self.raw = {kind: np.zeros((caps[kind], len(FEATURE_TABLES[kind][2]))) for kind in FEATURE_TABLES}
        # Rows without a feature package yet hold "no history" (raw zeros), transformed like any other row.
        empty = {kind: torch.from_numpy(node_features(self.raw[kind][:1], self.pre, kind)).repeat(caps[kind], 1)
                 for kind in FEATURE_TABLES}
        none = torch.full((caps['account'],), -1, dtype=torch.long)
        self.model.initialize(*(empty[k].to(a.device) for k in ('account', 'bank', 'entity')),
                              none.to(a.device), none.clone().to(a.device))
        self.latest = {'score': np.zeros(caps['account'], np.float32), 'scored_at': np.zeros(caps['account'], np.int64),
                       'risk_mean': np.zeros(caps['account'], np.float32), 'count': np.zeros(caps['account'], np.int64),
                       'warm': np.zeros(caps['account'], bool), 'seen': np.zeros(caps['account'], bool)}
        live = LiveConfig(account_red=self.thresholds['account_base'], account_yellow=a.account_yellow,
                          transaction_red=self.thresholds['transaction'], transaction_yellow=a.transaction_yellow,
                          top_k=a.top_k)
        self.network = LiveNetwork(live, self.out/'network_live')
        self.cursor = {'run_id': run_id, 'last_batch_id': -1, 'skipped_batch_ids': [], 'events_processed': 0,
                       'last_rescored_batch_id': -1,
                       'invalid_transactions': 0, 'unseen_categories': 0,
                       'time_origin_unix': int(self.events.timestamp_unix.iloc[0]) if len(self.events) else 0,
                       'checkpoint_sha256': CHECKPOINT_SHA256, 'contract_version': CONTRACT_VERSION}
        if (self.state_dir/'cursor.json').exists():
            self.load_state()
        self.out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(a.output_root/'LATEST', run_id)
        print(f'Run {run_id}: {len(self.events):,} replay transactions; capacities {caps}; '
              f'resuming after batch {self.cursor["last_batch_id"]}', flush=True)

    def save_state(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir/'model_state.pt.tmp'
        torch.save({name: b.cpu() for name, b in self.model.named_buffers()}, tmp)
        os.replace(tmp, self.state_dir/'model_state.pt')
        with open(self.state_dir/'host_state.pkl.tmp', 'wb') as stream:
            pickle.dump({'registry': self.registry, 'raw': self.raw, 'latest': self.latest,
                         'network': self.network}, stream)
        os.replace(self.state_dir/'host_state.pkl.tmp', self.state_dir/'host_state.pkl')
        # The cursor is written last: it only ever points at fully saved state.
        atomic_write_text(self.state_dir/'cursor.json', json.dumps(self.cursor, indent=2))

    def load_state(self):
        cursor = json.loads((self.state_dir/'cursor.json').read_text())
        if cursor['checkpoint_sha256'] != CHECKPOINT_SHA256 or cursor['contract_version'] != CONTRACT_VERSION:
            raise ValueError('Saved state belongs to a different checkpoint or contract version; refusing to resume')
        buffers = torch.load(self.state_dir/'model_state.pt', map_location='cpu')
        with torch.no_grad():
            for name, b in self.model.named_buffers():
                b.copy_(buffers[name])
        with open(self.state_dir/'host_state.pkl', 'rb') as stream:
            host = pickle.load(stream)
        self.registry, self.raw, self.latest, self.network = host['registry'], host['raw'], host['latest'], host['network']
        self.cursor = cursor

    # ---- per batch --------------------------------------------------------
    def batch_events(self, audit):
        offsets = audit['partition_offsets']
        if len(offsets) != 1:
            raise ValueError('Expected one Kafka partition per run topic')
        lo, hi = offsets[0]['minimum_offset'], offsets[0]['maximum_offset']
        if hi - lo + 1 != audit['kafka_transaction_count']:
            raise ValueError(f'Offsets {lo}..{hi} disagree with kafka_transaction_count')
        return self.events.iloc[lo:hi+1]

    def valid(self, ev):
        amounts = ev[['amount_paid', 'amount_received']].to_numpy()
        ok = np.isfinite(amounts).all(1) & (amounts >= 0).all(1) & ev.notna().all(1).to_numpy()
        return ev[ok]

    @torch.no_grad()
    def score(self, batch_id, ev):
        a, dev = self.args, self.args.device
        account, bank = self.registry['account'], self.registry['bank']
        src = account.ensure(ev.sender_account_id.to_numpy())
        dst = account.ensure(ev.receiver_account_id.to_numpy())
        # Hub maps come from static metadata (bank in the transaction, entity from the lookup); not leakage.
        for rows, banks, ids in ((src, ev.from_bank.to_numpy(), ev.sender_account_id.to_numpy()),
                                 (dst, ev.to_bank.to_numpy(), ev.receiver_account_id.to_numpy())):
            new = (self.model.account_bank[torch.from_numpy(rows).to(dev)] < 0).cpu().numpy()
            if new.any():
                self.model.account_bank[torch.from_numpy(rows[new]).to(dev)] = \
                    torch.from_numpy(bank.ensure(banks[new])).to(dev)
                entity = [self.entity_lookup.get(x) for x in ids[new]]
                known = np.array([e is not None for e in entity])
                if known.any():
                    erows = self.registry['entity'].ensure([e for e in entity if e is not None])
                    self.model.account_entity[torch.from_numpy(rows[new][known]).to(dev)] = torch.from_numpy(erows).to(dev)
        edges, unseen = edge_features(ev, self.pre)
        self.cursor['unseen_categories'] += sum(unseen.values())
        hours = (ev.timestamp_unix.to_numpy(np.float64) - self.cursor['time_origin_unix'])/3600.
        self.cursor['now_hours'] = float(hours[-1])
        position = self.cursor['events_processed'] + np.arange(1, len(ev)+1)
        in_warmup = position <= a.warmup_records
        txn_scores, touched = [], {}
        for lo in range(0, len(ev), MICRO_BATCH):
            hi = min(lo+MICRO_BATCH, len(ev))
            out = self.model(source_ids=torch.from_numpy(src[lo:hi]).to(dev), dest_ids=torch.from_numpy(dst[lo:hi]).to(dev),
                             edge_features=torch.from_numpy(edges[lo:hi]).to(dev),
                             timestamps=torch.from_numpy(hours[lo:hi]).to(dev), do_relay=True, score_accounts=True)
            txn_scores.append(out['transaction_logits'].sigmoid().cpu().numpy())
            ids = out['node_ids'].cpu().numpy()
            L = self.latest
            L['score'][ids] = out['base_account_logits'].sigmoid().cpu().numpy()
            L['risk_mean'][ids] = out['transaction_risk_mean'].cpu().numpy()
            L['count'][ids] = self.model.tgn_memory.account_event_count[out['node_ids']].cpu().numpy()
            L['scored_at'][ids] = int(ev.timestamp_unix.iloc[hi-1])
            L['warm'][ids] = bool(in_warmup[hi-1])
            L['seen'][ids] = True
            touched.update(dict.fromkeys(ids.tolist()))
        self.cursor['events_processed'] += len(ev)
        if in_warmup.all():
            return  # warm-up: memory only; nothing is written, scored for output or shown
        keep = ~in_warmup
        scores = np.concatenate(txn_scores)[keep]
        ev = ev[keep]
        in_warmup = in_warmup[keep]
        tx = ev[['event_idx', 'timestamp_unix', 'sender_account_id', 'receiver_account_id', 'amount_paid',
                 'amount_received']].copy()
        tx.insert(0, 'batch_id', batch_id)
        tx.insert(0, 'run_id', self.run_id)
        tx['transaction_risk_score'] = scores.astype(np.float32)
        tx['transaction_flag'] = scores >= self.thresholds['transaction']
        tx['in_warmup'] = in_warmup
        rows = np.fromiter(touched, dtype=np.int64)
        acc = self.account_frame(rows)
        acc.insert(1, 'batch_id', batch_id)
        for name, frame in (('transaction_scores', tx), ('account_scores', acc)):
            (self.out/name).mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self.out/name/f'batch_id={batch_id:08d}.parquet', index=False)
        # The model scores the whole pipeline batch (as trained); the network replays it in UI-sized frames.
        latest = dict(zip(acc.account_id, acc.risk_score.astype(float)))
        for lo in range(0, len(tx), a.ui_chunk):
            started = time.monotonic()
            chunk = tx.iloc[lo:lo+a.ui_chunk]
            ends = set(chunk.sender_account_id) | set(chunk.receiver_account_id)
            self.network.write_delta(self.network.process_batch(batch_id, chunk, {k: latest[k] for k in ends}))
            if a.min_seconds_per_frame > 0:
                time.sleep(max(0.0, a.min_seconds_per_frame - (time.monotonic() - started)))

    @torch.no_grad()
    def rescore_all(self, batch_id):
        """After warm-up: rescore every account seen so far, as of now.

        Training scored each account at its last touch with features measured at the end of the data.
        Packages only refresh accounts touched in a batch, so 'hours since last transaction' is first
        brought up to now for every seen account; each account is then scored at its last touch
        (its memory and neighbours unchanged) with those current features. On the final batch this
        matches the training setup. Transaction scores are never revised.
        """
        dev = self.args.device
        rows = np.flatnonzero(self.latest['seen'])
        if not len(rows):
            return
        memory = self.model.tgn_memory
        index = torch.from_numpy(rows).to(dev)
        last = memory.account_last_update[index]
        self.raw['account'][rows, HOURS_SINCE] = np.maximum(self.cursor['now_hours'] - last.cpu().numpy(), 0.0)
        memory.account_features[index] = torch.from_numpy(
            node_features(self.raw['account'][rows], self.pre, 'account')).to(dev)
        scores = []
        for chunk in index.split(4096):
            out = self.model._score(chunk, memory.read('account', chunk), memory.account_last_update[chunk][:, None])
            scores.append(out['mule_logits'].sigmoid().cpu().numpy())
        self.latest['score'][rows] = np.concatenate(scores)
        graph = list(self.network.nodes)
        if graph:
            lookup = self.registry['account'].rows
            values = self.latest['score'][[lookup[n] for n in graph]].astype(float)
            self.network.write_delta(self.network.update_scores(batch_id, dict(zip(graph, values))))
        self.cursor['last_rescored_batch_id'] = batch_id

    def account_frame(self, rows):
        L, ids = self.latest, self.registry['account'].ids
        return pd.DataFrame({'run_id': self.run_id, 'account_id': [ids[r] for r in rows],
                             'scored_at_unix': L['scored_at'][rows], 'risk_score': L['score'][rows],
                             'account_flag': L['score'][rows] >= self.thresholds['account_base'],
                             'transaction_risk_mean': L['risk_mean'][rows], 'event_count': L['count'][rows],
                             'in_warmup': L['warm'][rows]})

    @torch.no_grad()
    def apply_package(self, package):
        """Upsert batch-k features AFTER batch k is scored (DATA_CONTRACT §7, one-package lag)."""
        dev = self.args.device
        for kind, (table, key, columns) in FEATURE_TABLES.items():
            if package['columns'][table][:1] != [key] or package['columns'][table][1:] != list(columns):
                raise ValueError(f'{table} columns differ from DATA_CONTRACT §3')
            frame = pd.read_parquet(package['tables'][table])
            if not len(frame):
                continue
            values = frame[list(columns)].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite values in {table} for batch {package["batch_id"]}')
            rows = self.registry[kind].ensure(frame[key].astype(str).to_numpy())
            self.raw[kind][rows] = values
            target = getattr(self.model.tgn_memory, kind + '_features')
            target[torch.from_numpy(rows).to(dev)] = torch.from_numpy(node_features(values, self.pre, kind)).to(dev)
        for table, hub, mapping in (('account_to_bank', 'bank', 'account_bank'),
                                    ('entity_to_account', 'entity', 'account_entity')):
            frame = pd.read_parquet(package['tables'][table], columns=['account_id', hub + '_id'])
            if len(frame):
                rows = self.registry['account'].ensure(frame.account_id.astype(str).to_numpy())
                hubs = self.registry[hub].ensure(frame[hub + '_id'].astype(str).to_numpy())
                getattr(self.model, mapping)[torch.from_numpy(rows).to(dev)] = torch.from_numpy(hubs).to(dev)

    def finalize(self):
        # Final scores are always "as of the end of the replay", whatever --rescore-every was.
        if (self.cursor['events_processed'] > self.args.warmup_records
                and self.cursor['last_rescored_batch_id'] != self.cursor['last_batch_id']):
            self.rescore_all(self.cursor['last_batch_id'])
        seen = np.flatnonzero(self.latest['seen'])
        self.account_frame(seen).to_parquet(self.out/'account_scores_latest.parquet', index=False)
        self.network.write_snapshot(self.out/'network')
        manifest = {**self.cursor, 'warmup_records': self.args.warmup_records, 'thresholds': {
            'account': self.thresholds['account_base'], 'transaction': self.thresholds['transaction'],
            'account_yellow': self.args.account_yellow, 'transaction_yellow': self.args.transaction_yellow},
            'network': self.network.stats(), 'finished_at_unix': int(time.time())}
        atomic_write_text(self.out/'manifest.json', json.dumps(manifest, indent=2))

    # ---- driver -----------------------------------------------------------
    def run_once(self, run_id):
        a = self.args
        self.start_run(run_id)
        report = a.pipeline_data/'streaming'/'reports'/f'{run_id}.json'
        since_save = 0
        while True:
            progressed = False
            for package in self.ready_batches(self.updates_root, self.metrics_root, run_id, self.cursor['last_batch_id']):
                k = package['batch_id']
                if package['status'] == 'ready':
                    ev = self.batch_events(package['audit'])
                    good = self.valid(ev)
                    self.cursor['invalid_transactions'] += len(ev) - len(good)
                    if len(good):
                        self.score(k, good)
                    self.apply_package(package)
                    if (self.cursor['events_processed'] > a.warmup_records
                            and (k + 1) % a.rescore_every == 0):
                        self.rescore_all(k)
                else:
                    self.cursor['skipped_batch_ids'].append(k)
                self.cursor['last_batch_id'] = k
                since_save += 1
                if since_save >= a.state_every:
                    self.save_state()
                    since_save = 0
                done = self.cursor['events_processed']
                if done <= a.warmup_records:
                    print(f'Batch {k}: warm-up, memory only ({done:,}/{a.warmup_records:,})', flush=True)
                else:
                    print(f'Batch {k}: {done:,} transactions processed; '
                          f'{self.network.stats()["networks"]:,} networks', flush=True)
                progressed = True
            if not progressed:
                if report.exists() or not a.follow:
                    break
                # A failed run never gets a report; move on once a newer run has started.
                if not a.run_id and newest_run(self.updates_root) != run_id:
                    print(f'Run {run_id} has no more batches and a newer run exists; switching', flush=True)
                    break
                time.sleep(a.poll_seconds)
        self.save_state()
        self.finalize()
        print(f'Run {run_id} finished: {self.cursor["events_processed"]:,} transactions; outputs in {self.out}', flush=True)


def newest_run(updates_root):
    runs = sorted(updates_root.glob('run_id=*'), key=lambda p: p.stat().st_mtime)
    return runs[-1].name.split('=', 1)[1] if runs else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-data', type=Path, default=Path('muledetector_pipeline/data'))
    p.add_argument('--pipeline-source', type=Path, default=Path('muledetector_pipeline/source'))
    p.add_argument('--entity-lookup', type=Path, help='account->entity lookup used by the replay; default: '
                                                         '<pipeline-data>/processed/account_entity_lookup.parquet')
    p.add_argument('--checkpoint', type=Path, default=Path('inference/model/best.pt'))
    p.add_argument('--output-root', type=Path, default=Path('inference/outputs'))
    p.add_argument('--run-id', help='pipeline run to score; default: newest run, and keep watching for new runs')
    p.add_argument('--follow', action='store_true', help='wait for batches until the run report appears')
    p.add_argument('--watch', action='store_true', help='after a run finishes, wait for and score the next run')
    p.add_argument('--warmup-records', type=int, default=500_000,
                   help='records flagged as warm-up; full account rescoring starts after this many')
    p.add_argument('--rescore-every', type=int, default=1, help='after warm-up, rescore all accounts every N batches')
    p.add_argument('--account-yellow', type=float, default=.55)
    p.add_argument('--transaction-yellow', type=float, default=.55)
    p.add_argument('--top-k', type=int, default=25)
    p.add_argument('--state-every', type=int, default=50, help='save model state every N batches (and at run end)')
    p.add_argument('--account-margin', type=int, default=1024)
    p.add_argument('--bank-margin', type=int, default=256)
    p.add_argument('--poll-seconds', type=float, default=1.0)
    p.add_argument('--ui-chunk', type=int, default=1000, help='transactions per live-network frame')
    p.add_argument('--min-seconds-per-frame', type=float, default=0.0, help='slow frames down to watch networks grow')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    a.follow = a.follow or a.watch
    consumer = Consumer(a)
    done = set()
    while True:
        run_id = a.run_id or newest_run(consumer.updates_root)
        if run_id is not None and not a.run_id and (a.output_root/f'run_id={run_id}'/'manifest.json').exists():
            done.add(run_id)  # already finished (possibly by an earlier version); watch for the next run
        if run_id is None or run_id in done:
            if not a.watch:
                if run_id is None:
                    p.error(f'No pipeline runs under {consumer.updates_root}')
                return
            time.sleep(a.poll_seconds)
            continue
        consumer.run_once(run_id)
        done.add(run_id)
        if a.run_id or not a.watch:
            return


if __name__ == '__main__':
    main()
