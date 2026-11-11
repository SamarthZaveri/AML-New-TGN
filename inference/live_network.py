"""Incremental suspicious-network builder (DATA_CONTRACT §10).

Same rules as stage2/network_postprocess.py (bins, per-pair edges, A->B->C flow paths,
network_score), applied batch by batch so networks grow while scoring runs.
Networks only merge, never split; nodes are recoloured and networks rescored as scores change.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiveConfig:
    account_red: float = 0.7912804484367371
    account_yellow: float = 0.55
    transaction_red: float = 0.9086251258850098
    transaction_yellow: float = 0.55
    entry_threshold: float = 0.55  # on transaction score x sqrt(sender score x receiver score)
    max_flow_gap_hours: float = 72.0
    amount_ratio_low: float = 0.70
    amount_ratio_high: float = 1.30
    max_paths_per_intermediary: int = 200
    top_k: int = 25
    render_nodes: int = 300
    render_edges: int = 1000
    render_paths: int = 100


def bin_of(score, red, yellow):
    return 'red' if score >= red else 'yellow' if score >= yellow else 'green'


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


class LiveNetwork:
    def __init__(self, config: LiveConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.nodes = {}        # account_id -> {score, bin, in_warmup, cid}
        self.parent = {}       # union-find over account_id
        self.root_cid = {}     # union-find root -> component id
        self.components = {}   # cid -> aggregates
        self.edges = {}        # (source, target) -> aggregates
        self.inbound = {}      # account_id -> deque of unmatched included incoming txns
        self.paths = []
        self.paths_per_intermediary = {}
        self.next_cid = 0
        self.frame = -1        # UI update counter; several frames can come from one pipeline batch
        self.top = []          # current top-k cids, ranked
        self.totals = {'transactions_scored': 0, 'transactions_included': 0,
                       'red_transactions': 0, 'yellow_transactions': 0}

    # ---- union-find -------------------------------------------------------
    def _find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def _cid(self, node):
        return self.root_cid[self._find(node)]

    def _add_node(self, account_id, score, in_warmup):
        cid = f'net_{self.next_cid:06d}'
        self.next_cid += 1
        self.parent[account_id] = account_id
        self.root_cid[account_id] = cid
        self.nodes[account_id] = {'score': score, 'bin': self._account_bin(score), 'in_warmup': in_warmup}
        self.components[cid] = {'nodes': {account_id}, 'edges': set(), 'paths': [], 'transaction_count': 0,
                                'suspicious_value': 0.0, 'flow_path_count': 0, 'network_score': 0.0}

    def _union(self, left, right, merges):
        a, b = self._find(left), self._find(right)
        if a == b:
            return
        ca, cb = self.root_cid[a], self.root_cid[b]
        # The larger network keeps its id, so on-screen identities stay stable.
        if (len(self.components[ca]['nodes']), cb) < (len(self.components[cb]['nodes']), ca):
            a, b, ca, cb = b, a, cb, ca
        self.parent[b] = a
        del self.root_cid[b]
        keep, gone = self.components[ca], self.components.pop(cb)
        keep['nodes'] |= gone['nodes']
        keep['edges'] |= gone['edges']
        keep['paths'] += gone['paths']
        for key in ('transaction_count', 'suspicious_value', 'flow_path_count'):
            keep[key] += gone[key]
        merges.append([cb, ca])

    def _account_bin(self, score):
        return bin_of(score, self.config.account_red, self.config.account_yellow)

    # ---- per batch --------------------------------------------------------
    def process_batch(self, batch_id: int, transactions: pd.DataFrame, account_scores: dict) -> dict:
        """transactions: event_idx, timestamp_unix, sender_account_id, receiver_account_id, amount_paid,
        amount_received, transaction_risk_score, in_warmup (time-ordered). account_scores: id -> latest score."""
        c = self.config
        touched, merges = set(), []  # touched holds account ids; network ids can be retired by merges
        # Recolour accounts already in the graph with their latest scores.
        for account_id, score in account_scores.items():
            node = self.nodes.get(account_id)
            if node is not None:
                node['score'], node['bin'] = score, self._account_bin(score)
                touched.add(account_id)
        gap_limit = c.max_flow_gap_hours*3600.0
        for row in transactions.itertuples(index=False):
            self.totals['transactions_scored'] += 1
            tbin = bin_of(row.transaction_risk_score, c.transaction_red, c.transaction_yellow)
            if tbin != 'green':
                self.totals[tbin + '_transactions'] += 1
            s, r = row.sender_account_id, row.receiver_account_id
            ss, rs = account_scores[s], account_scores[r]
            # Entry combines all three scores; a low-risk transaction or account on either side keeps it out.
            if row.transaction_risk_score * math.sqrt(ss * rs) < c.entry_threshold:
                continue
            self.totals['transactions_included'] += 1
            warm = bool(row.in_warmup)
            for account_id, score in ((s, ss), (r, rs)):
                if account_id not in self.nodes:
                    self._add_node(account_id, score, warm)
                else:
                    self.nodes[account_id]['in_warmup'] = warm  # faded only while its latest activity is warm-up
            self._union(s, r, merges)
            cid = self._cid(s)
            component = self.components[cid]
            key = (s, r)
            edge = self.edges.get(key)
            if edge is None:
                edge = self.edges[key] = {'transaction_count': 0, 'first_timestamp_unix': int(row.timestamp_unix),
                                          'amount_paid_total': 0.0, 'amount_received_total': 0.0,
                                          'max_transaction_score': 0.0}
                component['edges'].add(key)
            edge['transaction_count'] += 1
            edge['last_timestamp_unix'] = int(row.timestamp_unix)
            edge['amount_paid_total'] += float(row.amount_paid)
            edge['amount_received_total'] += float(row.amount_received)
            edge['max_transaction_score'] = max(edge['max_transaction_score'], float(row.transaction_risk_score))
            edge['in_warmup'] = warm
            component['transaction_count'] += 1
            component['suspicious_value'] += float(row.amount_paid)
            touched.add(s)
            # Flow paths: each incoming txn is matched once, to its intermediary's first later outgoing txn.
            queue = self.inbound.get(s)
            if queue:
                kept = deque()
                while queue:
                    incoming = queue.popleft()
                    gap = row.timestamp_unix - incoming['timestamp_unix']
                    if gap <= 0:
                        kept.append(incoming)
                        continue
                    if gap > gap_limit or incoming['amount_received'] <= 0:
                        continue
                    ratio = float(row.amount_paid)/incoming['amount_received']
                    if (c.amount_ratio_low <= ratio <= c.amount_ratio_high
                            and self.paths_per_intermediary.get(s, 0) < c.max_paths_per_intermediary):
                        source = incoming['sender_account_id']
                        self._union(source, s, merges)
                        path = {
                            'source_account_id': source, 'intermediary_account_id': s, 'target_account_id': r,
                            'in_event_id': incoming['event_idx'], 'out_event_id': int(row.event_idx),
                            'in_timestamp_unix': incoming['timestamp_unix'], 'out_timestamp_unix': int(row.timestamp_unix),
                            'gap_hours': gap/3600.0, 'in_amount': incoming['amount_received'],
                            'out_amount': float(row.amount_paid), 'amount_ratio': ratio,
                            'path_risk_score': float(np.mean([self.nodes[source]['score'], self.nodes[s]['score'],
                                                              self.nodes[r]['score']]))}
                        self.paths.append(path)
                        self.paths_per_intermediary[s] = self.paths_per_intermediary.get(s, 0) + 1
                        self.components[self._cid(s)]['flow_path_count'] += 1
                        self.components[self._cid(s)]['paths'].append(path)
                self.inbound[s] = kept
            self.inbound.setdefault(r, deque()).append({
                'event_idx': int(row.event_idx), 'timestamp_unix': int(row.timestamp_unix),
                'sender_account_id': s, 'amount_received': float(row.amount_received)})
        touched = {self._cid(n) for n in touched}
        for cid in touched:
            self._rescore(cid)
        warm = bool(transactions.in_warmup.any()) if len(transactions) else False
        return self._delta(batch_id, touched, merges, warm)

    def update_scores(self, batch_id: int, account_scores: dict) -> dict:
        """Recolour graph accounts whose score changed (full rescoring) and rescore their networks."""
        touched = set()
        for account_id, score in account_scores.items():
            node = self.nodes.get(account_id)
            if node is not None and node['score'] != score:
                node['score'], node['bin'] = score, self._account_bin(score)
                touched.add(account_id)
        touched = {self._cid(n) for n in touched}
        for cid in touched:
            self._rescore(cid)
        return self._delta(batch_id, touched, [], False)

    def _rescore(self, cid):
        c, component = self.config, self.components[cid]
        scores = np.fromiter((self.nodes[n]['score'] for n in component['nodes']), dtype=np.float64)
        n = len(scores)
        density = len(component['edges'])/max(n*(n-1), 1)
        component['risk_score_max'] = float(scores.max())
        component['risk_score_mean'] = float(scores.mean())
        component['red_accounts'] = int((scores >= c.account_red).sum())
        component['yellow_accounts'] = int(((scores >= c.account_yellow) & (scores < c.account_red)).sum())
        component['network_score'] = float(np.clip(
            0.45*scores.max() + 0.20*scores.mean()
            + 0.15*min(1.0, math.log1p(component['flow_path_count'])/math.log(11.0))
            + 0.10*min(1.0, math.log1p(component['suspicious_value'])/math.log(1_000_001.0))
            + 0.10*min(1.0, density*5.0), 0.0, 1.0))
        component['network_bin'] = bin_of(component['network_score'], c.account_red, c.account_yellow)

    # ---- outputs ----------------------------------------------------------
    def summary(self, cid):
        component = self.components[cid]
        return {'component_id': cid, 'network_score': component['network_score'],
                'network_bin': component['network_bin'], 'account_count': len(component['nodes']),
                'red_accounts': component['red_accounts'], 'yellow_accounts': component['yellow_accounts'],
                'edge_count': len(component['edges']), 'transaction_count': component['transaction_count'],
                'suspicious_value': component['suspicious_value'], 'flow_path_count': component['flow_path_count'],
                'risk_score_max': component['risk_score_max'], 'risk_score_mean': component['risk_score_mean']}

    def detail(self, cid):
        """Render-capped view of one network: highest-risk nodes, edges among them, paths among them."""
        c, component = self.config, self.components[cid]
        ranked = sorted(component['nodes'], key=lambda n: -self.nodes[n]['score'])[:c.render_nodes]
        shown = set(ranked)
        nodes = [{'id': n, 'score': self.nodes[n]['score'], 'bin': self.nodes[n]['bin'],
                  'in_warmup': self.nodes[n]['in_warmup']} for n in ranked]
        edges = []
        for s, t in component['edges']:
            if s in shown and t in shown:
                e = self.edges[(s, t)]
                edges.append({'source': s, 'target': t, 'transaction_count': e['transaction_count'],
                              'amount_paid_total': e['amount_paid_total'],
                              'max_transaction_score': e['max_transaction_score'],
                              'bin': bin_of(e['max_transaction_score'], c.transaction_red, c.transaction_yellow),
                              'edge_risk_score': math.sqrt(self.nodes[s]['score']*self.nodes[t]['score']),
                              'in_warmup': e['in_warmup']})
        edges = heapq.nlargest(c.render_edges, edges, key=lambda e: e['edge_risk_score'])
        paths = [p for p in component['paths'] if p['source_account_id'] in shown
                 and p['intermediary_account_id'] in shown and p['target_account_id'] in shown]
        paths = heapq.nlargest(c.render_paths, paths, key=lambda p: (p['path_risk_score'], -p['gap_hours']))
        return {'summary': self.summary(cid), 'nodes': nodes, 'edges': edges, 'paths': paths,
                'hidden_nodes': len(component['nodes']) - len(nodes)}

    def _delta(self, batch_id, touched, merges, warm):
        c = self.config
        ranked = heapq.nlargest(c.top_k, self.components, key=lambda cid: (self.components[cid]['network_score'], cid))
        previous = set(self.top)
        self.top = ranked
        entered = [cid for cid in ranked if cid not in previous]
        changed = [cid for cid in ranked if cid in touched or cid in entered]
        self.frame += 1
        return {'frame': self.frame, 'batch_id': int(batch_id), 'stats': self.stats(warm),
                'top': [self.summary(cid) for cid in ranked],
                'entered': entered, 'exited': [cid for cid in previous if cid not in set(ranked)],
                'merges': merges, 'details': {cid: self.detail(cid) for cid in changed}}

    def stats(self, in_warmup=False):
        return {**self.totals, 'nodes': len(self.nodes), 'edges': len(self.edges),
                'networks': len(self.components), 'flow_paths': len(self.paths), 'in_warmup': in_warmup}

    def write_delta(self, delta):
        """Per-frame change file for the stream, plus the current top-k state for late joiners."""
        deltas = self.output_dir/'deltas'
        deltas.mkdir(parents=True, exist_ok=True)
        text = json.dumps(delta, allow_nan=False)
        atomic_write_text(deltas/f'{delta["frame"]:08d}.json', text)
        state = {**delta, 'details': {cid: self.detail(cid) for cid in self.top}}
        atomic_write_text(self.output_dir/'top.json', json.dumps(state, allow_nan=False))

    def write_snapshot(self, directory: Path):
        """Full graph in the offline network_postprocess schema (readable by network/server.py --data-dir)."""
        directory.mkdir(parents=True, exist_ok=True)
        cid = {n: self._cid(n) for n in self.nodes}
        nodes = pd.DataFrame([{'node_id': n, 'risk_score': v['score'], 'alert_bin': v['bin'], 'component_id': cid[n]}
                              for n, v in self.nodes.items()],
                             columns=['node_id', 'risk_score', 'alert_bin', 'component_id'])
        edges = pd.DataFrame([{'source_account_id': s, 'target_account_id': t,
                               'transaction_count': e['transaction_count'],
                               'first_timestamp_unix': e['first_timestamp_unix'],
                               'last_timestamp_unix': e['last_timestamp_unix'],
                               'amount_paid_total': e['amount_paid_total'],
                               'amount_received_total': e['amount_received_total'],
                               'source_risk_score': self.nodes[s]['score'], 'target_risk_score': self.nodes[t]['score'],
                               'edge_risk_score': math.sqrt(self.nodes[s]['score']*self.nodes[t]['score']),
                               'component_id': cid[s]} for (s, t), e in self.edges.items()])
        paths = pd.DataFrame(self.paths)
        if len(paths):
            paths['component_id'] = paths.intermediary_account_id.map(cid)
        networks = pd.DataFrame([self.summary(k) for k in self.components])
        if len(networks):
            networks = networks.sort_values('network_score', ascending=False, ignore_index=True)
        for name, frame in (('network_nodes', nodes), ('network_edges', edges),
                            ('flow_paths', paths), ('networks', networks)):
            frame.to_parquet(directory/f'{name}.parquet', index=False)
