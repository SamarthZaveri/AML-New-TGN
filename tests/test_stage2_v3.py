"""Synthetic correctness tests; no project data/config is needed or modified.
Run: python -m unittest discover -s tests -p test_stage2_v3.py -v
"""
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch

# Isolated test-only config; never replaces config.py on disk.
fixture_config = types.ModuleType('config')
for k, v in dict(EVENT_STREAM_FILE=Path('unused'), SPARK_STATIC_DIR=Path('unused'),
                 CHAIN_SIGNALS_FILE=Path('unused'), ACCOUNT_FEATURE_COLUMNS=['f1', 'f2'],
                 BANK_FEATURE_COLUMNS=['f1'], ENTITY_FEATURE_COLUMNS=['f1'], BATCH_SIZE=8,
                 DEVICE='cpu', HIDDEN_DIM=8, TIME_DIM=4, NUM_ATTENTION_HEADS=2, DROPOUT=.2).items():
    setattr(fixture_config, k, v)
sys.modules['config'] = fixture_config
from stage2.model import MuleDetectorTGN
from stage2.tgat_attention import NeighborSampler, TemporalMultiHeadAttention
from stage2.tgn_memory import elapsed
from stage2.conservation_loss import ConservationLoss
from stage2.dataset import TemporalEventDataset, _sha
from stage2.preprocessing import fit_transform, transform
from stage2.checkpointing import atomic_save, load_trusted, validate_checkpoint, FORMAT_VERSION
from stage2.train import parser, seed_run, replay_temporal, metric_pair, TrainingPaused
from stage2.evaluate import find_optimal_threshold, compute_metrics

torch.set_num_threads(1)


def small_model():
    m = MuleDetectorTGN(6, 2, 2, 3, 2, 2, 2, 8, 4, 2, .0, True, 3)
    m.initialize(torch.randn(6, 3), torch.randn(2, 2), torch.randn(2, 2),
                 torch.tensor([0, 0, 1, 1, 0, 1]), torch.tensor([0, 1, 0, 1, 0, 1]))
    return m


def small_batch():
    return dict(source_ids=torch.tensor([0, 1, 2, 3]), dest_ids=torch.tensor([1, 2, 3, 4]),
                edge_features=torch.randn(4, 2), timestamps=torch.tensor([0., .1, .2, .3], dtype=torch.float64), do_relay=True)


def fixture(root, split_mode='account'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = 80
    ids = [f'a{i}' for i in range(n)]
    pd.DataFrame({'account_id': ids, 'account_label': np.arange(n)%2}).to_parquet(root/'account_labels.parquet')
    pd.DataFrame({'account_id': ids, 'f1': np.arange(n)%2, 'f2': np.linspace(1, 100, n)}).to_parquet(root/'node_account_features.parquet')
    pd.DataFrame({'bank_id': ['b0', 'b1'], 'f1': [1., 2.]}).to_parquet(root/'node_bank_features.parquet')
    pd.DataFrame({'entity_id': ['e0', 'e1'], 'f1': [2., 1.]}).to_parquet(root/'node_entity_features.parquet')
    pd.DataFrame({'account_id': ids, 'bank_id': ['b'+str(i%2) for i in range(n)]}).to_parquet(root/'edge_account_to_bank.parquet')
    pd.DataFrame({'account_id': ids, 'entity_id': ['e'+str(i%2) for i in range(n)]}).to_parquet(root/'edge_entity_to_account.parquet')
    count = 160
    pd.DataFrame({'sender_account_id': [ids[i%n] for i in range(count)],
                  'receiver_account_id': [ids[(i+3)%n] for i in range(count)],
                  'timestamp_unix': 1700000000 + np.arange(count)*.1,
                  'amount_paid': np.arange(count)+1., 'amount_received': np.arange(count)+1.}).iloc[::-1].to_parquet(root/'events.parquet')
    pd.DataFrame({'account_id': ids[:4], 'in_chain': [1, 0, 1, 0],
                  'conservation_score': [.8, np.nan, .6, np.nan]}).to_parquet(root/'chains.parquet')
    return TemporalEventDataset(root/'events.parquet', root, root/'chains.parquet',
                                batch_size=32, split_mode=split_mode)


class ChronologicalEvaluationTests(unittest.TestCase):
    def test_event_boundaries_and_future_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = fixture(Path(tmp), split_mode='chronological')
            self.assertEqual(d.event_windows, {'train': (0, 112), 'val': (112, 136), 'test': (136, 160)})
            for left, right in (('train', 'val'), ('val', 'test')):
                self.assertLess(d.get_batch(d.num_batches(left)-1, 'cpu', left)['timestamps'][-1],
                                d.get_batch(0, 'cpu', right)['timestamps'][0])
            m = MuleDetectorTGN(d.num_accounts, d.num_banks, d.num_entities,
                                d.account_features.shape[1], d.bank_features.shape[1],
                                d.entity_features.shape[1], 2, 8, 4, 2, 0., True, 3)
            c = ConservationLoss(8, d.num_accounts)
            stop = types.SimpleNamespace(requested=False)
            logits1, active1 = replay_temporal(m, c, d, 'cpu', 1, stop, 'val')
            d.event_edge_features[136:] += 500.
            logits2, active2 = replay_temporal(m, c, d, 'cpu', 1, stop, 'val')
            torch.testing.assert_close(logits1['val'], logits2['val'], rtol=0, atol=0)
            torch.testing.assert_close(active1['val'], active2['val'])
            prior = d.val_mask.clone()
            metric_pair(logits2['val'], d, 'val', 'cpu', .5, active2['val'])
            torch.testing.assert_close(prior, d.val_mask)

    def test_five_epoch_run_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = fixture(root/'data', split_mode='chronological')
            args = parser().parse_args(['--device', 'cpu', '--num-epochs', '5', '--patience', '5',
                                       '--split-mode', 'chronological', '--hidden-dim', '8',
                                       '--time-dim', '4', '--num-heads', '2', '--max-neighbors', '3',
                                       '--checkpoint-every-batches', '0', '--checkpoint-every-seconds', '0'])
            args.run_dir = root/'run'
            result = seed_run(args, d, 42, types.SimpleNamespace(requested=False))
            self.assertEqual(len(json.loads((args.run_dir/'seed_42/history.json').read_text())), 5)
            self.assertGreaterEqual(result['best_epoch'], 1)
            ck = load_trusted(args.run_dir/'seed_42/best.pt')
            self.assertEqual(ck['settings']['split_mode'], 'chronological')
            self.assertEqual(ck['dataset_metadata']['event_windows'], d.event_windows)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_all_backbone_gradient_paths_and_multiple_steps(self):
        m = small_model()
        opt = torch.optim.Adam(m.parameters(), lr=.01)
        for step in range(3):
            batch = small_batch()
            batch['timestamps'] += step
            out = m(**batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out['mule_logits'],
                                                                        (out['node_ids']%2).float())
            opt.zero_grad()
            loss.backward()
            for name in ['msg_account_txn_in', 'msg_account_txn_out', 'account_proj', 'bank_proj', 'entity_proj',
                         'account_gru', 'bank_gru', 'entity_gru', 'relay_gru', 'msg_account_to_bank',
                         'msg_account_to_entity', 'msg_bank_to_account', 'msg_entity_to_account']:
                grads = [p.grad for p in getattr(m.tgn_memory, name).parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), name)
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
            opt.step()

    def test_readout_matches_forward_at_same_cutoff(self):
        m = small_model().eval()
        with torch.no_grad():
            batch = small_batch()
            out = m(**batch)
            pred = m.predict_accounts(out['node_ids'], float(batch['timestamps'].max()), chunk=2)
        torch.testing.assert_close(out['mule_logits'], pred['mule_logits'], rtol=1e-5, atol=1e-6)
        self.assertEqual(m.predict_accounts(torch.tensor([], dtype=torch.long), .3)['mule_logits'].numel(), 0)

    def test_first_seen_and_prior_counts(self):
        torch.testing.assert_close(elapsed(torch.tensor([500.]), torch.tensor([0.]), torch.tensor([False])), torch.zeros(1))
        m = small_model()
        batch = small_batch()
        out = m(**batch)
        self.assertEqual(int(out['prior_event_counts'].sum()), 0)
        self.assertEqual(int(out['event_counts'].sum()), 8)
        last = m.tgn_memory.account_last_update.clone()
        m.tgn_memory.relay_batch(out['node_ids'], out['embeddings'], m.account_bank, m.account_entity, 20.)
        torch.testing.assert_close(last, m.tgn_memory.account_last_update)

    def test_neighbor_overflow_both_directions_and_restore(self):
        sampler = NeighborSampler(10, 3)
        source = torch.tensor([0, 2, 0, 4, 0, 6, 0])
        dest = torch.tensor([1, 0, 3, 0, 5, 0, 7])
        times = torch.arange(7, dtype=torch.float64)
        sampler.update(source, dest, times)
        ids = sampler.neighbor_ids[0]
        order = sampler.neighbor_times[0].argsort()
        self.assertEqual(ids[order].tolist(), [5, 6, 7])
        clone = NeighborSampler(10, 3)
        clone.load_state_dict(sampler.state_dict())
        clone.update(torch.tensor([8]), torch.tensor([0]), torch.tensor([8.], dtype=torch.float64))
        order = clone.neighbor_times[0].argsort()
        self.assertEqual(clone.neighbor_ids[0][order].tolist(), [6, 7, 8])

    def test_masked_attention_finite_gradients(self):
        for dtype in [torch.float32, torch.float16]:
            a = TemporalMultiHeadAttention(8, 4, 2, 0).to(dtype)
            target = torch.randn(2, 8, dtype=dtype, requires_grad=True)
            nb = torch.zeros(2, 3, 8, dtype=dtype, requires_grad=True)
            y = a(target, nb, torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
            y.square().sum().backward()
            self.assertTrue(torch.isfinite(target.grad).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters()))

    def test_conservation_decays_both_sides(self):
        c = ConservationLoss(8, 3)
        c.update_flows(torch.tensor([0]), torch.tensor([1]), torch.tensor([100.]), torch.tensor([100.]), torch.tensor([0.]))
        c.update_flows(torch.tensor([1]), torch.tensor([2]), torch.tensor([20.]), torch.tensor([20.]), torch.tensor([72.]))
        self.assertAlmostEqual(c.in_flow[1].item(), 100/math.e, places=4)
        self.assertAlmostEqual(c.out_flow[1].item(), 20., places=4)
        # Contributions inside one wide batch are also decayed to its end.
        c.reset_state()
        c.update_flows(torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([100., 20.]),
                       torch.tensor([100., 20.]), torch.tensor([0., 72.]))
        self.assertAlmostEqual(c.out_flow[0].item(), 100/math.e+20, places=4)

    def test_training_can_overfit_small_fixture(self):
        m = small_model()
        batch = small_batch()
        opt = torch.optim.Adam(m.parameters(), lr=.005)
        losses = []
        for i in range(200):
            m.reset_state()
            out = m(**batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out['mule_logits'], (out['node_ids']%2).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0]*.15)


class DataTests(unittest.TestCase):
    def test_spark_parquet_directories_for_all_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = fixture(root)
            for path in list(root.glob('*.parquet')):
                frame = pd.read_parquet(path)
                path.unlink()
                path.mkdir()
                middle = max(1, len(frame)//2)
                frame.iloc[:middle].to_parquet(path/'part-00000.parquet', index=False)
                frame.iloc[middle:].to_parquet(path/'part-00001.parquet', index=False)
                (path/'_SUCCESS').touch()
            loaded = TemporalEventDataset(root/'events.parquet', root, root/'chains.parquet', 32)
            torch.testing.assert_close(loaded.account_labels, original.account_labels)
            torch.testing.assert_close(loaded.event_timestamps, original.event_timestamps)
            torch.testing.assert_close(loaded.event_edge_features, original.event_edge_features)
            digest = _sha(root/'account_labels.parquet')
            part = root/'account_labels.parquet/part-00000.parquet'
            frame = pd.read_parquet(part)
            frame.loc[0, 'account_label'] = 1-int(frame.loc[0, 'account_label'])
            frame.to_parquet(part, index=False)
            self.assertNotEqual(_sha(root/'account_labels.parquet'), digest)

    def test_directory_hash_is_location_independent_and_sensitive(self):
        import shutil
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/'a', root/'b'
            (a/'nested').mkdir(parents=True)
            (a/'nested/part.parquet').write_bytes(b'example')
            shutil.copytree(a, b)
            self.assertEqual(_sha(a), _sha(b))
            self.assertEqual(_sha(a/'nested/part.parquet'), hashlib.sha256(b'example').hexdigest())
            before = _sha(b)
            (b/'nested/part.parquet').rename(b/'nested/renamed.parquet')
            self.assertNotEqual(before, _sha(b))
            (root/'empty').mkdir()
            with self.assertRaisesRegex(ValueError, 'Empty dataset'):
                _sha(root/'empty')

    def test_training_only_normalization_and_roundtrip(self):
        x = np.array([[1.], [2.], [10000.]])
        out, state = fit_transform(x, np.array([True, True, False]))
        self.assertEqual(state['signed_log'], [False])
        self.assertEqual(state['mean'], [1.5])
        np.testing.assert_array_equal(out, transform(x, state))

    def test_dataset_sorted_precise_and_unknown_chain_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = fixture(tmp)
            self.assertTrue((d.event_timestamps[1:] > d.event_timestamps[:-1]).all())
            self.assertEqual(d.event_timestamps.dtype, torch.float64)
            self.assertEqual(int(d.chain_known_mask.sum()), 4)
            self.assertFalse(d.chain_known_mask[5])
            again = TemporalEventDataset(Path(tmp)/'events.parquet', Path(tmp), Path(tmp)/'chains.parquet', 32)
            self.assertEqual(d.fingerprint, again.fingerprint)
            label_path = Path(tmp)/'account_labels.parquet'
            labels = pd.read_parquet(label_path)
            pd.concat([labels, labels.iloc[:1]]).to_parquet(label_path)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                TemporalEventDataset(Path(tmp)/'events.parquet', Path(tmp), Path(tmp)/'chains.parquet', 32)

    def test_reject_multiple_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            path = Path(tmp)/'edge_entity_to_account.parquet'
            frame = pd.read_parquet(path)
            pd.concat([frame, pd.DataFrame({'account_id': ['a0'], 'entity_id': ['e1']})]).to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'Multiple'):
                TemporalEventDataset(Path(tmp)/'events.parquet', Path(tmp), Path(tmp)/'chains.parquet', 32)

    def test_validation_threshold_and_metrics(self):
        logits = torch.tensor([-3., -2., -1., -.5])
        labels = torch.tensor([0, 0, 1, 1])
        threshold = find_optimal_threshold(logits, labels)
        self.assertEqual(compute_metrics(logits, labels, threshold)['f1'], 1.)
        self.assertEqual(compute_metrics(logits, labels)['f1'], 0.)


class RecoveryTests(unittest.TestCase):
    def test_atomic_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.pt'
            atomic_save({'value': 1}, path)
            original = path.read_bytes()
            def fail(payload, stream):
                stream.write(b'partial')
                raise OSError('simulated disk failure')
            with mock.patch('stage2.checkpointing.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_save({'value': 2}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name('latest.pt.tmp').exists())

    def test_replay_interruption_keeps_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = fixture(root/'data')
            args = parser().parse_args(['--device', 'cpu', '--num-epochs', '1', '--hidden-dim', '8',
                                       '--time-dim', '4', '--num-heads', '2', '--max-neighbors', '3'])
            args.run_dir = root/'run'
            stop = types.SimpleNamespace(requested=False)
            original = d.get_batch
            calls = 0
            def interrupted(index, device):
                nonlocal calls
                calls += 1
                value = original(index, device)
                if calls == d.num_batches()+1:
                    stop.requested = True
                return value
            d.get_batch = interrupted
            with self.assertRaises(TrainingPaused):
                seed_run(args, d, 42, stop)
            saved = load_trusted(args.run_dir/'seed_42/latest.pt')
            self.assertEqual(saved['progress']['epoch'], 0)
            self.assertEqual(saved['progress']['next_batch'], d.num_batches())
            d.get_batch = original
            args.resume = True
            result = seed_run(args, d, 42, types.SimpleNamespace(requested=False))
            self.assertEqual(result['best_epoch'], 1)

    def test_checkpoint_guards(self):
        ck = {'format_version': FORMAT_VERSION, 'dataset_fingerprint': 'a', 'settings': {'lr': .1}}
        with self.assertRaises(ValueError):
            validate_checkpoint(ck, 'b', {'lr': .1})
        with self.assertRaises(ValueError):
            validate_checkpoint(ck, 'a', {'lr': .2})

    def test_interrupted_training_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = fixture(root/'data')
            args = parser().parse_args(['--device', 'cpu', '--num-epochs', '2', '--hidden-dim', '8',
                                       '--time-dim', '4', '--num-heads', '2', '--max-neighbors', '3',
                                       '--lambda-conservation', '.01', '--lambda-chain-aux', '.01',
                                       '--checkpoint-every-batches', '0', '--checkpoint-every-seconds', '0'])
            args.run_dir = root/'full'
            seed_run(args, d, 42, types.SimpleNamespace(requested=False))
            expected = load_trusted(args.run_dir/'seed_42/latest.pt')
            args.run_dir = root/'resumed'
            stop = types.SimpleNamespace(requested=False)
            original = d.get_batch
            def stopping_batch(index, device):
                batch = original(index, device)
                if index == 1:
                    stop.requested = True
                return batch
            d.get_batch = stopping_batch
            with self.assertRaises(TrainingPaused):
                seed_run(args, d, 42, stop)
            saved = load_trusted(args.run_dir/'seed_42/latest.pt')
            self.assertEqual(saved['progress']['next_batch'], 2)
            self.assertGreater(saved['model']['tgn_memory.account_event_count'].sum(), 0)
            self.assertTrue((saved['model']['neighbor_sampler.neighbor_ids'] >= 0).any())
            d.get_batch = original
            args.resume = True
            seed_run(args, d, 42, types.SimpleNamespace(requested=False))
            actual = load_trusted(args.run_dir/'seed_42/latest.pt')
            for key in expected['model']:
                torch.testing.assert_close(expected['model'][key], actual['model'][key], rtol=0, atol=0, msg=key)
            for key in expected['cons']:
                torch.testing.assert_close(expected['cons'][key], actual['cons'][key], rtol=0, atol=0, msg=key)
            for pid, state in expected['optimizer']['state'].items():
                for name, value in state.items():
                    other = actual['optimizer']['state'][pid][name]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, other, rtol=0, atol=0)
                    else:
                        self.assertEqual(value, other)
            self.assertEqual(expected['progress']['global_step'], actual['progress']['global_step'])
            self.assertEqual(expected['progress']['best_ap'], actual['progress']['best_ap'])
            torch.testing.assert_close(expected['rng']['torch'], actual['rng']['torch'], rtol=0, atol=0)
            self.assertFalse(list(args.run_dir.rglob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
