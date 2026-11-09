"""Correctness-first TGN training: batch recovery, frozen replay, validation threshold.
Run from repository root: python -m stage2.train --help
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
import config
from stage2.dataset import TemporalEventDataset
from stage2.model import MuleDetectorTGN
from stage2.conservation_loss import ConservationLoss, FocalLoss, ChainAuxiliaryLoss
from stage2.evaluate import compute_metrics, find_optimal_threshold, print_metrics, aggregate_seed_metrics
from stage2.checkpointing import (FORMAT_VERSION, StopRequest, atomic_save, load_trusted,
                                  rng_state, restore_rng, validate_checkpoint)


class TrainingPaused(Exception):
    pass


def initialize(model, cons, dataset, device):
    model.initialize(dataset.account_features.to(device), dataset.bank_features.to(device),
                     dataset.entity_features.to(device), dataset.account_bank_map.to(device),
                     dataset.account_entity_map.to(device))
    cons.reset_state()


def replay(model, cons, dataset, device, relay_interval, stop):
    """Frozen weights, batch-end last-touch logits; cold accounts scored at stream end."""
    model.eval()
    initialize(model, cons, dataset, device)
    logits = torch.zeros(dataset.num_accounts, device=device)
    touched = torch.zeros(dataset.num_accounts, dtype=torch.bool, device=device)
    with torch.no_grad():
        for b in range(dataset.num_batches()):
            if stop.requested:
                raise TrainingPaused('Replay interrupted; resume restarts replay from the saved training boundary')
            batch = dataset.get_batch(b, device)
            out = model(**{k: batch[k] for k in ('source_ids', 'dest_ids', 'edge_features', 'timestamps')},
                        do_relay=b % relay_interval == 0)
            if not torch.isfinite(out['mule_logits']).all():
                raise FloatingPointError(f'Non-finite replay logits at batch {b}; checkpoint left intact')
            logits[out['node_ids']] = out['mule_logits']
            touched[out['node_ids']] = True
            if (b+1) % 500 == 0:
                print(f'    Replay {b+1}/{dataset.num_batches()}', flush=True)
        cold = (~touched).nonzero(as_tuple=True)[0]
        if len(cold):
            logits[cold] = model.predict_accounts(cold, float(dataset.event_timestamps[-1]))['mule_logits']
    return logits


@torch.no_grad()
def replay_temporal(model, cons, dataset, device, relay_interval, stop, target):
    """Replay only through target window. Score accounts active in that window.

    Return target and training-prefix logits/eligibility without using future events.
    """
    if target not in ('val', 'test') or dataset.split_mode != 'chronological':
        raise ValueError('Expected a chronological val or test replay')
    model.eval()
    initialize(model, cons, dataset, device)
    logits = {name: torch.zeros(dataset.num_accounts, device=device) for name in ('train', target)}
    active = {name: torch.zeros(dataset.num_accounts, dtype=torch.bool, device=device)
              for name in ('train', target)}
    steps = 0
    windows = ('train', 'val') if target == 'val' else ('train', 'val', 'test')
    for window in windows:
        for batch_idx in range(dataset.num_batches(window)):
            if stop.requested:
                raise TrainingPaused('Temporal replay interrupted; recovery checkpoint preserves training state')
            batch = dataset.get_batch(batch_idx, device, split=window)
            out = model(**{k: batch[k] for k in ('source_ids', 'dest_ids', 'edge_features', 'timestamps')},
                        do_relay=steps % relay_interval == 0)
            if not torch.isfinite(out['mule_logits']).all():
                raise FloatingPointError(f'Non-finite {window} replay logits at batch {batch_idx}')
            if window in logits:
                ids = out['node_ids']
                logits[window][ids] = out['mule_logits']
                active[window][ids] = True
            steps += 1
        print(f'    Temporal replay through {window}: {steps:,} batches', flush=True)
    return logits, active


def metric_pair(logits, dataset, split, device, threshold, eligible=None):
    mask = getattr(dataset, split+'_mask').to(device).clone()
    if eligible is not None:
        mask &= eligible
    labels = dataset.account_labels.to(device)[mask]
    return {'at_0_5': compute_metrics(logits[mask], labels),
            'at_validation_threshold': compute_metrics(logits[mask], labels, threshold)}


def seed_run(args, dataset, seed, stop):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model_args = dict(num_account_nodes=dataset.num_accounts, num_bank_nodes=dataset.num_banks,
                      num_entity_nodes=dataset.num_entities, account_feat_dim=dataset.account_features.size(1),
                      bank_feat_dim=dataset.bank_features.size(1), entity_feat_dim=dataset.entity_features.size(1),
                      edge_feat_dim=dataset.edge_feat_dim, memory_dim=args.hidden_dim, time_dim=args.time_dim,
                      num_attention_heads=args.num_heads, dropout=args.dropout,
                      enable_relay=not args.disable_relay, max_neighbors=args.max_neighbors)
    model = MuleDetectorTGN(**model_args).to(device)
    cons = ConservationLoss(args.hidden_dim, dataset.num_accounts, gap_weight=args.conservation_gap_weight).to(device)
    primary_fn = FocalLoss(args.focal_gamma, dataset.compute_pos_weight(args.max_pos_weight))
    chain_fn = ChainAuxiliaryLoss()
    # Avoid retaining optimizer moments for objectives disabled in this run.
    for p in cons.parameters():
        p.requires_grad_(args.lambda_conservation > 0)
    for head in (model.chain_membership_head, model.chain_score_head):
        for p in head.parameters():
            p.requires_grad_(args.lambda_chain_aux > 0)
    params = [p for p in list(model.parameters()) + list(cons.parameters()) if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    settings = {k: getattr(args, k) for k in ('lr', 'weight_decay', 'lambda_conservation', 'lambda_chain_aux',
                'focal_gamma', 'max_pos_weight', 'relay_interval', 'min_history', 'conservation_gap_weight', 'split_mode')}
    settings.update(seed=seed, model=model_args, device_type=device.type,
                    torch_version=str(torch.__version__), deterministic=args.deterministic)
    run_dir = args.run_dir / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = run_dir/'latest.pt', run_dir/'best.pt'
    progress = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'best_ap': -1., 'best_epoch': 0,
                'no_improve': 0, 'totals': [0., 0., 0., 0.], 'history': []}
    initialize(model, cons, dataset, device)
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f'No resumable checkpoint at {latest}')
        ck = load_trusted(latest, 'cpu')
        validate_checkpoint(ck, dataset.fingerprint, settings)
        model.load_state_dict(ck['model'])
        cons.load_state_dict(ck['cons'])
        optim.load_state_dict(ck['optimizer'])
        progress = ck['progress']
        restore_rng(ck['rng'])
        del ck
        print(f"Resumed seed {seed}: epoch {progress['epoch']+1}, next batch {progress['next_batch']+1}")
    elif latest.exists() or best_path.exists():
        raise FileExistsError(f'{run_dir} already contains a run; use --resume or a new --run-dir')

    def save_latest():
        if any(not torch.isfinite(p).all() for p in params):
            raise FloatingPointError('Non-finite parameters; refusing to replace the recovery checkpoint')
        atomic_save({'format_version': FORMAT_VERSION, 'kind': 'resume', 'dataset_fingerprint': dataset.fingerprint,
                     'dataset_metadata': dataset.metadata, 'settings': settings, 'progress': progress,
                     'model': model.state_dict(), 'cons': cons.state_dict(),
                     'optimizer': optim.state_dict(), 'rng': rng_state()}, latest)
        print(f"    Saved recovery checkpoint: epoch {progress['epoch']+1}, next batch {progress['next_batch']+1}", flush=True)

    if not args.resume:
        save_latest()
    labels = dataset.account_labels.to(device)
    train = dataset.train_mask.to(device)
    known = (dataset.chain_known_mask & dataset.train_mask).to(device)
    if not dataset.chain_has_binary_labels:
        known.zero_()
    scored = (dataset.chain_score_mask & dataset.train_mask).to(device)
    chain_y, chain_score = dataset.chain_membership.to(device), dataset.chain_scores.to(device)
    last_save = time.monotonic()
    train_window = 'train' if dataset.split_mode == 'chronological' else 'all'
    nb = dataset.num_batches(train_window)
    while progress['epoch'] < args.num_epochs and progress['no_improve'] < args.patience:
        epoch_start = time.monotonic()
        if progress['next_batch'] == 0:
            initialize(model, cons, dataset, device)
        model.train()
        for b in range(progress['next_batch'], nb):
            if stop.requested:
                save_latest()
                raise TrainingPaused(f'Saved at completed batch boundary: {latest}')
            batch = (dataset.get_batch(b, device, split='train') if dataset.split_mode == 'chronological'
                     else dataset.get_batch(b, device))
            out = model(**{k: batch[k] for k in ('source_ids', 'dest_ids', 'edge_features', 'timestamps')},
                        do_relay=b % args.relay_interval == 0)
            if not torch.isfinite(out['embeddings']).all():
                raise FloatingPointError(f'Non-finite embeddings at batch {b+1}; restore {latest}')
            ids = out['node_ids']
            selected = train[ids] & (out['prior_event_counts'] >= args.min_history)
            primary = primary_fn(out['mule_logits'][selected], labels[ids[selected]])
            flow = (cons(out['embeddings'], ids, batch['source_ids'], batch['dest_ids'],
                         batch['amounts_paid'], batch['amounts_received'], batch['timestamps'], train[ids])
                    if args.lambda_conservation > 0 else primary.new_zeros(()))
            chain = (chain_fn(out['chain_membership_logits'], out['chain_score_preds'], chain_y[ids],
                              chain_score[ids], known[ids], scored[ids])
                     if args.lambda_chain_aux > 0 and dataset.has_chain_signals else primary.new_zeros(()))
            loss = primary + args.lambda_conservation*flow + args.lambda_chain_aux*chain
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {progress["epoch"]+1} batch {b+1}; restore {latest}')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Finite loss can still yield non-finite gradients. Never step on those.
            nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optim.step()
            progress['next_batch'] = b+1
            progress['global_step'] += 1
            for j, value in enumerate((loss, primary, flow, chain)):
                progress['totals'][j] += float(value.detach())
            if (b+1) % 100 == 0:
                print(f"    Batch {b+1}/{nb}: loss={loss.item():.4f} primary={primary.item():.4f} "
                      f'flow={flow.item():.4f} chain={chain.item():.4f}', flush=True)
            due = ((args.checkpoint_every_batches > 0 and (b+1) % args.checkpoint_every_batches == 0) or
                   (args.checkpoint_every_seconds > 0 and time.monotonic()-last_save >= args.checkpoint_every_seconds))
            if due or stop.requested:
                save_latest()
                last_save = time.monotonic()
            if stop.requested:
                raise TrainingPaused(f'Saved at completed batch boundary: {latest}')
        # This is the authoritative recovery boundary if evaluation is interrupted.
        save_latest()
        if dataset.split_mode == 'chronological':
            replay_logits, active = replay_temporal(model, cons, dataset, device, args.relay_interval, stop, 'val')
            vm = dataset.val_mask.to(device) & active['val']
            if int(labels[vm].sum()) < 1 or int((labels[vm] == 0).sum()) < 1:
                raise ValueError('Validation window has fewer than one active account from a class; adjust protocol')
            threshold = find_optimal_threshold(replay_logits['val'][vm], labels[vm])
            val = metric_pair(replay_logits['val'], dataset, 'val', device, threshold, active['val'])
            training = metric_pair(replay_logits['train'], dataset, 'train', device, threshold, active['train'])
            print(f'    Eligible validation accounts: {int(vm.sum()):,}, positives: {int(labels[vm].sum()):,}')
        else:
            logits = replay(model, cons, dataset, device, args.relay_interval, stop)
            vm = dataset.val_mask.to(device)
            threshold = find_optimal_threshold(logits[vm], labels[vm])
            val = metric_pair(logits, dataset, 'val', device, threshold)
            training = metric_pair(logits, dataset, 'train', device, threshold)
        epoch = progress['epoch']+1
        print_metrics(val['at_0_5'], f'Epoch {epoch} validation @0.5')
        print_metrics(val['at_validation_threshold'], f'Epoch {epoch} validation tuned')
        print_metrics(training['at_validation_threshold'], f'Epoch {epoch} train')
        ap = val['at_0_5']['auprc']
        if ap > progress['best_ap']:
            progress.update(best_ap=ap, best_epoch=epoch, no_improve=0)
            # Compact learned weights only; replay reconstructs state for evaluation.
            atomic_save({'format_version': FORMAT_VERSION, 'kind': 'inference', 'dataset_fingerprint': dataset.fingerprint,
                         'dataset_metadata': dataset.metadata, 'settings': settings, 'epoch': epoch,
                         'weights': {k: p.detach() for k, p in model.named_parameters()},
                         'threshold': threshold, 'val_metrics': val}, best_path)
        else:
            progress['no_improve'] += 1
        progress['history'].append({'epoch': epoch, 'losses': [v/nb for v in progress['totals']],
                                    'validation': val, 'train': training, 'seconds_this_session': time.monotonic()-epoch_start})
        progress.update(epoch=epoch, next_batch=0, totals=[0., 0., 0., 0.])
        initialize(model, cons, dataset, device)
        save_latest()
        with open(run_dir/'history.json', 'w') as stream:
            json.dump(progress['history'], stream, indent=2)
    if stop.requested:
        raise TrainingPaused(f'Training boundary saved: {latest}')
    if not best_path.exists():
        raise RuntimeError('No validated best checkpoint; check epoch limit and validation data')
    ck = load_trusted(best_path, 'cpu')
    validate_checkpoint(ck, dataset.fingerprint, settings)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ck['weights'][name])
    threshold = ck['threshold']
    del ck
    if dataset.split_mode == 'chronological':
        replay_logits, active = replay_temporal(model, cons, dataset, device, args.relay_interval, stop, 'test')
        test = metric_pair(replay_logits['test'], dataset, 'test', device, threshold, active['test'])
        print(f"    Eligible test accounts: {int((dataset.test_mask.to(device) & active['test']).sum()):,}")
    else:
        logits = replay(model, cons, dataset, device, args.relay_interval, stop)
        test = metric_pair(logits, dataset, 'test', device, threshold)
    print_metrics(test['at_validation_threshold'], f'Seed {seed} TEST (frozen validation threshold)')
    result = {'seed': seed, 'best_epoch': progress['best_epoch'], 'best_val_ap': progress['best_ap'],
              'test': test, 'dataset_fingerprint': dataset.fingerprint}
    with open(run_dir/'results.json', 'w') as stream:
        json.dump(result, stream, indent=2)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    def default(name, fallback):
        return getattr(config, name, fallback)
    p.add_argument('--num-seeds', type=int, default=1)
    p.add_argument('--num-epochs', type=int, default=default('NUM_EPOCHS', 10))
    p.add_argument('--batch-size', type=int, default=default('BATCH_SIZE', 10000))
    p.add_argument('--split-mode', choices=['account', 'chronological'], default='account',
                   help='Use chronological to train only on early events; account keeps the legacy protocol')
    p.add_argument('--lr', type=float, default=default('LEARNING_RATE', 1e-4))
    p.add_argument('--weight-decay', type=float, default=default('WEIGHT_DECAY', 1e-5))
    p.add_argument('--hidden-dim', type=int, default=default('HIDDEN_DIM', 128))
    p.add_argument('--time-dim', type=int, default=default('TIME_DIM', 32))
    p.add_argument('--num-heads', type=int, default=default('NUM_ATTENTION_HEADS', 4))
    p.add_argument('--dropout', type=float, default=default('DROPOUT', .1))
    p.add_argument('--max-neighbors', type=int, default=20)
    p.add_argument('--device', default=str(default('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')))
    # Establish a primary-only baseline before enabling optional objectives.
    p.add_argument('--lambda-conservation', type=float, default=0.)
    p.add_argument('--lambda-chain-aux', type=float, default=0.)
    p.add_argument('--conservation-gap-weight', type=float, default=.1)
    p.add_argument('--focal-gamma', type=float, default=2.)
    p.add_argument('--max-pos-weight', type=float, default=50.)
    p.add_argument('--disable-relay', action='store_true')
    p.add_argument('--relay-interval', type=int, default=1)
    p.add_argument('--min-history', type=int, default=0, help='Required incident endpoint occurrences BEFORE this batch')
    p.add_argument('--patience', type=int, default=default('PATIENCE', 5))
    p.add_argument('--checkpoint-every-batches', type=int, default=300)
    p.add_argument('--checkpoint-every-seconds', type=float, default=600.)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--deterministic', action='store_true', help='Strict PyTorch deterministic kernels; unsupported CUDA operations may raise')
    p.add_argument('--events-file', type=Path, default=config.EVENT_STREAM_FILE)
    p.add_argument('--static-dir', type=Path, default=config.SPARK_STATIC_DIR)
    p.add_argument('--chain-signals-file', type=Path, default=config.CHAIN_SIGNALS_FILE)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for key in ('num_seeds', 'num_epochs', 'batch_size', 'hidden_dim', 'time_dim', 'num_heads', 'max_neighbors', 'patience', 'relay_interval'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.hidden_dim < 2 or args.hidden_dim % args.num_heads:
        p.error('hidden-dim must be >=2 and divisible by num-heads')
    if args.lr <= 0 or args.max_pos_weight <= 0 or not 0 <= args.dropout < 1:
        p.error('Invalid learning rate, class weight, or dropout')
    for key in ('min_history', 'lambda_conservation', 'lambda_chain_aux', 'focal_gamma', 'weight_decay',
                'conservation_gap_weight', 'checkpoint_every_batches', 'checkpoint_every_seconds'):
        if getattr(args, key) < 0:
            p.error(f'{key} must be nonnegative')
    if args.resume and args.run_dir is None:
        p.error('--resume requires the original --run-dir and training options')
    if args.run_dir is None:
        args.run_dir = Path(getattr(config, 'CHECKPOINT_DIR', 'checkpoints')) / ('stage2_v3_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    torch.use_deterministic_algorithms(args.deterministic)
    dataset = TemporalEventDataset(args.events_file, args.static_dir, args.chain_signals_file, args.batch_size,
                                   split_mode=args.split_mode)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Run directory: {args.run_dir.resolve()}\nSettings: {vars(args)}', flush=True)
    if args.lambda_chain_aux > 0 and not dataset.has_chain_signals:
        p.error('Chain loss enabled but chain file does not exist')
    if args.lambda_chain_aux > 0 and not dataset.chain_has_binary_labels:
        print('Chain membership disabled: no verified training negatives. Positive score regression only.')
    results = []
    with StopRequest() as stop:
        try:
            for i in range(args.num_seeds):
                results.append(seed_run(args, dataset, 42+111*i, stop))
        except TrainingPaused as exc:
            print(f'Paused safely. {exc}\nResume with the same command plus --resume.', flush=True)
            return
    summary = {'runs': results, 'aggregate': aggregate_seed_metrics([r['test']['at_validation_threshold'] for r in results])}
    with open(args.run_dir/'summary.json', 'w') as stream:
        json.dump(summary, stream, indent=2)


if __name__ == '__main__':
    main()
