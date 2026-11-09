"""Atomic trusted-local checkpoints, full RNG state, and deferred SIGINT handling."""
import os
import random
import signal
from pathlib import Path
import numpy as np
import torch

FORMAT_VERSION = 3


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device topology differs from the resumable checkpoint')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with open(temporary, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_trusted(path, device='cpu'):
    # torch checkpoints can execute code; only load checkpoints you generated/trust.
    return torch.load(path, map_location=device, weights_only=False)


def validate_checkpoint(ck, fingerprint, settings):
    if ck.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint version; start a new run with v3')
    if ck.get('dataset_fingerprint') != fingerprint:
        raise ValueError('Dataset, ID mapping, preprocessing, split, or batch size changed; cannot resume')
    if ck.get('settings') != settings:
        changed = [k for k in set(ck.get('settings', {})) | set(settings)
                   if ck.get('settings', {}).get(k) != settings.get(k)]
        raise ValueError(f'Incompatible resume settings: {changed}')


class StopRequest:
    """Ctrl+C sets a flag; training saves only after a complete optimizer step.
    During replay the pre-replay training checkpoint remains authoritative.
    """
    def __init__(self):
        self.requested = False
        self.previous = None

    def _signal(self, signum, frame):
        if not self.requested:
            print('\nStop requested; finishing the current batch/save safely...', flush=True)
        self.requested = True

    def __enter__(self):
        self.previous = signal.signal(signal.SIGINT, self._signal)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self.previous)
