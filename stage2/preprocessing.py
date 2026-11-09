"""Serializable numeric transforms; fit on training-associated observations only."""
import numpy as np


def fit_transform(x, fit_mask=None, *, force_log=False, clip=5.):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite numeric features; repair upstream data')
    fit = x if fit_mask is None else x[np.asarray(fit_mask)]
    if not len(fit):
        raise ValueError('No training observations available to fit preprocessing')
    logged = np.ones(x.shape[1], dtype=bool) if force_log else np.abs(fit).max(0) > 50
    transformed = x.copy()
    transformed[:, logged] = np.sign(x[:, logged]) * np.log1p(np.abs(x[:, logged]))
    training = transformed if fit_mask is None else transformed[np.asarray(fit_mask)]
    state = {'signed_log': logged.tolist(), 'mean': training.mean(0).tolist(),
             'std': np.maximum(training.std(0), 1e-6).tolist(), 'clip': clip}
    return transform(x, state), state


def transform(x, state):
    x = np.asarray(x, dtype=np.float64).copy()
    if not np.isfinite(x).all():
        raise ValueError('Non-finite features')
    logged = np.asarray(state['signed_log'], dtype=bool)
    x[:, logged] = np.sign(x[:, logged]) * np.log1p(np.abs(x[:, logged]))
    return np.clip((x - state['mean']) / state['std'], -state['clip'], state['clip']).astype(np.float32)
