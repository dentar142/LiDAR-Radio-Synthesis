"""Small contracts shared by the bounded H20 experiment (no labels loaded here)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

PHYSICAL = ('BASE', 'MAT', 'CLASS', 'GRID', 'MAT_CLASS', 'MAT_GRID',
            'CLASS_MAT', 'GRID_MAT', 'PAR_MC', 'PAR_MG', 'PAR_MCG')
RAW = PHYSICAL + ('NW_FULLP',)
SIGNAL = tuple(x + '_GP' for x in PHYSICAL) + ('GP_N', 'NW_N',
    'PAR_GP_MAT', 'PAR_GP_MAT_CLASS', 'PAR_GP_MAT_GRID', 'PAR_GP_MCG')
BANDS = ('n41', 'n79')
MODES = ('infill', 'spatial30', 'spatial60')
BUDGETS = (30, 100, 300, 1000, 2500)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite JSON value')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise RuntimeError('unsafe artifact path')
    if not path.resolve().is_relative_to(root):
        raise RuntimeError('artifact escapes root')
    return path


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('changed frozen input/runtime: ' + str(path))


def runtime_hashes(release):
    release = Path(release).resolve()
    return {str(p): sha(p) for folder in ('src', 'vendor/wedt_full_v1/src')
            for p in sorted((release / folder).rglob('*.py'))}


def load_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest['root'] != str(root) or manifest['schema'] != 'h20-v1':
        raise RuntimeError('wrong H20 root or schema')
    expected = {f'f{f}_{b}_{m}' for f in (3,4) for b in BANDS for m in MODES}
    ids = [s['id'] for s in manifest['supports']]
    signal_ids = [s['id'] for s in manifest['signals']]
    expected_signals = {f'{key}_b{n}_r{r:02d}' for key in expected for n in BUDGETS for r in range(4)}
    if len(ids) != 12 or set(ids) != expected or len(signal_ids) != 240 or set(signal_ids) != expected_signals:
        raise RuntimeError('incomplete/duplicate factorial manifest')
    if tuple(manifest['physical_methods']) != PHYSICAL or tuple(manifest['signal_methods']) != SIGNAL:
        raise RuntimeError('changed method matrix')
    verify_hashes(manifest['runtime_sha256'])
    verify_hashes(manifest['asset_sha256'])
    verify_hashes({manifest['protocol_path']: manifest['protocol_sha256']})
    return manifest


def load_support(root, key):
    import pandas as pd
    manifest = load_manifest(root)
    spec = next(s for s in manifest['supports'] if s['id'] == key)
    directory = inside(root, spec['input_dir'])
    for relative, expected in spec['files'].items():
        if sha(inside(directory, relative)) != expected:
            raise RuntimeError('support artifact changed: ' + relative)
    train = pd.read_csv(directory / 'fit.csv', dtype={'point_id': str, 'date': str})
    query = pd.read_csv(directory / 'query.csv', dtype={'point_id': str, 'date': str})
    if 'observed_dbm' in query or not np.isfinite(train.observed_dbm).all():
        raise RuntimeError('query target leak or invalid training target')
    if set(train.point_id) & set(query.point_id):
        raise RuntimeError('ID leak')
    if set(train.position_group) & set(query.position_group):
        raise RuntimeError('position leak')
    if len(train) != spec['train_n'] or len(query) != spec['test_n']:
        raise RuntimeError('row count changed')
    return manifest, spec, train, query


def nw50(train_xy, train_y, query_xy):
    x = np.asarray(train_xy, float)
    y = np.asarray(train_y, float)
    q = np.asarray(query_xy, float)
    if len(x) < 1 or y.shape != (len(x),) or not all(
            np.isfinite(a).all() for a in (x, y, q)):
        raise ValueError('invalid NW inputs')
    d, ids = cKDTree(x).query(q, k=min(128, len(x)), workers=1)
    if d.ndim == 1:
        d, ids = d[:, None], ids[:, None]
    logits = -0.5 * (d / 50.0) ** 2
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (weights * y[ids]).sum(axis=1) / weights.sum(axis=1)


def physical_mean(y, train_gain, query_gain, fallback_train, fallback_query,
                  offset_train=None, offset_query=None):
    train_gain, query_gain = np.asarray(train_gain), np.asarray(query_gain)
    good = np.isfinite(train_gain)
    if not good.any() or np.isinf(train_gain).any() or np.isinf(query_gain).any():
        raise RuntimeError('invalid/no physical training paths')
    ot = np.zeros(len(y)) if offset_train is None else np.asarray(offset_train)
    oq = np.zeros(len(query_gain)) if offset_query is None else np.asarray(offset_query)
    beta = float(np.mean((np.asarray(y) - ot - train_gain)[good]))
    return (np.where(good, train_gain + beta + ot, fallback_train),
            np.where(np.isfinite(query_gain), query_gain + beta + oq, fallback_query), beta)


def completed(directory, contract, runtime, required=()):
    directory = Path(directory)
    path = directory / 'status.json'
    if not path.exists():
        return False
    status = read_json(path)
    if (status.get('status') != 'COMPLETE' or status.get('input_contract') != contract
            or status.get('runtime_sha256') != runtime):
        raise RuntimeError('refuse inconsistent completed result')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != 'status.json'}
    if set(required) != set(status['artifacts']) or actual != set(status['artifacts']):
        raise RuntimeError('missing mandatory/unrecorded artifact')
    for name, expected in status['artifacts'].items():
        if sha(inside(directory, name)) != expected:
            raise RuntimeError('completed artifact changed')
    return True


def finish(directory, contract, runtime, started, extra=None):
    directory = Path(directory)
    artifacts = {str(p.relative_to(directory)): sha(p)
                 for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name not in ('status.json',) and not p.name.endswith('.tmp')}
    write_json(directory / 'status.json', dict(status='COMPLETE', input_contract=contract,
        runtime_sha256=runtime, artifacts=artifacts,
        elapsed_seconds=time.time() - started, **(extra or {})))


def job_contract(manifest, spec, stage):
    payload = json.dumps([manifest['protocol_sha256'], manifest['runtime_sha256'],
                          spec, stage], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
