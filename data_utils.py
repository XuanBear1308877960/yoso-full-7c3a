from pathlib import Path
import json
import os
import numpy as np
from contextlib import contextmanager
import builtins
import io
from trajectory_factors import TemporalJointPath
from trajectory_factors import feature_speeds
canonical_replay_BASE = Path(os.environ.get('YOMO_BASE', '.'))
PARENTS = [canonical_replay_BASE / 'runs/direction_v80/trajectory_incumbent_M2_selected_v1', canonical_replay_BASE / 'runs/direction_v80/motion_family_v1/complete_pool']
_FEATURES = None
_PRIORS = {}

def priors(parent):
    if parent not in _PRIORS:
        p = json.loads((PARENTS[parent] / 'prediction_complete.json').read_text())['source_prior']
        p = dict(p)
        p.pop('additional_parent_solver_root', None)
        p.pop('solver_polish_parent_root', None)
        p.update(save_all_calibration_candidates=False, path_solver_maxiter=300, cache_solver_groups=False)
        if os.environ.get('YOMO_CANONICAL_NEURAL_MOTION_PRIOR') == '1':
            p['canonical_neural_motion_prior'] = 'Frozen77187parameter RTS-shape-GRU; recomputed from this context, no full-data query feature cache'
        if os.environ.get('YOMO_CANONICAL_DIRECTION_SEED') == '1':
            p['canonical_direction_seed'] = 'Frozen source directional GRU/Viterbi, regenerated separately from sanitized context observations'
        _PRIORS[parent] = p
    return dict(_PRIORS[parent])

def evidence(cohort, tid):
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = feature_speeds(canonical_replay_BASE / 'runs/direction_v80/trajectory_geometry_readout_full_v1')
    return _FEATURES[cohort, tid]

class ContextFitPath(TemporalJointPath):
    pass

def solver_for(spec, parent):
    folder = Path(spec['folder'])
    s = spec['segment']
    cohort = spec['cohort']
    e = evidence(cohort, s['trajectory_id'])
    xy = np.array([r['xy_m'] for r in json.loads((folder / 'predictions.json').read_text())])
    p = priors(parent)
    cls = ContextFitPath
    learned = xy
    if os.environ.get('YOMO_CANONICAL_NEURAL_MOTION_PRIOR') == '1':
        path = folder.parent.parent / 'source_motion_prior' / cohort / s['trajectory_id'] / 'predictions.json'
        learned = np.array([r['xy_m'] for r in json.loads(path.read_text())])
        name = 'direction_motion_seed' if os.environ.get('YOMO_CANONICAL_DIRECTION_SEED') == '1' else 'rts_mean'
        alternative = folder.parent.parent / name / cohort / s['trajectory_id'] / 'predictions.json'
        xy = np.array([r['xy_m'] for r in json.loads(alternative.read_text())])
    solver = cls(folder, s, p, (e['speed'], e['sigma'], np.ones(len(xy), bool)), False, learned, force_context=e)
    return (solver, xy)

def select_indices(xy, count):
    xy = np.asarray(xy, dtype=float)
    if count >= len(xy):
        return np.arange(len(xy))
    if count < 2:
        raise ValueError('Full M3 needs at least two spatial candidates')
    first = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
    selected = [first]
    nearest = np.square(xy - xy[first]).sum(-1)
    while len(selected) < count:
        nearest[selected] = -1.0
        following = int(np.argmax(nearest))
        selected.append(following)
        nearest = np.minimum(nearest, np.square(xy - xy[following]).sum(-1))
    return np.asarray(sorted(selected), dtype=int)

@contextmanager
def deny_label_reads():
    opened = []
    blocked = []
    originals = [(builtins, 'open', builtins.open), (io, 'open', io.open), (os, 'open', os.open)]

    def wrap(fn):

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.fsdecode(path)
                if any((token in name for token in ('evaluation_labels', 'heldout_target_map', 'paired_target_map', 'target_gt'))):
                    blocked.append(name)
                    raise PermissionError('Evaluator labels forbidden during prediction')
                opened.append(name)
            return fn(path, *args, **kwargs)
        return guarded
    try:
        for module, key, fn in originals:
            setattr(module, key, wrap(fn))
        yield dict(opened_files=opened, blocked_label_attempts=blocked)
    finally:
        for module, key, fn in originals:
            setattr(module, key, fn)
