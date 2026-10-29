import argparse
import concurrent.futures
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
BASE = Path(os.environ.get('YOMO_BASE', '.'))
HERE = Path(__file__).resolve().parent
OLD = BASE / 'experiments/ablation_claims_20261019'
CANONICAL = Path(os.environ.get('FULL_A_CANONICAL_ROOT', str(OLD / 'canonical_v2')))
CAPTURE = BASE / 'runs/direction_v80/trajectory_factor_capture_v1'

def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2) + '\n')

def export_one(spec):
    import torch
    torch.set_num_threads(1)
    from data_utils import solver_for
    from data_utils import deny_label_reads
    t0 = time.monotonic()
    cohort, tid, fold = (spec['cohort'], spec['segment']['trajectory_id'], spec['fold'])
    out = HERE / os.environ.get('FULL_A_DATA_DIRECTORY', 'data') / cohort / tid / ('fold_%03d.npz' % fold)
    receipt = out.with_suffix('.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    ppath = CANONICAL / 'conditional_packages/parent_0' / cohort / tid / ('fold_%03d.pkl' % fold)
    with deny_label_reads() as audit:
        with ppath.open('rb') as f:
            p = pickle.load(f)
        assert p['context_replayed_without_query'] and p['raw_overlap'] == 0
        solver, _ = solver_for(spec, 0)
        model, radio = (solver.observations, solver.radio_field)
        paths = p['groups']
        means, bases, gradients, positions, geo = ([], [], [], [], [])
        for group in paths:
            q = group['context_q']
            mean, B, dm, db = model.geometry(solver.raw_spline @ q, radio, True)
            means.append(mean)
            bases.append(B)
            xy = solver.B @ q
            _, _, g, _ = radio(xy, solver.pre['motion'], True)
            gradients.append(g)
            positions.append(xy)
            geo.append(group['objective'] - solver.radio(q, 0)[0])
        mean, basis, grad, xy = map(np.asarray, (means, bases, gradients, positions))
        prior = solver.device_prior
        pw = np.exp(prior['log_weight'])
        mu0 = pw @ prior['mean']
        predicted = mean + np.einsum('ktad,d->kta', basis, mu0)
        y = solver.pre['rssi'].astype(float)
        support = p['support_mask'].copy()
        assert np.array_equal(support, model.mask)
        base_noise = np.stack([c.diagonal() for c in solver.radio_covariances], -1)
        spread = np.var(xy, axis=0) + 0.04
        pre = solver.pre
        if 'pre_covariance' in pre:
            cov = pre['pre_covariance'][..., :2, :2]
            if cov.ndim == 4:
                spread += np.einsum('tk,tkd->td', pre['weights'], np.diagonal(cov, axis1=-2, axis2=-1))
        projected = np.einsum('ktad,td->kta', grad ** 2, spread)
        variance = base_noise[None] + projected
        raw_budget = np.zeros_like(y)
        for a, S in enumerate(model.S):
            usage = (S * support[:, a, None]).sum(0)
            raw_budget[:, a] = (S / np.maximum(usage, 1.0)[None]).sum(-1) * support[:, a]
        prefix = json.loads((CANONICAL / 'prefix_capture' / cohort / tid / 'predictions.json').read_text())
        partition_xy = np.asarray([r['xy_m'] for r in prefix])
        cell = np.floor(partition_xy / 2).astype(int)
        unique = sorted(set(map(tuple, cell[support.any(-1)])), key=lambda c: np.where(np.all(cell == c, -1))[0].mean())
        region = np.full(len(y), -1, np.int64)
        for j, c in enumerate(unique):
            region[np.all(cell == c, -1)] = min(3, j * 4 // max(len(unique), 1))
        dt = np.r_[0.0, np.diff(solver.t)].clip(0, 30)
        observed = np.broadcast_to(y, predicted.shape)
        dr = np.zeros_like(observed)
        dr[:, 1:] = np.diff(observed, axis=1)
        dp = np.zeros_like(predicted)
        dp[:, 1:] = np.diff(predicted, axis=1)
        k = len(paths)
        band = np.unique(radio.band, axis=0, return_inverse=True)[1]
        encoder = np.stack((observed / 50, predicted / 50, (observed - predicted) / 10, dr / 10, dp / 10, np.broadcast_to(np.log1p(base_noise)[None] / 5, predicted.shape), grad[..., 0] / 10, grad[..., 1] / 10, projected / 100, np.broadcast_to(band[None, None], predicted.shape), np.broadcast_to(dt[None, :, None] / 10, predicted.shape), np.broadcast_to(support[None], predicted.shape)), -1)
        scan = np.unique(p['query_scan'])
        grid = np.load(CAPTURE / cohort / tid / 'trajectory_factors.npz', allow_pickle=False)['grid_xy'].copy()
        t, Q, a = (len(scan), len(grid), 4)
        qm = np.zeros((t, Q, a))
        qb = np.zeros((t, Q, a, 6))
        qg = np.zeros((t, Q, a, 2))
        qbg = np.zeros((t, Q, a, 6, 2))
        vis = np.zeros((t, Q, a))
        query_y = np.zeros((t, a))
        query_mask = np.zeros((t, a), bool)
        ixscan = {int(v): i for i, v in enumerate(scan)}
        for j, (i, ap) in enumerate(zip(p['query_scan'], p['query_AP'])):
            local = ixscan[int(i)]
            query_y[local, ap] = p['query_y'][j]
            query_mask[local, ap] = True
            atoms = np.where(p['atom_row'] == j)[0]
            for atom in atoms:
                raw = p['atom_raw'][atom]
                points = grid + p['atom_shift'][atom]
                context = np.broadcast_to(p['raw_context'][raw], (Q, 32))
                apid = np.full(Q, ap)
                m, B, g, bg = radio.ap_query(points, context, apid, True)
                w = p['atom_weight'][atom]
                qm[local, :, ap] += w * m
                qb[local, :, ap] += w * B
                qg[local, :, ap] += w * g
                qbg[local, :, ap] += w * bg
                vis[local, :, ap] += w * radio.descriptor.basis_inputs_ap(points, context, apid)[1][:, 6].clip(0, 1) if hasattr(radio, 'descriptor') else w
        continuity = np.square(xy[:, None] - xy[None]).sum(-1).mean(-1) / 8.0
        geo = -np.asarray(geo)
        geo -= logsumexp(geo)
        original_pre = np.load(CAPTURE / cohort / tid / 'PRE_forecasts.npz', allow_pickle=False)
        budget = np.minimum(1.0, np.sum(raw_budget[scan], -1) / np.maximum(query_mask.sum(-1), 1))
        budget[:] = np.minimum(1.0, query_mask.sum(-1) / 4.0)
        for ap, S in enumerate(model.S):
            qr = p['query_mask'][:, ap]
            assert not np.any((S[qr].sum(0) > 0) & (S[support[:, ap]].sum(0) > 0))
        arrays = dict(support_y=y, support_mean=mean, support_basis=basis, support_mask=raw_budget, support_variance=variance, encoder=encoder, prior_mean=prior['mean'], prior_covariance=prior['covariance'], prior_log_weight=prior['log_weight'], region=region, geometry_log_weight=geo, continuity_cost=continuity, query_y=query_y, query_mean=qm, query_basis=qb, query_gradient=qg, query_basis_gradient=qbg, query_visibility=vis, query_variance=base_noise[scan], query_mask=query_mask, ap_embedding=radio.ap, grid_xy=grid, query_scan=scan, query_ids=original_pre['observation_ids'][scan], query_times_ns=np.asarray([prefix[i]['timestamp_ns'] for i in scan], np.int64), query_budget=budget, posterior_path_xy=xy, segment_initial=np.asarray(spec['segment']['initial_position_xy_m']))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{key: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == 'f' else v for key, v in arrays.items()})
        assert not audit['blocked_label_attempts']
    result = dict(cohort=cohort, trajectory_id=tid, fold=fold, path=str(out), device=spec['segment']['device'], query_scans=len(scan), query_AP=int(query_mask.sum()), support_AP=int(support.sum()), hypotheses=k, context_excludes_query=True, raw_contributor_overlap=0, reference_positions_read=False, seconds=time.monotonic() - t0)
    save(receipt, result)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--cohorts', nargs='+', default=['source_train', 'original_fixed20'])
    a = p.parse_args()
    manifest = json.loads((CANONICAL / 'fold_manifest.json').read_text())['specs']
    specs = [s for s in manifest if s['cohort'] in a.cohorts]
    start = time.monotonic()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(export_one, spec) for spec in specs]
        for future in concurrent.futures.as_completed(jobs):
            rows.append(future.result())
            if len(rows) % 8 == 0 or len(rows) == len(specs):
                print(json.dumps(dict(stage='PREPARE_CAVITIES', completed=len(rows), total=len(specs), seconds=time.monotonic() - start)), flush=True)
    save(HERE / 'data_complete.json', dict(status='COMPLETE_LABEL_FREE_FULL_FACTORS', rows=rows, source_split='original ordinal modulo5;43 train11held', query_label_reads=0, seconds=time.monotonic() - start))
if __name__ == '__main__':
    main()
