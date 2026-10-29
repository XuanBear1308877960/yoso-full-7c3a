import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from fullmodel import FullModel
from prepare import BASE
from prepare import CAPTURE
from prepare import HERE
from prepare import save

def load(path, device):
    with np.load(path, allow_pickle=False) as z:
        d = {k: torch.as_tensor(z[k].copy(), device=device) for k in z.files if z[k].dtype.kind not in 'USO'}
    return d

def source_rows(manifest=None):
    manifest = manifest or os.environ.get('FULL_A_DATA_MANIFEST', 'data_complete.json')
    meta = json.loads((HERE / manifest).read_text())
    segments = json.loads((CAPTURE / 'segments.json').read_text())['source_train']
    split = {s['trajectory_id']: i % 5 == 0 for i, s in enumerate(segments)}
    labels, rows = ({}, [])
    for r in meta['rows']:
        if r['cohort'] != 'source_train':
            continue
        phone = r['device']
        if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Only permitted source references may enter training')
        if phone not in labels:
            labels[phone] = json.loads((BASE / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
        with np.load(r['path'], allow_pickle=False) as z:
            ids = z['query_ids'].copy()
        valid = np.array([str(oid) in labels[phone] for oid in ids])
        gt = np.array([labels[phone][str(oid)]['gt_xy_m'] if ok else [0.0, 0.0] for oid, ok in zip(ids, valid)], np.float32)
        if valid.any():
            rows.append(dict(r, heldout=split[r['trajectory_id']], truth=gt, valid=valid))
    assert len({r['trajectory_id'] for r in rows if not r['heldout']}) == 43
    assert len({r['trajectory_id'] for r in rows if r['heldout']}) == 11
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=400)
    p.add_argument('--batch-folds', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--training-name', default='training')
    p.add_argument('--resume')
    p.add_argument('--resume-state')
    p.add_argument('--geometry-aware', action='store_true')
    p.add_argument('--data-manifest', default=os.environ.get('FULL_A_DATA_MANIFEST', 'data_complete.json'))
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    np.random.seed(20261020)
    random.seed(20261020)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = FullModel().to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device)['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    rows = source_rows(a.data_manifest)
    training = [r for r in rows if not r['heldout']]
    validation = [r for r in rows if r['heldout']]
    cache = OrderedDict()

    def positioning_scores(value, d):
        if not a.geometry_aware:
            return value
        centres = d['posterior_path_xy'][:, d['query_scan']]
        variance = centres.var(0, unbiased=False) + 1.0
        displacement = d['grid_xy'][None, None] - centres[:, :, None]
        d2 = (displacement.square() / variance[None, :, None]).sum(-1)
        logmass = -2 * torch.log1p(d2 / 3.0) + d['geometry_log_weight'].log_softmax(0)[:, None, None]
        geometry = torch.logsumexp(logmass, 0)
        return value + geometry

    def item(r):
        key = r['path']
        if key not in cache:
            d = load(key, device)
            d['truth'] = torch.tensor(r['truth'], device=device)
            d['valid'] = torch.tensor(r['valid'], device=device)
            cache[key] = d
            if len(cache) > 24:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def metrics(values):
        x = np.asarray(values)
        return dict(n=len(x), median_m=float(np.median(x)), p90_m=float(np.percentile(x, 90)), p95_m=float(np.percentile(x, 95)), mean_m=float(x.mean()))

    @torch.no_grad()
    def evaluate():
        model.eval()
        errors, sigmas, outliers, corrections, probability_shifts = ([], [], [], [], [])
        for r in validation:
            d = item(r)
            value, diag = model(d)
            xy = d['grid_xy'][positioning_scores(value, d).argmax(-1)]
            errors.extend(torch.linalg.vector_norm(xy[d['valid']] - d['truth'][d['valid']], dim=-1).cpu().tolist())
            sigmas.extend(diag['sigma'].flatten()[::17].cpu().tolist())
            outliers.extend(diag['outlier'].flatten()[::17].cpu().tolist())
            corrections.extend(diag['correction'].flatten()[::37].cpu().tolist())
            probability_shifts.append(float((diag['path_probability'] - d['geometry_log_weight'].softmax(-1)).abs().sum()))
        return dict(position=metrics(errors), noise_sigma_quantiles=np.quantile(sigmas, [0.1, 0.5, 0.9]).tolist(), outlier_quantiles=np.quantile(outliers, [0.1, 0.5, 0.9]).tolist(), M3_correction_quantiles=np.quantile(corrections, [0.1, 0.5, 0.9]).tolist(), M2_mean_probability_shift=float(np.mean(probability_shifts)))
    out = HERE / a.training_name
    out.mkdir(exist_ok=True)
    start, best, trace = (time.monotonic(), float('inf'), [])
    first_step = 0
    if a.resume_state and Path(a.resume_state).exists():
        restored = torch.load(a.resume_state, map_location=device)
        model.load_state_dict(restored['state_dict'])
        optimizer.load_state_dict(restored['optimizer'])
        random.setstate(restored['python_rng'])
        np.random.set_state(restored['numpy_rng'])
        torch.set_rng_state(restored['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in restored['cuda_rng']])
        first_step, best, trace = (restored['next_step'], restored['best'], restored['trace'])
        print(json.dumps(dict(stage='RESUME_EXACT_TRAINING_STATE', next_step=first_step)), flush=True)
    for step in range(first_step, a.updates + 1):
        if step % 50 == 0 or step == a.updates:
            val = evaluate()
            row = dict(step=step, validation=val, seconds=time.monotonic() - start)
            trace.append(row)
            print(json.dumps(row), flush=True)
            checkpoint = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, selection=row, source_only=True, target_positions_used=False, M1='Frozen source-pretrained OPPO field and1732parameter response basis', M2='Learned reliability, Gaussian-mixture calibration,2cavity-message rounds', M3='Active spatial discrimination; joint source gradients reach M2')
            checkpoint['geometry_aware_positioning_loss'] = a.geometry_aware
            torch.save(checkpoint, out / ('step_%04d.pt' % step))
            if step > 0 and val['position']['median_m'] < best:
                best = val['position']['median_m']
                torch.save(checkpoint, out / 'trained.pt')
            save(out / 'progress.json', dict(status='TRAINING' if step < a.updates else 'TRAINING_FINISHED', trace=trace))
        if step == a.updates:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = random.choices(training, k=a.batch_folds)
        losses = []
        for r in batch:
            d = item(r)
            value, diag = model(d)
            structured_value = positioning_scores(value, d)
            distance2 = (d['grid_xy'][None] - d['truth'][:, None]).square().sum(-1)
            soft = (-distance2 / (0.35 ** 2 * 2)).softmax(-1)
            valid = d['valid']
            ce = -(soft * structured_value.log_softmax(-1)).sum(-1)[valid].mean()
            expected_error = (structured_value.softmax(-1) * (distance2 + 0.01).sqrt()).sum(-1)[valid].mean()
            global_ce = -(soft * value.log_softmax(-1)).sum(-1)[valid].mean() if a.geometry_aware else 0.0
            gaussian = (soft * diag['gaussian']).sum(-1)[valid].mean()
            reg = diag['correction'].square().mean()
            noise_reg = 0.001 * (diag['sigma'].square().mean() + 10 * diag['outlier'].mean())
            loss = ce + 0.1 * expected_error + 0.1 * global_ce - 0.01 * gaussian + 0.002 * reg + noise_reg
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Full loss; no target results committed')
            (loss / a.batch_folds).backward()
            losses.append(float(loss.detach()))
        if step == 0:
            grad = {name: sum((float(p.grad.square().sum()) for p in group.parameters() if p.grad is not None)) ** 0.5 for name, group in [('M2', model.calibration), ('M3', model.positioning)]}
            if min(grad.values()) <= 0:
                raise ValueError('A required trainable module has no source positioning gradient')
            save(out / 'joint_gradient_receipt.json', dict(source_position_loss_gradient_norms=grad, target_labels_used=False))
            print(json.dumps(dict(stage='ACTUAL_JOINT_M2_M3_GRADIENT', values=grad)), flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 10 == 0:
            state = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), next_step=step + 1, best=best, trace=trace, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            temporary = out / 'resume.pending'
            torch.save(state, temporary)
            temporary.replace(out / 'resume.pt')
        if (step + 1) % 10 == 0:
            print(json.dumps(dict(step=step + 1, loss=float(np.mean(losses)), seconds=time.monotonic() - start)), flush=True)
    saved = torch.load(out / 'trained.pt', map_location='cpu')
    save(out / 'completion.json', dict(status='COMPLETE_SOURCE_JOINT_M2_M3_TRAINING', selected=saved['selection'], parameters=sum((p.numel() for p in model.parameters())), source_train=43, source_heldout=11, geometry_aware_positioning_loss=a.geometry_aware, target_labels_used=False, trace=trace, device=device, seconds=time.monotonic() - start))
if __name__ == '__main__':
    main()
