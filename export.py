import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
import torch
from fullmodel import FullModel
from prepare import BASE
from prepare import CAPTURE
from prepare import HERE
from prepare import save
from train import load

def wifi_cavity(model, s, prior, device, cohort='original_fixed20'):
    from radio_field import field
    from trajectory_factors import robust_initial_prior
    from trajectory_factors import ObservationModel
    folder = CAPTURE / cohort / s['trajectory_id']
    with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
        pre = {k: z[k].copy() for k in z.files}
    with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
        grid, ns = (z['grid_xy'].copy(), z['times_ns'].copy())
    times = (ns - s['start_ns']) / 1000000000.0
    radio = field(BASE, prior)
    initial = robust_initial_prior(pre, prior)
    prefix_clock_ns = int(ns[min(5, len(ns) - 1)])
    context = np.broadcast_to(pre['motion'][4], (len(grid), 32))
    mean, basis, gradient, basis_gradient = radio(grid, context, True)
    mu = np.exp(initial['log_weight']) @ initial['mean']
    prediction = mean + np.einsum('qad,d->qa', basis, mu)
    observation = ObservationModel(Path(prior['temporal_RSSI_query_root']) / cohort / s['trajectory_id'] / 'radio_queries.npz', pre)
    covariances, _, _ = observation.noise(np.asarray(prior['temporal_RSSI_parameters']))
    noise = np.stack([c.diagonal() for c in covariances], -1)
    mask = observation.mask.copy()
    variance = noise[:, None] + 4.0
    emission = -0.5 * (((pre['rssi'][:, None] - prediction[None]) ** 2 / variance + np.log(variance)) * mask[:, None]).sum(-1)
    distance2 = np.square(grid[:, None] - grid[None]).sum(-1)
    forward = emission[0] - 0.5 * np.square(grid - s['initial_position_xy_m']).sum(-1) / (1.5 * max(times[0], 0.1) + 0.3) ** 2
    predecessors = []
    for i in range(1, len(times)):
        transition = -0.5 * distance2 / (1.5 * max(times[i] - times[i - 1], 0.1) + 0.3) ** 2
        joint = forward[:, None] + transition
        prev = joint.argmax(0)
        predecessors.append(prev)
        forward = joint[prev, np.arange(len(grid))] + emission[i]
    endpoints = []
    for j in np.argsort(-forward):
        if all((np.linalg.norm(grid[j] - grid[old]) >= 0.75 for old in endpoints)):
            endpoints.append(int(j))
        if len(endpoints) == 4:
            break
    paths = []
    for j in endpoints:
        seq = [j]
        for back in reversed(predecessors):
            seq.append(int(back[seq[-1]]))
        paths.append(seq[::-1])
    index = np.asarray(paths)
    xy = grid[index]
    support_mean = mean[index]
    support_basis = basis[index]
    grad = gradient[index]
    predicted = support_mean + np.einsum('ktad,d->kta', support_basis, mu)
    spread = np.var(xy, axis=0) + 0.25
    projected = np.einsum('ktad,td->kta', grad ** 2, spread)
    expected = np.einsum('k,ktd->td', np.full(len(paths), 1 / len(paths)), xy)
    cells = np.floor(expected / 2).astype(int)
    groups = sorted(set(map(tuple, cells[mask.any(-1)])))
    potential = np.zeros((len(times), len(grid)), np.float32)
    continuity = np.square(xy[:, None] - xy[None]).sum(-1).mean(-1) / 8.0
    geometry = forward[endpoints] - logsumexp(forward[endpoints])
    dt = np.r_[0.0, np.diff(times)].clip(0, 30)

    def tensor(x):
        return torch.as_tensor(x, device=device)
    for region in groups:
        queried = np.all(cells == region, -1) & mask.any(-1)
        support = mask.copy()
        support[queried] = False
        for a, S in enumerate(observation.S):
            touched = S[queried & mask[:, a]].sum(0) > 0
            support[S[:, touched].sum(-1) > 0, a] = False
        budget = np.zeros_like(pre['rssi'], float)
        for a, S in enumerate(observation.S):
            use = (S * support[:, a, None]).sum(0)
            budget[:, a] = (S / np.maximum(use, 1)[None]).sum(-1) * support[:, a]
        local_region = np.full(len(times), -1, np.int64)
        other = [c for c in groups if c != region]
        for j, c in enumerate(other):
            local_region[np.all(cells == c, -1)] = min(3, j * 4 // max(len(other), 1))
        observed = np.broadcast_to(np.where(support, pre['rssi'], 0.0), predicted.shape)
        dr = np.zeros_like(observed)
        dr[:, 1:] = np.diff(observed, axis=1)
        dp = np.zeros_like(predicted)
        dp[:, 1:] = np.diff(predicted, axis=1)
        band = np.unique(radio.band, axis=0, return_inverse=True)[1]
        encoder = np.stack((observed / 50, predicted / 50, (observed - predicted) / 10, dr / 10, dp / 10, np.broadcast_to(np.log1p(noise)[None] / 5, predicted.shape), grad[..., 0] / 10, grad[..., 1] / 10, projected / 100, np.broadcast_to(band[None, None], predicted.shape), np.broadcast_to(dt[None, :, None] / 10, predicted.shape), np.broadcast_to(support[None], predicted.shape)), -1)
        select = np.where(queried)[0]
        n = len(select)
        data = dict(support_y=pre['rssi'], support_mean=support_mean, support_basis=support_basis, support_mask=budget, support_variance=noise[None] + projected, encoder=encoder, prior_mean=initial['mean'], prior_covariance=initial['covariance'], prior_log_weight=initial['log_weight'], region=local_region, geometry_log_weight=geometry, continuity_cost=continuity, query_y=pre['rssi'][select], query_mean=np.broadcast_to(mean, (n,) + mean.shape), query_basis=np.broadcast_to(basis, (n,) + basis.shape), query_gradient=np.broadcast_to(gradient, (n,) + gradient.shape), query_basis_gradient=np.broadcast_to(basis_gradient, (n,) + basis_gradient.shape), query_visibility=np.ones((n, len(grid), 4)), query_variance=noise[select], query_mask=mask[select], ap_embedding=radio.ap)
        data = {k: tensor(np.array(v, dtype=np.int64 if k == 'region' else np.float32, copy=True)) for k, v in data.items()}
        value, _ = model(data)
        potential[select] = value.cpu().numpy()
    potential -= logsumexp(potential, -1, keepdims=True)
    return (dict(grid_xy=grid, times_ns=ns, log_probability=potential, prefix_cutoff_ns=np.asarray(prefix_clock_ns, np.int64), initial_mean=initial['mean'], initial_covariance=initial['covariance']), dict(CS_after_prefix_used=False, IMU_or_CS_derived_path_used=False, target_labels_used=False, conditional_device_mapping_uses='WiFi-only4Viterbi hypotheses and two cross-region cavity-message rounds', query_support_raw_overlap=0, independently_validates='All CS events after immutable initial-prior cutoff', prefix_cutoff_ns=prefix_clock_ns, regions=len(groups)))

def main():
    from data_utils import deny_label_reads
    torch.set_num_threads(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    training = os.environ.get('FULL_A_TRAINING_DIR', 'training')
    potential_name = os.environ.get('FULL_A_POTENTIAL_DIR', 'potentials')
    saved = torch.load(HERE / training / 'trained.pt', map_location='cpu')
    model = FullModel().to(device).eval()
    model.load_state_dict(saved['state_dict'])
    meta = json.loads((HERE / os.environ.get('FULL_A_DATA_MANIFEST', 'data_complete.json')).read_text())
    cohort = os.environ.get('FULL_A_COHORT', 'original_fixed20')
    segments = json.loads((CAPTURE / 'segments.json').read_text())[cohort]
    if cohort == 'source_train':
        segments = [s for i, s in enumerate(segments) if i % 5 == 0]
    prior = json.loads((BASE / 'runs/direction_v80/trajectory_incumbent_M2_selected_v1/prediction_complete.json').read_text())['source_prior']
    started = time.monotonic()
    reports = []
    with deny_label_reads() as audit, torch.no_grad():
        for s in segments:
            tid = s['trajectory_id']
            dest = HERE / potential_name / cohort / tid
            if (dest / 'completion.json').exists():
                reports.append(json.loads((dest / 'completion.json').read_text()))
                continue
            with np.load(CAPTURE / cohort / tid / 'trajectory_factors.npz', allow_pickle=False) as z:
                ns, grid = (z['times_ns'].copy(), z['grid_xy'].copy())
            potential = np.zeros((len(ns), len(grid)), np.float32)
            gaussian_potential = np.zeros_like(potential)
            correction_potential = np.zeros_like(potential)
            predicted_rssi_grid = np.zeros((len(ns), len(grid), 4), np.float32)
            query_valid_AP = np.zeros((len(ns), 4), bool)
            weights = np.zeros(len(ns), np.float32)
            assigned = np.zeros(len(ns), bool)
            mu = np.zeros((len(ns), 6), np.float32)
            cov = np.broadcast_to(np.eye(6), (len(ns), 6, 6)).copy()
            sigma, outlier, correction, shifts = ([], [], [], [])
            rows = [r for r in meta['rows'] if r['cohort'] == cohort and r['trajectory_id'] == tid]
            from trajectory_factors import ObservationModel
            with np.load(CAPTURE / cohort / tid / 'PRE_forecasts.npz', allow_pickle=False) as z:
                original_pre = {k: z[k].copy() for k in z.files}
            observation = ObservationModel(Path(prior['temporal_RSSI_query_root']) / cohort / tid / 'radio_queries.npz', original_pre)
            unique = np.zeros_like(original_pre['rssi'], float)
            for a, S in enumerate(observation.S):
                usage = (S * observation.mask[:, a, None]).sum(0)
                unique[:, a] = (S / np.maximum(usage, 1.0)[None]).sum(-1) * observation.mask[:, a]
            scan_budget = np.minimum(1.0, unique.sum(-1) / observation.mask.sum(-1).clip(1))
            for r in rows:
                d = load(r['path'], device)
                value, diag = model(d)
                scans = d['query_scan'].cpu().numpy()
                if assigned[scans].any():
                    raise ValueError('A query WiFi scan was counted twice')
                assigned[scans] = True
                potential[scans] = value.cpu().numpy()
                gaussian_potential[scans] = diag['gaussian'].cpu().numpy()
                correction_potential[scans] = diag['correction'].cpu().numpy()
                predicted_rssi_grid[scans] = diag['predicted_RSSI'].cpu().numpy()
                query_valid_AP[scans] = d['query_mask'].cpu().numpy()
                weights[scans] = scan_budget[scans]
                w = diag['posterior_log_weight'].exp()
                m = torch.einsum('c,cd->d', w, diag['posterior_mean'])
                delta = diag['posterior_mean'] - m
                P = torch.einsum('c,cde->de', w, diag['posterior_covariance'] + delta[:, :, None] * delta[:, None, :])
                mu[scans] = m.cpu().numpy()
                cov[scans] = P.cpu().numpy()
                sigma.extend(diag['sigma'].flatten()[::31].cpu().tolist())
                outlier.extend(diag['outlier'].flatten()[::31].cpu().tolist())
                correction.extend(diag['correction'].flatten()[::37].cpu().tolist())
                shifts.append(float((diag['path_probability'] - d['geometry_log_weight'].softmax(-1)).abs().sum()))
            spatial_head_required = True
            if spatial_head_required and (not correction or max(abs(np.asarray(correction))) < 1e-07):
                raise ValueError('An inactive M3 is not a completed Full')
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'head.npz', potential=potential, weight=weights, grid_xy=grid, times_ns=ns, device_mean=mu, device_covariance=cov, gaussian=gaussian_potential, correction=correction_potential, predicted_rssi_grid=predicted_rssi_grid, query_valid_AP=query_valid_AP)
            cavity, feedback = wifi_cavity(model, s, prior, device, cohort)
            np.savez_compressed(dest / 'wifi_CS_cavity.npz', **cavity)
            report = dict(trajectory_id=tid, target_labels_read=False, context_query_exclusion=True, message_rounds=2, M2_path_probability_mean_shift=float(np.mean(shifts)), M3_active=spatial_head_required, correction_quantiles=np.quantile(correction, [0.1, 0.5, 0.9]).tolist(), sigma_quantiles=np.quantile(sigma, [0.1, 0.5, 0.9]).tolist(), outlier_quantiles=np.quantile(outlier, [0.1, 0.5, 0.9]).tolist(), CS_feedback=feedback)
            save(dest / 'completion.json', report)
            reports.append(report)
            print(json.dumps(dict(stage='EXPORT_FULL_AND_CS_CAVITY', completed=len(reports), total=len(segments), seconds=time.monotonic() - started)), flush=True)
        if audit['blocked_label_attempts']:
            raise ValueError('Full export attempted true-position reads')
    marker = potential_name + ('_source_train' if cohort == 'source_train' else '') + '_export_complete.json'
    save(HERE / marker, dict(status='COMPLETE_ACTIVE_M2_M3_AND_INDEPENDENT_CS_CAVITIES', reports=reports, checkpoint_selection=saved['selection'], labels_read=False, seconds=time.monotonic() - started))
if __name__ == '__main__':
    main()
