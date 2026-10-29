def asset_path(base, value):
    value = Path(value)
    if value.exists():
        return value
    if not value.is_absolute():
        return Path(base) / value
    for folder in ('data', 'runs', 'experiments'):
        if folder in value.parts:
            candidate = Path(base).joinpath(*value.parts[value.parts.index(folder):])
            if candidate.exists():
                return candidate
    return value

import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
import math
from scipy.linalg import cholesky, solve_triangular, cho_solve
from encoders import ThreeModuleNetwork
from encoders import mlp
AVAILABLE_CPUS = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None

def evaluate_load_actor(base, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    root = asset_path(base, saved['data_root'])
    with np.load(root / 'source_map.npz', allow_pickle=False) as z:
        cache = {key: z[key] for key in z.files}
    cfg = json.loads((base / 'runs/fusion_network_v46_batched_wifi_candidates/configs/runtime_v46.json').read_text())
    net = ThreeModuleNetwork(cache, cfg['pcrt']['source_only_propagation_geometry'], saved['architecture']['variant'], saved['architecture'].get('localization_context', False), saved['architecture'].get('localization_kind') == 'precision', saved['architecture'].get('radio_likelihood') == 'gaussian_student_t_mixture', saved['architecture'].get('initial_context_scans', 1), saved['architecture'].get('motion_history_policy') == 'wifi_only_transient_cs_v2', saved['architecture'].get('calibration_geometry') == 'pre_pose_full_AP_covariance', saved['architecture'].get('innovation_memory') == 'conditional_nuisance', saved['architecture'].get('orientation_basis') == 'relative_AP_harmonics', saved['architecture'].get('mode_evidence', 'local'), saved['architecture'].get('initial_posterior', 'none'), saved['architecture'].get('ap_basis_prior', False), saved['architecture'].get('fixed_ap_basis', False), saved['architecture'].get('robust_transfer', False), saved['architecture'].get('calibrated_geometry', False), saved['architecture'].get('localization_geometry', 'none') == 'whitened_tangent_and_orthogonal', saved['architecture'].get('source_field', 'idw'), saved['architecture'].get('source_inducing', 32), saved['architecture'].get('source_lengthscale_m', 1.2), saved['architecture'].get('online_device_policy', 'cavity'), saved['architecture'].get('memory_recenter', False), saved['architecture'].get('trajectory_mode_weight', 0.0), saved['architecture'].get('cs_temporal_noise'), saved['architecture'].get('mode_spatial_evidence', 'none') == 'head_partition_ratio', saved['architecture'].get('position_integration', 'legacy'), saved['architecture'].get('region_reliability', 'none'), saved['architecture'].get('temporal_contrast', 0), saved['architecture'].get('initial_recognition', 'first'), saved['architecture'].get('field_tangent_weight', 0.0))
    net.load_state_dict(saved['state_dict'], strict=True)
    net.eval()
    from encoders import configure_filter
    cfg['filter'] = configure_filter(cfg['filter'], net.cs_temporal_noise, cfg['anchors'])
    aps = json.loads((root / 'source_map.json').read_text())['ap_ids']
    return (net, cfg, aps, root, saved)
KIND = 'causal_source_difference_residual'
LAGS = (0.35, 0.9, 1.8)
PARENT_RELATIVE = 'runs/counterfactual_v79_set/initial_window_evidence_phase/epoch_291.pt'
PARENT_SHA = '5e1fc5ddf62c983523283a6eb901ca4180297c9060ffa291740cc31664636e2b'

def temporal_radio_features(points, field, correction, rssi, mask, context):
    b, k, q, a = field['mean'].shape
    age = -context['times']
    lag = age.new_tensor(LAGS)
    time_weight = torch.exp(-0.5 * ((age[..., None] - lag) / (0.1 + 0.2 * lag)).square())
    time_weight *= ((age > 0) & (age <= 3.0))[..., None]
    weight = time_weight[:, :, None] * context['mask'][..., None]
    mass = weight.sum(1)
    weight = weight / mass.clamp_min(1e-08)[:, None]
    old_rssi = torch.einsum('bhal,bha->bal', weight, context['rssi'])
    old_source = torch.einsum('bhal,bhka->bkal', weight, context['source'])
    old_correction = torch.einsum('bhal,bhka->bkal', weight, context['correction'])
    old_position = torch.einsum('bhal,bhkz->bkalz', weight, context['position'])
    old_variance = torch.einsum('bhal,bhk->bkal', weight, context['variance'])
    elapsed = torch.einsum('bhal,bh->bal', weight, age)
    valid = (mass > 1e-08) & mask[..., None]
    shape = (b, k, q, a, len(LAGS))
    raw = (rssi[:, None, None, :, None] - old_rssi[:, None, None]).expand(shape)
    calibrated = raw - correction[..., None] + old_correction[:, :, None]
    expected = field['mean'][..., None] - old_source[:, :, None]
    residual = calibrated - expected
    displacement = points[..., None, None, :] - old_position[:, :, None]
    channels = (raw.clamp(-40, 40) / 10, calibrated.clamp(-40, 40) / 10, expected.clamp(-40, 40) / 10, residual.clamp(-40, 40) / 10, displacement[..., 0].clamp(-15, 15) / 3, displacement[..., 1].clamp(-15, 15) / 3, elapsed[:, None, None] / 2, old_variance[:, :, None].clamp_min(0).sqrt() / 4, valid[:, None, None].float())
    ap = torch.stack([v.expand(shape) for v in channels], -1)
    ap *= valid[:, None, None, ..., None]
    left, right = torch.triu_indices(a, a, 1, device=points.device)
    pair_valid = (valid[:, left] & valid[:, right])[:, None, None]
    pair = torch.stack(((calibrated[..., left, :] - calibrated[..., right, :]).clamp(-40, 40) / 10, (expected[..., left, :] - expected[..., right, :]).clamp(-40, 40) / 10, (residual[..., left, :] - residual[..., right, :]).clamp(-40, 40) / 10, pair_valid.expand(b, k, q, len(left), len(LAGS)).float()), -1)
    pair *= pair_valid[..., None]
    return (ap.reshape(b, k, q, a, 27), pair.reshape(b, k, q, len(left), 12))

class TemporalRadioResidual(nn.Module):

    def __init__(self):
        super().__init__()
        self.ap = mlp(43, 48, 24)
        self.ap_out = mlp(48, 32, 1, last_zero=True)
        self.pair = mlp(33, 32, 16)
        self.pair_out = mlp(32, 16, 1, last_zero=True)

    @staticmethod
    def pool(hidden, valid):
        weight = valid[..., None].float()
        count = weight.sum(-2).clamp_min(1.0)
        mean = (hidden * weight).sum(-2) / count
        variance = ((hidden - mean[..., None, :]).square() * weight).sum(-2) / count
        return torch.cat((mean, (variance + 1e-06).sqrt()), -1)

    def forward(self, ap, pair, mask, pair_mask):
        av = self.ap_out(self.pool(self.ap(ap), mask)).squeeze(-1)
        pv = self.pair_out(self.pool(self.pair(pair), pair_mask)).squeeze(-1)
        return 4 * ((av + pv).float() / 4).tanh()

def head_inputs(rssi, mask, prediction, variance, field, device_variance, embedding, temporal):
    error = (rssi - prediction) / variance.clamp_min(0.25).sqrt()
    scalar = torch.stack((error, variance.clamp_min(0.25).log() / 5, field['mean'] / 50, field['gradient'][..., 0] / 10, field['gradient'][..., 1] / 10, device_variance / 100, field['visibility'], mask.float()), -1)
    ap_embedding = embedding.expand(*scalar.shape[:-1], 8)
    ap = torch.cat((scalar, ap_embedding, temporal[0]), -1)
    a = prediction.shape[-1]
    left, right = torch.triu_indices(a, a, 1, device=prediction.device)
    pair_mask = mask[..., left] & mask[..., right]
    source = field['mean'][..., left] - field['mean'][..., right]
    residual = (rssi[..., left] - rssi[..., right] - (prediction[..., left] - prediction[..., right])) / 10
    gradient = field['gradient'][..., left, :] - field['gradient'][..., right, :]
    pair_scalar = torch.cat((residual[..., None], (source / 20)[..., None], gradient / 10, pair_mask[..., None].float()), -1)
    pair = torch.cat((pair_scalar, ap_embedding[..., left, :], ap_embedding[..., right, :], temporal[1]), -1)
    valid = mask & (temporal[0][..., [8, 17, 26]].sum(-1) > 0)
    pair_valid = pair_mask & (temporal[1][..., [3, 7, 11]].sum(-1) > 0)
    return (ap, pair, valid, pair_valid)

class SpatialTemporalHead(nn.Module):

    def __init__(self, static):
        super().__init__()
        self.static = static
        self.residual = TemporalRadioResidual()
        self.history_context = static.history_context
        self.temporal_contrast = 0
        self.last_residual = None

    def forward(self, rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence=None, motion=None, observation_covariance=None, observation_cholesky=None, temporal=None):
        static = self.static(rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence, motion, observation_covariance, observation_cholesky)
        if temporal is None:
            raise ValueError('Radio displacement head needs strictly causal context')
        ap, pair, valid, pair_valid = head_inputs(rssi, mask, prediction, variance, field, device_variance, ap_embedding, temporal)
        value = self.residual(ap, pair, valid, pair_valid)
        present = valid.any(-1)
        value = value * present
        self.last_residual = value.detach()
        return static + value

def upgrade(net):
    if net.temporal_contrast or getattr(net, 'direction_temporal_head', None):
        raise ValueError('Expected the selected static epoch291 architecture')
    net.localization = SpatialTemporalHead(net.localization)
    net.temporal_contrast = 3
    net.direction_temporal_head = KIND
    net.requires_grad_(False)
    net.localization.residual.requires_grad_(True)
    return net

def temporal_radio_load_actor(base, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    temporal = saved['architecture'].get('direction_temporal_head') == KIND
    speed = bool(saved['architecture'].get('direction_speed_model'))
    if not temporal and (not speed):
        return evaluate_load_actor(base, checkpoint)
    parent = Path(base) / PARENT_RELATIVE
    if saved['parent_checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('Wrong selected parent')
    net, cfg, aps, data, _ = evaluate_load_actor(base, parent)
    if temporal:
        upgrade(net)
    if speed:
        from filtering import SpeedHead
        net.motion_speed_head = SpeedHead()
    net.load_state_dict(saved['state_dict'], strict=True)
    net.eval()
    return (net, cfg, aps, data, saved)

class ResidualBasisField:

    def __init__(self, base, path):
        self.base = base
        self.descriptor = base
        while type(self.descriptor).__name__ != 'FiniteRadioField':
            if not hasattr(self.descriptor, 'base'):
                raise ValueError('Missing frozen device descriptor')
            self.descriptor = self.descriptor.base
        with np.load(Path(path), allow_pickle=False) as z:
            self.W0 = z['weight0'].copy()
            self.b0 = z['bias0'].copy()
            self.W1 = z['weight1'].copy()
            self.b1 = z['bias1'].copy()
            self.amplitude = float(z['amplitude'])
        if self.W0.shape != (32, 49) or self.W1.shape != (4, 32):
            raise ValueError('Unexpected calibration residual shape')
        self.scale = np.ones(4) if self.descriptor.strength is None else 1 - 0.85 * abs(self.descriptor.strength)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extra(self, points, context, ap_index, derivative=False):
        if derivative:
            inputs, J = descriptors(self.descriptor, points, context, np.asarray(ap_index, int))
        else:
            inputs = self.descriptor.basis_inputs_ap(points, context, ap_index)[1]
        return self.from_descriptor(inputs, J if derivative else None)

    def from_descriptor(self, inputs, J=None):
        h = inputs @ self.W0.T + self.b0
        p = expit(h)
        activation = h * p
        value = np.tanh(activation @ self.W1.T + self.b1)
        output = self.amplitude * self.scale * value
        if J is None:
            return output
        J = np.einsum('mid,oi->mod', J, self.W0)
        J *= (p * (1 + h * (1 - p)))[..., None]
        J = np.einsum('mid,oi->mod', J, self.W1)
        J *= (self.amplitude * self.scale * (1 - value ** 2))[..., None]
        return (output, J)

    def ap_query(self, points, context, ap_index, derivative=False):
        if type(self.base).__name__ == 'FiniteRadioField':
            original = self.base.ap_query(points, context, ap_index, derivative, return_descriptor=True)
            descriptor, directions = original[-2:]
            original = original[:-2]
            if derivative:
                extra, J = self.from_descriptor(descriptor, directions)
            else:
                extra = self.from_descriptor(descriptor)
        else:
            original = self.base.ap_query(points, context, ap_index, derivative)
            if derivative:
                extra, J = self.extra(points, context, ap_index, True)
            else:
                extra = self.extra(points, context, ap_index)
        B = original[1].copy()
        B[:, 2:] += extra
        if not derivative:
            return (original[0], B)
        dB = original[3].copy()
        dB[:, 2:] += J
        return (original[0], B, original[2], dB)

    def __call__(self, points, context, derivative=False):
        original = self.base(points, context, derivative)
        N = len(points)
        B = original[1].copy()
        query = np.repeat(points, 4, axis=0)
        ctx = np.repeat(context, 4, axis=0)
        ap = np.tile(np.arange(4), N)
        if derivative:
            extra, J = self.extra(query, ctx, ap, True)
        else:
            extra = self.extra(query, ctx, ap)
        B[:, :, 2:] += extra.reshape(N, 4, 4)
        if not derivative:
            return (original[0], B)
        dB = original[3].copy()
        dB[:, :, 2:] += J.reshape(N, 4, 4, 2)
        return (original[0], B, original[2], dB)

def descriptors(f, points, context, ap):
    _, features = f.basis_inputs_ap(points, context, ap)
    delta = points[:, None] - f.xy[None]
    d2 = np.square(delta).sum(-1) + f.radius2
    w = f.mask[:, ap].T / d2
    den = w.sum(-1)
    values = f.values[:, ap, :].transpose(1, 0, 2)
    component = (w[..., None] * values).sum(1) / den[:, None]
    dw = -2 * delta / d2[..., None] * w[..., None]
    dc = np.einsum('mgz,mgc->mcz', dw, values - component[:, None]) / den[:, None, None]
    offset = points - f.centres[ap]
    radius = np.square(offset).sum(-1) + 1.0
    ds = np.stack((offset / (math.log(10) * radius[:, None]), dc[:, 0] / 50, dc[:, 1] / 10, dc[:, 2] / (1 + component[:, 2, None]) / 5, dc[:, 3] * ((component[:, 3] > 0.01) & (component[:, 3] < 0.99))[:, None]), -2)
    directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (len(points), 2, 2)), ds, np.zeros((len(points), 42, 2))), -2)
    return (features, directions)
_FIELD = None
_RESIDUAL_BASIS_FIELDS = {}

class FiniteRadioField:

    def __init__(self, net):
        f = net.field
        if f.orientation_basis or f.source_model != 'idw':
            raise ValueError('This experiment uses the selected non-harmonic IDW field')

        def array(t):
            return t.detach().cpu().numpy().astype(float)
        self.xy = array(f.xy)
        self.mask = array(f.mask)
        self.values = np.stack([array(v) for v in [f.trend, f.residual, f.var, f.visibility]], -1)
        self.centres = array(f.centres_m)
        self.ap = array(f.ap_embedding.weight)
        self.band = array(f.band_embedding(f.band))
        self.layers = [(array(layer.weight), array(layer.bias)) for layer in f.basis_net if isinstance(layer, torch.nn.Linear)]
        self.strength = array(f.AP_strength()) if f.ap_basis_prior else None
        self.radius2 = 0.15 ** 2

    def __call__(self, points, context, derivative=False):
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        w = self.mask[None] / d2[..., None]
        denominator = w.sum(1)
        component = np.einsum('nga,gac->nac', w, self.values) / denominator[..., None]
        mean = component[..., 0] + component[..., 1]
        radius2 = np.square(points[:, None] - self.centres[None]).sum(-1) + 1.0
        logdist = 0.5 * np.log10(radius2)
        visibility = component[..., 3].clip(0.01, 0.99)
        scalar = np.stack((logdist, component[..., 0] / 50, component[..., 1] / 10, np.log1p(component[..., 2]) / 5, visibility), -1)
        n, a = mean.shape
        inputs = np.concatenate((np.broadcast_to(points[:, None] / 10, (n, a, 2)), scalar, np.broadcast_to(self.ap, (n, a, 8)), np.broadcast_to(self.band, (n, a, 2)), np.broadcast_to(context[:, None], (n, a, 32))), -1)
        if derivative:
            dw = -2 * offset[:, :, None, :] / d2[:, :, None, None] * w[..., None]
            dc = np.einsum('ngaz,ngac->nacz', dw, self.values[None] - component[:, None]) / denominator[..., None, None]
            dm = dc[..., 0, :] + dc[..., 1, :]
            loggradient = (points[:, None] - self.centres[None]) / (math.log(10) * radius2[..., None])
            ds = np.stack((loggradient, dc[..., 0, :] / 50, dc[..., 1, :] / 10, dc[..., 2, :] / (1 + component[..., 2, None]) / 5, dc[..., 3, :] * ((component[..., 3] > 0.01) & (component[..., 3] < 0.99))[..., None]), -2)
            directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (n, a, 2, 2)), ds, np.zeros((n, a, 42, 2))), -2)
        h = inputs
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if derivative:
                directions = (directions.swapaxes(-1, -2) @ W.T).swapaxes(-1, -2)
            if j < len(self.layers) - 1:
                p = expit(h)
                if derivative:
                    directions *= (p * (1 + h * (1 - p)))[..., None]
                h = h * p
        learned = np.tanh(h)
        if derivative:
            directions *= (1 - learned ** 2)[..., None]
        if self.strength is not None:
            scale = 1 - 0.85 * abs(self.strength)
            learned = scale * learned + self.strength * np.eye(4)
            if derivative:
                directions *= np.asarray(scale)[..., None]
        basis = np.concatenate((np.ones_like(mean)[..., None], mean[..., None] / 50, learned), -1)
        if not derivative:
            return (mean, basis)
        db = np.concatenate((np.zeros_like(dm)[..., None, :], dm[..., None, :] / 50, directions), -2)
        return (mean, basis, dm, db)

    def source_ap(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = (self.values[:, :, 0] + self.values[:, :, 1])[:, ap_index].T
        weight = mask / d2
        denominator = weight.sum(1)
        mean = (weight * values).sum(1) / denominator
        if not derivative:
            return mean
        dw = -2 * offset / d2[..., None] * weight[..., None]
        gradient = (dw * (values - mean[:, None])[..., None]).sum(1) / denominator[:, None]
        return (mean, gradient)

    def variance_ap(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, 2].T
        weight = mask / d2
        denominator = weight.sum(1)
        variance = (weight * values).sum(1) / denominator
        if not derivative:
            return variance
        dw = -2 * offset / d2[..., None] * weight[..., None]
        gradient = (dw * (values - variance[:, None])[..., None]).sum(1) / denominator[:, None]
        return (variance, gradient)

    def basis_inputs_ap(self, points, context, ap_index):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, :].transpose(1, 0, 2)
        weight = mask / d2
        denominator = weight.sum(1)
        component = np.einsum('mg,mgc->mc', weight, values) / denominator[:, None]
        mean = component[:, 0] + component[:, 1]
        distance2 = np.square(points - self.centres[ap_index]).sum(-1) + 1.0
        scalar = np.stack((0.5 * np.log10(distance2), component[:, 0] / 50, component[:, 1] / 10, np.log1p(component[:, 2]) / 5, component[:, 3].clip(0.01, 0.99)), -1)
        inputs = np.concatenate((points / 10, scalar, self.ap[ap_index], self.band[ap_index], context), -1)
        return (mean, inputs)

    def ap_query(self, points, context, ap_index, derivative=False, return_descriptor=False):
        ap_index = np.asarray(ap_index, int)
        M = len(points)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, :].transpose(1, 0, 2)
        w = mask / d2
        denominator = w.sum(1)
        component = np.einsum('mg,mgc->mc', w, values) / denominator[:, None]
        mean = component[:, 0] + component[:, 1]
        centre = self.centres[ap_index]
        radius2 = np.square(points - centre).sum(-1) + 1.0
        visibility = component[:, 3].clip(0.01, 0.99)
        scalar = np.stack((0.5 * np.log10(radius2), component[:, 0] / 50, component[:, 1] / 10, np.log1p(component[:, 2]) / 5, visibility), -1)
        inputs = np.concatenate((points / 10, scalar, self.ap[ap_index], self.band[ap_index], context), -1)
        if derivative:
            dw = -2 * offset / d2[..., None] * w[..., None]
            dc = np.einsum('mgz,mgc->mcz', dw, values - component[:, None]) / denominator[:, None, None]
            dm = dc[:, 0] + dc[:, 1]
            log_gradient = (points - centre) / (math.log(10) * radius2[:, None])
            ds = np.stack((log_gradient, dc[:, 0] / 50, dc[:, 1] / 10, dc[:, 2] / (1 + component[:, 2, None]) / 5, dc[:, 3] * ((component[:, 3] > 0.01) & (component[:, 3] < 0.99))[:, None]), -2)
            directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (M, 2, 2)), ds, np.zeros((M, 42, 2))), -2)
        input_directions = directions if derivative and return_descriptor else None
        h = inputs
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if derivative:
                directions = (directions.swapaxes(-1, -2) @ W.T).swapaxes(-1, -2)
            if j < len(self.layers) - 1:
                probability = expit(h)
                if derivative:
                    directions *= (probability * (1 + h * (1 - probability)))[..., None]
                h = h * probability
        learned = np.tanh(h)
        if derivative:
            directions *= (1 - learned ** 2)[..., None]
        if self.strength is not None:
            scale = 1 - 0.85 * abs(self.strength)
            learned = scale * learned + self.strength * np.eye(4)[ap_index]
            if derivative:
                directions *= np.asarray(scale)[..., None]
        basis = np.concatenate((np.ones((M, 1)), mean[:, None] / 50, learned), -1)
        if not derivative:
            return (mean, basis, inputs, None) if return_descriptor else (mean, basis)
        db = np.concatenate((np.zeros((M, 1, 2)), dm[:, None] / 50, directions), -2)
        result = (mean, basis, dm, db)
        return result + (inputs, input_directions) if return_descriptor else result

def _mean_field(base, prior=None):
    global _FIELD
    if _FIELD is None:
        net, _, _, _, _ = evaluate_load_actor(base, base / PARENT_RELATIVE)
        _FIELD = FiniteRadioField(net)
        if AVAILABLE_CPUS:
            os.sched_setaffinity(0, AVAILABLE_CPUS)
    return _FIELD

def field(base, prior=None):
    selected = _mean_field(base, prior)
    paths = [] if prior is None else list(prior.get('calibration_basis_residual_models', []))
    if prior is not None and prior.get('calibration_basis_residual_model'):
        paths.append(prior['calibration_basis_residual_model'])
    if paths:
        if prior.get('field_decoder_override'):
            raise ValueError('The bounded residual preserves the frozen original basis network')
        resolved = [str(asset_path(base, path).resolve()) for path in paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError('Repeated calibration residual would be applied twice')
        for path in resolved:
            key = (id(selected), path)
            if key not in _RESIDUAL_BASIS_FIELDS:
                _RESIDUAL_BASIS_FIELDS[key] = ResidualBasisField(selected, path)
            selected = _RESIDUAL_BASIS_FIELDS[key]
    return selected

def radio_blocks(t, mask, parameters):
    blocks = []
    for ap in range(4):
        ix = np.where(mask[:, ap] & (np.arange(len(t)) >= 5))[0]
        if not len(ix):
            continue
        dt = abs(t[ix, None] - t[None, ix])
        white = np.exp(parameters[ap])
        colored = np.exp(parameters[4 + ap])
        tau = np.exp(parameters[8])
        K = colored ** 2 * np.exp(-dt / tau) + np.eye(len(ix)) * (white ** 2 + 0.1 ** 2)
        L = cholesky(K, lower=True, check_finite=False)
        blocks.append((ap, ix, L))
    return blocks
