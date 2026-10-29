import itertools
import math
import numpy as np
from scipy.special import expit, logsumexp
from functools import lru_cache
import json
from pathlib import Path
from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
from scipy.special import expit, i0e, logsumexp
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg import expm
from scipy.special import expit, softmax, logsumexp
from scipy.linalg import cholesky
from scipy.special import expit
from scipy.special import i0e, i1e
from scipy.special import logsumexp
from scipy.special import softmax, logsumexp, i0e
from scipy.special import gammaln
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.linalg import block_diag, cholesky, cho_solve
from scipy.special import logsumexp, softmax
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from radio_field import field
from radio_field import radio_blocks
STATES = np.asarray(list(itertools.product([0, 1], repeat=4)), float)

def family_parameters(configuration):
    probability = np.asarray(configuration['RSSI_AP_high_noise_probability'], float)
    if probability.shape != (4,) or np.any((probability <= 0) | (probability >= 1)):
        raise ValueError('Per-AP probabilities must be strictly between zero and one')
    scale = float(configuration['RSSI_AP_high_noise_scale'])
    if not scale > 1:
        raise ValueError('The broad Gaussian family must have scale > 1')
    logprior = (STATES * np.log(probability) + (1 - STATES) * np.log1p(-probability)).sum(-1)
    inverse_variance = np.exp(-2 * math.log(scale) * STATES)
    return (logprior, inverse_variance)

def ap_posterior(z, A, prior, logdet, blocks, configuration, derivative=False):
    logprior, inverse_variance = family_parameters(configuration)
    AA = np.zeros((4, 6, 6))
    Az = np.zeros((4, 6))
    zz = np.zeros(4)
    counts = np.zeros(4)
    offset = 0
    row_ap = np.zeros(len(z), int)
    for ap, ix, _ in blocks:
        n = len(ix)
        sl = slice(offset, offset + n)
        offset += n
        row_ap[sl] = ap
        AA[ap] = A[sl].T @ A[sl]
        Az[ap] = A[sl].T @ z[sl]
        zz[ap] = z[sl] @ z[sl]
        counts[ap] = n
    if offset != len(z):
        raise ValueError('Per-AP Gaussian block lengths differ')
    precision = prior['precision'][None] + np.einsum('ba,aij->bij', inverse_variance, AA)[:, None]
    vector = prior['vector'][None] + np.einsum('ba,ai->bi', inverse_variance, Az)[:, None]
    L = np.linalg.cholesky(precision)
    covariance = np.linalg.solve(np.swapaxes(L, -1, -2), np.linalg.solve(L, np.broadcast_to(np.eye(6), precision.shape)))
    means = np.einsum('bkij,bkj->bki', covariance, vector)
    extra_logdet = 2 * math.log(configuration['RSSI_AP_high_noise_scale']) * (STATES @ counts)
    prior_quadratic = np.einsum('ki,kij,kj->k', prior['mean'], prior['precision'], prior['mean'])
    values = 0.5 * (np.einsum('ba,a->b', inverse_variance, zz)[:, None] - np.einsum('bki,bki->bk', vector, means) + prior_quadratic[None] + logdet + extra_logdet[:, None] + prior['logdet'][None] + 2 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1) + len(z) * math.log(2 * math.pi))
    score = logprior[:, None] + prior['log_weight'][None] - values
    normalizer = logsumexp(score)
    joint = np.exp(score - normalizer)
    mean = np.einsum('bk,bki->i', joint, means)
    difference = means - mean
    P = np.einsum('bk,bkij->ij', joint, covariance + difference[..., None] * difference[..., None, :])
    device_weight = joint.sum(0)
    family_weight = joint.sum(1)
    result = (-float(normalizer), mean, P, z - A @ mean, device_weight)
    if not derivative:
        return result
    gz = np.zeros_like(z)
    gA = np.zeros_like(A)
    second = covariance + means[..., None] * means[..., None, :]
    for ap in range(4):
        ix = np.where(row_ap == ap)[0]
        if not len(ix):
            continue
        mass = joint * inverse_variance[:, ap, None]
        first = np.einsum('bk,bki->i', mass, means)
        moment = np.einsum('bk,bkij->ij', mass, second)
        gz[ix] = z[ix] * mass.sum() - A[ix] @ first
        gA[ix] = A[ix] @ moment - z[ix, None] * first[None]
    return result + (dict(g_z=gz, g_A=gA, family_posterior=family_weight, AP_high_noise_posterior=family_weight @ STATES, family_values=-logsumexp(prior['log_weight'][None] - values, axis=-1)),)

def tokens(evidence, times):
    index = np.searchsorted(evidence['times'], times, side='right') - 1
    available = index >= 0
    context = np.asarray(evidence['motion'])[index.clip(0)].copy()
    context[~available] = 0.0
    return np.c_[context, available.astype(float)]

def numpy_forward(x, model):
    value = (np.asarray(x, float) - np.asarray(model['normalization_mean'])) / np.asarray(model['normalization_scale'])
    value = value.clip(-8, 8)
    weights = model['state']
    h = value
    for name, dilation in [('c0', 1), ('c1', 2)]:
        W = np.asarray(weights[name + '.weight'])
        b = np.asarray(weights[name + '.bias'])
        padded = np.pad(h, ((2 * dilation, 0), (0, 0)))
        h = sum((padded[j * dilation:j * dilation + len(h)] @ W[:, :, j].T for j in range(3))) + b
        h = h / (1 + np.exp(-h.clip(-700, 700)))
    return np.tanh(h @ np.asarray(weights['head.weight']).T + np.asarray(weights['head.bias'])) * np.asarray([0.75, 0.75, 0.75, 0.5])

@lru_cache(maxsize=4)
def loaded(model_file, features):
    return (json.loads(Path(model_file).read_text()), feature_speeds(Path(features)))

def IMU_motion_context(prior, cohort, tid, mid, speed, dt, speed_sigma=None):
    model, evidence = loaded(prior['IMU_motion_context_model'], prior['IMU_motion_context_features'])
    x = tokens(evidence[cohort, tid], mid)
    effect = numpy_forward(x, model) * float(prior.get('IMU_motion_context_strength', 1.0))
    p = np.asarray(prior['velocity_state_parameters'])
    N = len(mid)
    gain = math.exp(p[3])
    new_speed = np.asarray(speed) * np.exp(effect[:, 3])
    means = np.zeros((N, 5, 2))
    means[:, 0, 0] = new_speed
    means[:, 1, 0] = -new_speed
    means[:, 2, 1] = gain * new_speed
    means[:, 3, 1] = -gain * new_speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        extra = np.zeros_like(sigma)
        variance = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for i, gap in enumerate(0.5 * (dt[:-1] + dt[1:])):
        rate = np.exp(effect[i + 1, [0, 0, 1, 1, 2]])
        stick = np.exp(-gap * rate / persistence) if prior['velocity_state_temporal_selected'] else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    return dict(means=means, sigma=sigma, probability=prob, transition=transition, effects=effect)

def IMU_motion_energy(velocity, ctx, derivative=False):
    e = np.asarray(velocity)[:, None] - ctx['means']
    sigma = ctx['sigma']
    emission = (CONST - np.log(sigma) - 2 * np.log1p(e * e / (3 * sigma * sigma))).sum(-1)
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    N = len(e)
    alpha = np.empty_like(density)
    normalizer = np.empty(N)
    alpha[0] = ctx['probability'] * density[0]
    normalizer[0] = alpha[0].sum()
    alpha[0] /= normalizer[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ ctx['transition'][i - 1] * density[i]
        normalizer[i] = alpha[i].sum()
        alpha[i] /= normalizer[i]
    value = -float((np.log(normalizer) + maximum).sum())
    if not derivative:
        return value
    beta = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        beta[i] = ctx['transition'][i] @ (density[i + 1] * beta[i + 1]) / normalizer[i + 1]
    posterior = alpha * beta
    posterior /= posterior.sum(-1, keepdims=True)
    gradient = (posterior[:, :, None] * 4 * e / (3 * sigma * sigma + e * e)).sum(1)
    return (value, gradient, posterior)

def axis_log_density(theta, parameters, derivative=False):
    mu = np.asarray(parameters[:2])
    kappa = np.exp(parameters[2:4])
    weight = expit(parameters[4])
    logs = np.log([weight, 1 - weight]) + kappa * np.cos(2 * (theta - mu)) - math.log(math.pi) - np.log(i0e(kappa)) - kappa
    total = logsumexp(logs)
    if not derivative:
        return total
    return (total, float(np.exp(logs - total) @ (2 * kappa * np.sin(2 * (theta - mu)))))

def feature_speeds(root):
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    with np.load(root / 'features.npz', allow_pickle=False) as z:
        lengths = z['lengths'].copy()
        sensor = z['sensor'].copy()
        times = z['times'].copy()
        ids = z['ids'].copy()
    bounds = np.r_[0, np.cumsum(lengths)]
    return {(e['cohort'], e['segment']['trajectory_id']): dict(speed=sensor[bounds[i]:bounds[i + 1], 89] * 3, sigma=np.maximum(sensor[bounds[i]:bounds[i + 1], 90] * 2, 0.5), times=times[bounds[i]:bounds[i + 1]], ids=ids[bounds[i]:bounds[i + 1]], motion=sensor[bounds[i]:bounds[i + 1], 92:117]) for i, e in enumerate(manifest['entries'])}

class SensorPath:

    def __init__(self, folder, segment, prior, speed, use_speed, learned_path=None):
        self.s, self.prior, self.use_speed = (segment, prior, use_speed)
        self.predictions = json.loads((folder / 'predictions.json').read_text())
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in z.files}
        self.times_ns = f['times_ns']
        self.t = (self.times_ns - segment['start_ns']) / 1000000000.0
        end = max(float(self.t[-1]), float((cs['timestamp_ns'][-1] - segment['start_ns']) / 1000000000.0))
        self.knots = np.linspace(0.0, end, int(np.ceil(end / prior['knot_interval_s'])) + 1)
        self.spline = CubicSpline(self.knots, np.eye(len(self.knots)), axis=0, bc_type='natural')
        self.B = self.spline(self.t)
        self.V = self.spline(self.t, 1)
        mid = 0.5 * (self.knots[:-1] + self.knots[1:])
        self.A = self.spline(mid, 2)
        self.adt = np.diff(self.knots)
        self.g = f['grid_xy'].astype(float)
        self.bandwidth2 = float(prior.get('radio_kernel_bandwidth_m', 0.35)) ** 2
        scores = f['radio_loglike'].astype(float)
        self.scores = scores - scores.max(axis=(1, 2), keepdims=True)
        self.wifi_weight = np.clip(np.diff(np.r_[0.0, self.t]), 0.025, 0.5) / 0.5
        self.ct = (cs['timestamp_ns'] - segment['start_ns']) / 1000000000.0
        self.C = self.spline(self.ct)
        self.cs_temporal_response = bool(prior.get('model_CS_temporal_response', False))
        self.cs_history_metadata = None
        self.cs_past_time_moment = None
        if self.cs_temporal_response:
            if prior.get('CS_actual_history_query_root'):
                self.cs_history_metadata = load_history(prior, folder, cs['observation_ids'])
                query = self.cs_history_metadata['history_times_s']
                self.cs_response_weights = self.cs_history_metadata['response_weights']
                self.cs_past_time_moment = self.cs_history_metadata['past_time_moment']
            else:
                lags = np.asarray(prior['CS_response_lags_s'])
                anchor_weights = prior.get('CS_anchor_response_weights')
                self.cs_response_weights = np.asarray(anchor_weights)[cs['anchor_index']] if anchor_weights is not None else np.broadcast_to(np.asarray(prior['CS_response_weights']), (len(self.ct), len(lags)))
                query = self.ct[:, None] - lags[None]
            self.C_history = self.spline(query.clip(0.0, self.knots[-1]))
            self.cs_past_mass = ((query < 0) * self.cs_response_weights).sum(-1)
        self.anchor = f['anchors'][cs['anchor_index']].astype(float)
        self.cs_range = cs['range_m'].astype(float)
        self.bias = cs['bias_mean'].astype(float)
        self.variance = cs['bias_variance'].astype(float) + (f['cs_sigma'][:, None] * prior.get('CS_noise_sigma_multiplier', 1.0)) ** 2 + 0.15 ** 2 + prior.get('CS_additional_noise_floor_m', 0.0) ** 2
        if self.cs_temporal_response:
            self.variance += (self.cs_past_mass[:, None] * prior['CS_unknown_past_sigma_m']) ** 2
        self.pi = np.clip(f['cs_pi'].astype(float), 1e-05, 1 - 1e-05)
        self.anchor_index = cs['anchor_index'].astype(int)
        self.cs_weight = np.ones(len(self.ct))
        self.speed, self.speed_sigma, self.speed_valid = speed
        self.initial = np.asarray(segment['initial_position_xy_m'], float)
        self.n = len(self.knots)
        self.sensor_factors_only = False
        self.learned_path = learned_path
        if learned_path is not None:
            horizon = prior['learned_displacement_horizon_s']
            query = np.arange(max(self.t[0], 0.5), self.t[-1] - horizon + 1e-06, prior['learned_displacement_interval_s'])
            self.D = self.spline(query + horizon) - self.spline(query)
            self.learned_delta = np.stack([np.interp(query + horizon, self.t, learned_path[:, j]) - np.interp(query, self.t, learned_path[:, j]) for j in range(2)], -1)
            if prior.get('learned_acceleration_center', False):
                proposed_knots = np.stack([np.interp(self.knots, self.t, learned_path[:, j]) for j in range(2)], -1)
                self.learned_acceleration = self.A @ proposed_knots

    def unpack(self, vector):
        return (np.vstack((self.initial, vector[:2 * (self.n - 1)].reshape(-1, 2))), vector[-4:])

    def objective(self, vector, mode):
        factor_weights = self.prior.get('trajectory_factor_weights', {})
        q, db = self.unpack(vector)
        path = self.B @ q
        offset = self.g[None, :, :] - path[:, None, :]
        kernel = -0.5 * (offset ** 2).sum(-1) / self.bandwidth2
        numerator = kernel + self.scores[:, mode, :]
        ln, ld = (logsumexp(numerator, -1), logsumexp(kernel, -1))
        pn, pd = (np.exp(numerator - ln[:, None]), np.exp(kernel - ld[:, None]))
        eradio = -float(((ln - ld) * self.wifi_weight).sum()) * factor_weights.get('radio', 1.0)
        gp = (pd - pn) @ self.g / self.bandwidth2 * self.wifi_weight[:, None] * factor_weights.get('radio', 1.0)
        grad = self.B.T @ gp
        if self.cs_temporal_response:
            cpath = np.einsum('mln,nd->mld', self.C_history, q)
            history_d = cpath - self.anchor[:, None]
            history_norm = np.sqrt((history_d * history_d).sum(-1) + 1e-08)
            norm = (history_norm * self.cs_response_weights).sum(-1)
        else:
            cpath = self.C @ q
            d = cpath - self.anchor
            norm = np.sqrt((d * d).sum(-1) + 1e-08)
        residual = self.cs_range - (norm + self.bias[:, mode] + db[self.anchor_index])
        variance = np.maximum(self.variance[:, mode], 0.05 ** 2)
        normal = -0.5 * (residual ** 2 / variance + np.log(variance) + math.log(2 * math.pi)) + np.log1p(-self.pi)
        heavy = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(10.0) - 2 * np.log1p(residual ** 2 / 300) + np.log(self.pi)
        logp = np.logaddexp(normal, heavy)
        r = np.exp(normal - logp)
        dc = -(r * residual / variance + (1 - r) * 4 * residual / (300 + residual ** 2)) * self.cs_weight
        ecs = -float((logp * self.cs_weight).sum())
        if self.cs_temporal_response:
            grad += np.einsum('mln,mld,ml,m->nd', self.C_history, history_d / history_norm[..., None], np.broadcast_to(self.cs_response_weights, history_norm.shape), dc)
        else:
            grad += self.C.T @ (dc[:, None] * d / norm[:, None])
        bias_sigma = np.asarray(self.prior.get('path_CS_bias_sigma_m', 0.5))
        gb = np.bincount(self.anchor_index, weights=dc, minlength=4) + db / bias_sigma ** 2
        ebias = 0.5 * float((db ** 2 / bias_sigma ** 2).sum())
        if self.sensor_factors_only:
            return (eradio + ecs + ebias, np.r_[grad[1:].ravel(), gb])
        acc = self.A @ q
        scale2 = self.prior['acceleration_scale_m_s2'] ** 2
        if self.prior.get('learned_acceleration_center', False) and self.learned_path is not None:
            acc = acc - self.learned_acceleration
            scale2 = self.prior['learned_acceleration_sigma_m_s2'] ** 2
        a2 = (acc * acc).sum(-1)
        em = float((2.5 * np.log1p(a2 / (3 * scale2)) * self.adt).sum()) * factor_weights.get('acceleration', 1.0)
        grad += self.A.T @ (5 * acc / (3 * scale2 + a2)[:, None] * self.adt[:, None]) * factor_weights.get('acceleration', 1.0)
        vel = self.V @ q
        vnorm = np.sqrt((vel * vel).sum(-1) + 1e-06)
        excessive = np.maximum(vnorm - self.prior['source_speed_p95_m_s'], 0.0)
        if self.prior.get('learned_acceleration_center', False) and self.learned_path is not None:
            excessive = np.zeros_like(excessive)
        em += float((0.5 * (excessive / 0.5) ** 2 * self.wifi_weight).sum())
        gv = excessive / 0.5 ** 2 * self.wifi_weight
        if self.use_speed:
            residual_speed = vnorm - self.speed
            weights = self.wifi_weight * self.speed_valid
            em += float((2 * np.log1p(residual_speed ** 2 / (3 * self.speed_sigma ** 2)) * weights).sum())
            gv += 4 * residual_speed / (3 * self.speed_sigma ** 2 + residual_speed ** 2) * weights
        grad += self.V.T @ (gv[:, None] * vel / vnorm[:, None])
        if self.learned_path is not None:
            error = self.D @ q - self.learned_delta
            e2 = (error * error).sum(-1)
            sigma2 = self.prior['learned_displacement_sigma_m'] ** 2
            interval_weight = 0.5
            em += float((2.5 * np.log1p(e2 / (3 * sigma2))).sum()) * interval_weight * factor_weights.get('chord', 1.0)
            grad += self.D.T @ (5 * error / (3 * sigma2 + e2)[:, None]) * interval_weight * factor_weights.get('chord', 1.0)
        value = eradio + ecs + em + ebias
        derivative = np.r_[grad[1:].ravel(), gb]
        if not np.isfinite(value) or not np.isfinite(derivative).all():
            raise FloatingPointError('Nonfinite trajectory objective')
        return (value, derivative)

    def solve(self):
        outcomes = []
        for mode in range(8):
            xy = np.asarray([p['mode_xy_m'][mode] for p in self.predictions])
            initial = np.stack([np.interp(self.knots, self.t, xy[:, j]) for j in range(2)], -1)
            vector = np.r_[initial[1:].ravel(), np.zeros(4)]
            starts = [vector]
            if self.learned_path is not None:
                seed = np.stack([np.interp(self.knots, self.t, self.learned_path[:, j]) for j in range(2)], -1)
                seed += self.initial - seed[0]
                starts.append(np.r_[seed[1:].ravel(), np.zeros(4)])
            for vector in starts:
                result = minimize(self.objective, vector, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=350, ftol=1e-09, gtol=0.0001, maxls=30))
                outcomes.append((float(result.fun), mode, result))
        _, mode, result = min(outcomes, key=lambda r: r[0])
        q, bias = self.unpack(result.x)
        xy = self.B @ q
        predictions = []
        for p, x in zip(self.predictions, xy):
            record = dict(p)
            record.update(xy_m=x.tolist(), inference_timing='OFFLINE_FULL_SEGMENT_SENSOR_FACTORS', trajectory_mode=mode, later_position_GT_used=False)
            predictions.append(record)
        return (predictions, dict(mode=mode, iterations=int(result.nit), converged=bool(result.success), termination=str(result.message), objective=float(result.fun), additional_anchor_bias_m=bias.tolist(), branch_objectives=[dict(mode=m, value=v, iterations=int(r.nit), converged=bool(r.success)) for v, m, r in outcomes]))

class CorrelatedPath(SensorPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cs_temporal_response:
            raise ValueError('Corrector response is required')
        self.cs_weight[:] = 0.0
        self.parameters = np.asarray(self.prior['CS_temporal_innovation_parameters'])
        self.blocks = []
        with np.load(args[0] / 'trajectory_factors.npz', allow_pickle=False) as z:
            sigma = z['cs_sigma'].copy()
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            dt = abs(self.ct[ix, None] - self.ct[None, ix])
            K = covariance(self.parameters, dt, sigma[ix], self.cs_past_mass[ix], None if self.cs_past_time_moment is None else self.cs_past_time_moment[ix])
            self.blocks.append((anchor, ix, cholesky(K, lower=True, check_finite=False)))

    def objective(self, vector, mode):
        value, gradient = super().objective(vector, mode)
        q, _ = self.unpack(vector)
        points = np.einsum('mln,nd->mld', self.C_history, q)
        delta = points - self.anchor[:, None]
        distance = np.sqrt((delta * delta).sum(-1) + 1e-08)
        expected = (distance * self.cs_response_weights).sum(-1)
        residual = self.cs_range - expected - self.parameters[4:][self.anchor_index]
        derivative = np.zeros_like(residual)
        for anchor, ix, L in self.blocks:
            loss, d = innovation_nll(residual[ix], L, self.pi[ix], True)
            value += loss
            derivative[ix] = -d
        grad = np.einsum('mln,mld,ml,m->nd', self.C_history, delta / distance[..., None], self.cs_response_weights, derivative)
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

class AxialPath(CorrelatedPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_parameters = np.asarray(self.prior['axial_motion_prior_parameters'])
        if 'known_start_axis_logit_coefficients' in self.prior:
            self.axis_parameters[4] = float(start_features(self.initial) @ np.asarray(self.prior['known_start_axis_logit_coefficients']))
        self.mid = 0.5 * (self.knots[:-1] + self.knots[1:])
        self.Vmid = self.spline(self.mid, 1)
        self.mid_speed = np.interp(self.mid, self.t, self.speed).clip(0.1, 3.0)
        self.mid_speed_evidence_sigma = np.interp(self.mid, self.t, self.speed_sigma)
        if self.prior.get('duration_motion_parameters'):
            if 'directional_velocity_parameters' in self.prior or self.prior.get('IMU_motion_context_model'):
                raise ValueError('Duration5state branch requires explicit composition with other alternative motion models')
            duration_context = duration_motion_context
            self.duration_motion_context = duration_context(self.prior, self.mid_speed, np.diff(self.knots), self.mid_speed_evidence_sigma)
        if self.prior.get('IMU_motion_context_model'):
            if 'directional_velocity_parameters' in self.prior:
                raise ValueError('Conditional5state context and circular family need an explicit mixture implementation')
            imu_context = IMU_motion_context
            self.IMU_motion_context = imu_context(self.prior, args[0].parent.name, args[0].name, self.mid, self.mid_speed, np.diff(self.knots), self.mid_speed_evidence_sigma)
        self.parallel_sigma = float(self.prior['axial_parallel_sigma_m_s'])
        self.transverse_sigma = float(self.prior['axial_transverse_sigma_m_s'])
        self.stop = float(np.clip(self.prior['axial_stationary_probability'], 0.02, 0.4))
        if self.prior.get('heading_marginal_hypotheses', 0):
            if not self.prior.get('velocity_state_temporal_selected') or self.prior.get('velocity_gain_marginalization') or self.prior.get('motion_norm_speed_evidence'):
                raise ValueError('Heading marginalization currently uses the unchanged temporal five-state motion model')
            heading_context = heading_marginal_context
            self.heading_context = heading_context(self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.axis_parameters, self.mid_speed_evidence_sigma, int(self.prior['heading_marginal_hypotheses']))
        if 'directional_velocity_parameters' in self.prior:
            context = directional_velocity_context
            self.directional_velocity_context = context(self.prior['directional_velocity_parameters'], np.diff(self.knots), self.mid, self.prior.get('directional_velocity_directions', 16))
        if self.prior.get('motion_norm_speed_evidence', False):
            folder = Path(self.prior['pure_IMU_evidence_root']) / args[0].parent.name / args[0].name
            with np.load(folder / 'motion_evidence.npz', allow_pickle=False) as z:
                if z['observation_ids'].tolist() != [r['observation_id'] for r in self.predictions]:
                    raise ValueError('IMU evidence IDs differ')
                if not np.allclose(z['times_s'], self.t, atol=1e-05, rtol=0):
                    raise ValueError('IMU evidence times differ')
                self.mid_speed = np.interp(self.mid, self.t, z['speed_m_s']).clip(0.1, 3.0)
                self.mid_speed_sigma = np.interp(self.mid, self.t, z['sigma_m_s']).clip(0.5, 2.0)
                self.mid_bend = np.interp(self.mid, self.t, z['bend_probability']).clip(0, 1)

    def path_objective(self, vector, mode):
        return super().objective(vector, mode)

    def axial_objective(self, vector, mode=0):
        value, gradient = self.path_objective(vector[:-1], mode)
        q, _ = self.unpack(vector[:-1])
        theta = float(vector[-1])
        n = np.array([math.cos(theta), math.sin(theta)])
        m = np.array([-n[1], n[0]])
        velocity = self.Vmid @ q
        vp = velocity @ n
        vt = velocity @ m
        if hasattr(self, 'heading_context'):
            heading_energy = heading_marginal_energy
            motion, g, _ = heading_energy(velocity, self.heading_context)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += (motion + 0.5 * theta * theta + 0.5 * math.log(2 * math.pi)) * strength
            grad = self.Vmid.T @ g
            gradient[:2 * (self.n - 1)] += grad[1:].ravel() * strength
            return (value, np.r_[gradient, theta * strength])
        if 'velocity_state_parameters' in self.prior or 'directional_velocity_parameters' in self.prior:
            if hasattr(self, 'duration_motion_context'):
                duration_energy = IMU_motion_energy
                energy, g, _ = duration_energy(np.stack((vp, vt), -1), self.duration_motion_context, True)
            elif hasattr(self, 'IMU_motion_context'):
                imu_energy = IMU_motion_energy
                energy, g, _ = imu_energy(np.stack((vp, vt), -1), self.IMU_motion_context, True)
            elif 'directional_velocity_parameters' in self.prior:
                if self.prior.get('directional_velocity_family_marginalization', False):
                    energy, g, _ = family_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.mid_speed_evidence_sigma, self.directional_velocity_context, float(self.prior['directional_velocity_family_probability']), True)
                else:
                    directional_energy = directional_velocity_energy
                    energy, g, _ = directional_energy(np.stack((vp, vt), -1), self.mid_speed, self.directional_velocity_context, True)
                    fraction = float(self.prior.get('directional_velocity_fraction', 1.0))
                    if not 0.0 <= fraction <= 1.0:
                        raise ValueError('Invalid directional motion fraction')
                    if fraction < 1.0:
                        old_energy, old_g, _ = velocity_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], True, self.mid_speed_evidence_sigma)
                        energy = fraction * energy + (1 - fraction) * old_energy
                        g = fraction * g + (1 - fraction) * old_g
            elif self.prior.get('velocity_gain_marginalization', False):
                energy, g, _ = gain_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'], True)
            else:
                energy, g, _ = velocity_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], True, self.mid_speed_evidence_sigma)
            logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += (energy - logaxis) * strength
            gv = g[:, 0, None] * n + g[:, 1, None] * m
            grad = self.Vmid.T @ gv
            gradient[:2 * (self.n - 1)] += grad[1:].ravel() * strength
            return (value, np.r_[gradient, float((g[:, 0] * vt - g[:, 1] * vp).sum() + daxis) * strength])
        if self.prior.get('motion_norm_speed_evidence', False):
            speed = np.sqrt(np.square(velocity).sum(-1) + 1e-06)
            residual = speed - self.mid_speed
            sigma = self.mid_speed_sigma
            transverse = self.prior['motion_transverse_sigma_m_s'] * np.exp(self.prior['motion_bend_log_sigma_scale'] * self.mid_bend)
            es = 2 * np.log1p(residual * residual / (3 * sigma * sigma)) + np.log(sigma)
            gs = 4 * residual / (3 * sigma * sigma + residual * residual)
            et = 2 * np.log1p(vt * vt / (3 * transverse * transverse)) + np.log(transverse)
            gt = 4 * vt / (3 * transverse * transverse + vt * vt)
            weight = np.diff(self.knots)
            logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += float(((es + et) * weight).sum() - logaxis) * strength
            gv = (gs[:, None] * velocity / speed[:, None] + gt[:, None] * m) * weight[:, None]
            g = self.Vmid.T @ gv
            gradient[:2 * (self.n - 1)] += g[1:].ravel() * strength
            return (value, np.r_[gradient, float((-gt * vp * weight).sum() + daxis) * strength])
        means = np.stack((self.mid_speed, -self.mid_speed, np.zeros(len(self.mid))), -1)
        sigmas = np.array([self.parallel_sigma, self.parallel_sigma, 0.4])
        logweights = np.log([(1 - self.stop) / 2, (1 - self.stop) / 2, self.stop])
        residual = vp[:, None] - means
        logp = -0.5 * residual ** 2 / sigmas ** 2 - np.log(sigmas) - 0.5 * math.log(2 * math.pi) + logweights
        marginal = logsumexp(logp, -1)
        responsibility = np.exp(logp - marginal[:, None])
        dparallel = (responsibility * residual / sigmas ** 2).sum(-1)
        s2 = self.transverse_sigma ** 2
        lateral = 2 * np.log1p(vt * vt / (3 * s2)) + math.log(self.transverse_sigma)
        dtransverse = 4 * vt / (3 * s2 + vt * vt)
        weight = np.diff(self.knots)
        logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
        axis_weight = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
        value += float(((-marginal + lateral) * weight).sum() - logaxis) * axis_weight
        gv = (dparallel[:, None] * n + dtransverse[:, None] * m) * weight[:, None]
        grad = self.Vmid.T @ gv
        gradient[:2 * (self.n - 1)] += grad[1:].ravel() * axis_weight
        dtheta = float(((dparallel * vt - dtransverse * vp) * weight).sum() + daxis) * axis_weight
        return (value, np.r_[gradient, dtheta])

    def solve_axis(self, alternative, modes):
        if hasattr(self, 'polish_parent_state'):
            return polish_axis(self, modes)
        outcomes = []
        initial_mixture = bool(self.prior.get('joint_marginal_initial_mixture', False))
        if hasattr(self, 'additional_parent_state'):
            parent = self.additional_parent_state['optimization_state']['vector'].copy()
            if hasattr(self, 'heading_context'):
                parent[-1] = 0.0
            value, _ = self.axial_objective(parent, 0)
            outcomes.append(OptimizeResult(x=parent, fun=value, success=False, nit=0, retained_parent=True, message='Retained sensor-only parent under current model', calibration_mode=0, axis_seed_component=0))
            result = minimize(self.axial_objective, parent, args=(0,), jac=True, method='L-BFGS-B', options=dict(maxiter=300, ftol=1e-10, gtol=2e-05, maxls=40))
            result.update(calibration_mode=0, axis_seed_component=0)
            outcomes.append(result)
        if hasattr(self, 'sensor_proposal_paths'):
            if not initial_mixture:
                raise ValueError('Sensor path proposals require the same joint initial device mixture')
            if self.prior.get('save_all_calibration_candidates', False):
                raise ValueError('Use the source joint-marginal candidate convention')
            if modes[0] == 0:
                parent = self.sensor_proposal_parent_vector.copy()
                value, _ = self.axial_objective(parent, 0)
                if abs(value - self.sensor_proposal_parent_objective) > 1e-08 * (1 + abs(value)):
                    raise ValueError('Sensor initialization changed the parent physical objective')
                outcomes.append(OptimizeResult(x=parent, fun=value, success=False, nit=0, retained_parent=True, message='Retained sensor-only parent; no new optimizer convergence claim', calibration_mode=0, axis_seed_component=int(np.argmin(abs(self.axis_parameters[:2] - parent[-1])))))
        mode_axes = [(0, axis) for axis in self.axis_parameters[:2]] if initial_mixture else [(mode, axis) for mode in modes for axis in self.axis_parameters[:2]]
        for mode, axis in mode_axes:
            native = np.asarray([r['mode_xy_m'][mode] for r in self.predictions])
            n = np.array([math.cos(axis), math.sin(axis)])
            normal = np.array([-n[1], n[0]])
            seeds = [self.learned_path, native, alternative]
            boost_paths = [self.learned_path, alternative]
            if initial_mixture:
                group = modes[0] // 2
                seeds = [np.asarray([r['mode_xy_m'][m] for r in self.predictions]) for m in modes]
                if group == 0:
                    seeds.extend([self.learned_path, alternative])
                if group == 1 and self.prior.get('joint_opposite_direction_seeds', False):
                    seeds.extend([2 * self.initial - self.learned_path, 2 * self.initial - alternative])
                if group == 1 and hasattr(self, 'counterfactual_parent_path'):
                    seeds.append(self.counterfactual_parent_path)
                if hasattr(self, 'sensor_proposal_paths'):
                    seeds.extend(self.sensor_proposal_paths[group::4])
                boost_paths = [self.learned_path] if group == 2 else [alternative] if group == 3 else []
            for path in boost_paths:
                grid = np.stack([np.interp(self.knots, self.t, path[:, j]) for j in range(2)], -1)
                derivative = np.gradient(grid, self.knots, axis=0) @ n
                derivative = gaussian_filter1d(derivative, 0.25 / np.median(np.diff(self.knots)))
                sign = np.where(derivative >= 0, 1.0, -1.0)
                speed = np.interp(self.knots, self.t, self.speed)
                longitudinal = np.r_[0.0, np.cumsum(0.5 * (sign[1:] * speed[1:] + sign[:-1] * speed[:-1]) * np.diff(self.knots))]
                lateral = (grid - grid[0]) @ normal
                boosted = self.initial + longitudinal[:, None] * n + lateral[:, None] * normal
                seeds.append(np.stack([np.interp(self.t, self.knots, boosted[:, j]) for j in range(2)], -1))
            for path in seeds:
                q = np.stack([np.interp(self.knots, self.t, path[:, j]) for j in range(2)], -1)
                q += self.initial - q[0]
                x = np.r_[q[1:].ravel(), np.zeros(4), axis]
                result = minimize(self.axial_objective, x, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=int(self.prior.get('path_solver_maxiter', 300)), ftol=1e-08, gtol=0.0002, maxls=30))
                result['calibration_mode'] = int(mode)
                result['axis_seed_component'] = int(np.argmin(abs(self.axis_parameters[:2] - axis)))
                outcomes.append(result)
        result = min(outcomes, key=lambda r: r.fun)
        q, _ = self.unpack(result.x[:-1])
        xy = self.B @ q
        self.last_selected_vector = result.x.copy()
        predictions = [dict(p, xy_m=x.tolist(), inference_timing='OFFLINE_SOURCE_AXIAL_MOTION', trajectory_calibration_mode=int(result['calibration_mode']), later_position_GT_used=False) for p, x in zip(self.predictions, xy)]
        diagnostic = dict(converged=bool(result.success), iterations=int(result.nit), objective=float(result.fun), axis_degrees=float(np.rad2deg(result.x[-1]) % 180), branch_objectives=[float(r.fun) for r in outcomes], stopping_reason=str(result.message), optimization_starts=len(outcomes), branch_stopping_reasons=[str(r.message) for r in outcomes], calibration_mode=int(result['calibration_mode']), later_GT_used=False, source_scene_motion_prior=True, phone_heading_used=False)
        if initial_mixture:
            diagnostic.update(initial_device_mixture_marginalized=True, optimization_native_seed_modes=modes)
        if hasattr(self, 'heading_context'):
            heading_energy = heading_marginal_energy
            _, _, mass = heading_energy(self.Vmid @ q, self.heading_context)
            angles = self.heading_context['angle']
            diagnostic.update(map_axis_integrated=True, heading_hypotheses=len(mass), heading_posterior_MAP_degrees=float(np.rad2deg(angles[mass.argmax()])), heading_posterior_entropy=float(-(mass * np.log(mass.clip(1e-30))).sum()), heading_network_passes_added=0, heading_phone_yaw_used=False)
            diagnostic['axis_degrees'] = diagnostic['heading_posterior_MAP_degrees']
        if 'directional_velocity_parameters' in self.prior:
            diagnostic.update(motion_direction_states=self.prior.get('directional_velocity_directions', 16) + 1, directional_velocity_fraction=self.prior.get('directional_velocity_fraction', 1.0), phone_map_bearing_used=False)
            if self.prior.get('directional_velocity_family_marginalization', False):
                self.axial_objective(result.x, int(result['calibration_mode']))
                diagnostic.update(directional_motion_families_marginalized=True, directional_motion_family_posterior=self.directional_velocity_context['last_family_posterior'].tolist(), directional_motion_family_prior_probability=float(self.prior['directional_velocity_family_probability']), motion_models_are_alternatives_not_duplicate_measurements=True)
        if hasattr(self, 'sensor_proposal_paths'):
            diagnostic.update(sensor_grid_initializations_added=len(self.sensor_proposal_paths[modes[0] // 2::4]), sensor_grid_parent_candidate_retained=modes[0] == 0, sensor_grid_parent_objective=self.sensor_proposal_parent_objective, sensor_grid_parent_objective_recomputed_for_model=bool(self.prior.get('sensor_path_proposal_recompute_parent_objective', False)), sensor_grid_parent_candidate_selected=bool(result.get('retained_parent', False)), sensor_grid_changes_likelihood=False, sensor_grid_reference_positions_used=False)
        if self.prior.get('velocity_gain_marginalization', False):
            theta = float(result.x[-1])
            n = np.array([math.cos(theta), math.sin(theta)])
            m = np.array([-n[1], n[0]])
            velocity = self.Vmid @ q
            _, _, mass = gain_energy(np.stack((velocity @ n, velocity @ m), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'], True)
            gain = gain_components(self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'])
            diagnostic.update(motion_gain_marginalized_for_whole_trajectory=True, motion_gain_components=gain.tolist(), motion_gain_component_posterior=mass.tolist(), motion_gain_posterior_mean=float(mass @ gain))
        if self.prior.get('save_all_calibration_candidates', False):
            candidates = []
            for mode in modes:
                for component in range(2):
                    best = min((r for r in outcomes if r['calibration_mode'] == mode and r['axis_seed_component'] == component), key=lambda r: r.fun)
                    knots, _ = self.unpack(best.x[:-1])
                    points = self.B @ knots
                    candidates.append(dict(calibration_mode=int(mode), candidate_id=int(mode * 2 + component), axis_seed_component=component, objective=float(best.fun), converged=bool(best.success), iterations=int(best.nit), axis_degrees=float(np.rad2deg(best.x[-1]) % 180), xy_m=points.tolist(), knots_s=self.knots.tolist(), knots_xy_m=knots.tolist()))
            diagnostic['candidate_trajectories'] = candidates
        return (predictions, diagnostic)

def innovation_nll(residual, L, pi, gradient=False):
    z = solve_triangular(L, residual, lower=True, check_finite=False)
    diagonal = np.diag(L)
    normal = -0.5 * (z * z + math.log(2 * math.pi)) + np.log1p(-pi)
    scale = 10.0 / diagonal
    heavy = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(scale) - 2 * np.log1p(z * z / (3 * scale * scale)) + np.log(pi)
    logp = np.logaddexp(normal, heavy)
    loss = float((-logp + np.log(diagonal)).sum())
    if not gradient:
        return loss
    responsibility = np.exp(normal - logp)
    derivative = responsibility * z + (1 - responsibility) * 4 * z / (3 * scale * scale + z * z)
    return (loss, solve_triangular(L.T, derivative, lower=False, check_finite=False))

def covariance(parameters, dt, sigma, past_mass, past_time_moment=None):
    white, colored, constant, tau = np.exp(parameters[:4])
    result = np.diag((white * sigma) ** 2 + 0.05 ** 2)
    result += colored ** 2 * np.exp(-dt / tau) + constant ** 2
    result += 0.555741316 ** 2 * np.outer(past_mass, past_mass)
    if len(parameters) > 8:
        if past_time_moment is None:
            raise ValueError('Missing physical history time uncertainty')
        result += math.exp(2 * parameters[8]) * np.outer(past_time_moment, past_time_moment)
    return result

def load_history(prior, folder, event_ids=None):
    folder = Path(folder)
    root = Path(prior['CS_actual_history_query_root']) / folder.parent.name / folder.name
    with np.load(root / 'CS_history_queries.npz', allow_pickle=False) as z:
        h = {k: z[k].copy() for k in z.files}
    if event_ids is not None and (not np.array_equal(h['observation_ids'], event_ids)):
        raise ValueError('Physical CS history IDs differ')
    if 'CS_history_response_weights' not in prior:
        raise ValueError('This runtime requires the selected sample-order kernel')
    w = np.asarray(prior['CS_history_response_weights'])[None] * h['history_mask']
    total = w.sum(-1, keepdims=True)
    fallback = np.zeros_like(w)
    fallback[:, 0] = 1.0
    h['response_weights'] = np.divide(w, total, out=fallback.copy(), where=total > 1e-12)
    h['noise_weights'] = np.zeros_like(w)
    for i in range(len(w)):
        ix = np.where(h['history_mask'][i])[0]
        order = ix[np.argsort(h['raw_history_m'][i, ix], kind='stable')]
        n = len(order)
        if n % 2:
            h['noise_weights'][i, order[n // 2]] = 1.0
        else:
            h['noise_weights'][i, order[n // 2 - 1:n // 2 + 1]] = 0.5
    h['past_time_moment'] = (np.maximum(-h['history_times_s'], 0) * h['response_weights']).sum(-1)
    return h

def history_subset(history, indices):
    return {key: value[indices] for key, value in history.items()}

def measured_noise_response(t, history):
    query = history['history_times_s']
    noise = history['noise_weights']
    S = np.zeros((len(query), len(t)))
    for i in range(len(query)):
        for j in np.where(noise[i] > 0)[0]:
            found = np.where(abs(t - query[i, j]) <= 1e-07)[0]
            if len(found):
                S[i, found[0]] += noise[i, j]
    return S

def directional_velocity_context(parameters, dt, times=None, directions=16):
    p = np.asarray(parameters, float)
    dt = np.asarray(dt, float)
    angle = np.arange(directions) * 2 * math.pi / directions
    u = np.stack((np.cos(angle), np.sin(angle)), -1)
    v = np.stack((-np.sin(angle), np.cos(angle)), -1)
    angular = softmax(p[5] * np.cos(2 * angle) + p[6] * np.cos(4 * angle))
    stop = expit(p[7])
    prob = np.r_[(1 - stop) * angular, stop]
    cosine = np.cos(angle[:, None] - angle[None])
    kappa = np.exp(p[10])
    switch = logsumexp(np.stack((kappa * cosine, p[9] - kappa * cosine), -1), -1) + np.log(angular)[None]
    np.fill_diagonal(switch, -np.inf)
    switch = softmax(switch, axis=-1)
    Q = np.zeros((directions + 1, directions + 1))
    Q[:-1, :-1] = np.exp(p[8]) * switch
    Q[:-1, -1] = np.exp(p[11])
    Q[-1, :-1] = np.exp(p[12]) * angular
    np.fill_diagonal(Q, -Q.sum(-1))
    mid = np.cumsum(dt) - 0.5 * dt if times is None else np.asarray(times, float)
    transitions = [expm(Q * gap) for gap in np.diff(mid)]
    if transitions and (min((float(T.min()) for T in transitions)) < -1e-12 or max((float(abs(T.sum(-1) - 1).max()) for T in transitions)) > 1e-10):
        raise ValueError('Directional motion transition is not a probability matrix')
    sigma = np.exp(p[:2])[None, None] / np.sqrt(dt)[:, None, None]
    gain = np.exp(p[3] * np.cos(angle) ** 2 + p[4] * np.sin(angle) ** 2)
    return dict(u=u, v=v, gain=gain, sigma=sigma, stop_sigma=np.exp(p[2]) / np.sqrt(dt), probability=prob, transitions=transitions)

def directional_velocity_energy(velocity, speed, ctx, derivative=False):
    velocity = np.asarray(velocity)
    speed = np.asarray(speed)
    N = len(speed)
    K = len(ctx['probability'])
    residual = np.stack((velocity @ ctx['u'].T - speed[:, None] * ctx['gain'][None], velocity @ ctx['v'].T), -1)
    sig = ctx['sigma']
    moving = (CONST - np.log(sig) - 2 * np.log1p(residual ** 2 / (3 * sig ** 2))).sum(-1)
    stop = ctx['stop_sigma'][:, None]
    still = (CONST - np.log(stop) - 2 * np.log1p(velocity ** 2 / (3 * stop ** 2))).sum(-1)
    emission = np.column_stack((moving, still))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty((N, K))
    scales = np.empty(N)
    alpha[0] = ctx['probability'] * density[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ ctx['transitions'][i - 1] * density[i]
        scales[i] = alpha[i].sum()
        alpha[i] /= scales[i]
    value = -float((np.log(scales) + maximum).sum())
    if not derivative:
        return value
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[i] = ctx['transitions'][i] @ (density[i + 1] * backward[i + 1]) / scales[i + 1]
    weight = alpha * backward
    weight /= weight.sum(-1, keepdims=True)
    score = 4 * residual / (3 * sig ** 2 + residual ** 2)
    g = (weight[:, :-1, None] * (score[..., 0, None] * ctx['u'][None] + score[..., 1, None] * ctx['v'][None])).sum(1)
    g += weight[:, -1, None] * 4 * velocity / (3 * stop ** 2 + velocity ** 2)
    return (value, g, weight)

def raw_scale(t, raw):
    result = []
    for now in t:
        values = raw[abs(t - now) <= 1.5]
        if len(values) < 3:
            values = raw[abs(t - now) <= 3.0]
        mad = 1.4826 * np.median(abs(values - np.median(values))) if len(values) else 1.0
        result.append(0.35 + min(float(mad), 10.0))
    return np.asarray(result)

def response_matrix(t, lags, weights):
    S = np.zeros((len(t), len(t)))
    for i, now in enumerate(t):
        for lag, w in zip(lags, weights):
            query = now - lag
            if query < t[0] or query > t[-1] or (not w):
                continue
            right = int(np.searchsorted(t, query))
            if right == 0:
                S[i, 0] += w
            elif right == len(t):
                S[i, -1] += w
            else:
                left = right - 1
                fraction = (query - t[left]) / max(t[right] - t[left], 1e-08)
                S[i, left] += w * (1 - fraction)
                S[i, right] += w * fraction
    return S

def joint_noise(p, b, prior):
    scale = np.asarray(b['raw_sigma_pred'], float) if 'raw_sigma_pred' in b else np.exp(p[b['anchor']]) * b['raw_scale']
    diagonal = scale ** 2 + 0.05 ** 2
    bias_sigma = np.exp(p[4])
    shared = math.tanh(p[5])
    cross = shared * diagonal[:, None] * b['response'].T
    corrected = covariance(np.asarray(prior['CS_temporal_innovation_parameters']), b['dt'], b['sigma'], b['past_mass'], b.get('past_time_moment'))
    corrected = corrected * math.exp(2 * p[12]) + shared ** 2 * (b['response'] * diagonal[None]) @ b['response'].T
    n = len(diagonal)
    K = np.zeros((2 * n, 2 * n))
    K[:n, :n] = np.diag(diagonal) + bias_sigma ** 2
    K[:n, n:] = cross
    K[n:, :n] = cross.T
    K[n:, n:] = corrected
    L = cholesky(K, lower=True, check_finite=False)
    pi = np.r_[np.asarray(b['raw_pi_pred'], float) if 'raw_pi_pred' in b else np.full(n, expit(p[7])), b['pi']]
    return (L, pi)

def metadata(anchor, t, raw, sigma, pi, past_mass, prior, history=None):
    response = response_matrix(t, np.asarray(prior['CS_response_lags_s']), np.asarray(prior['CS_response_weights']))
    if history is not None:
        response = measured_noise_response(t, history)
    return dict(anchor=anchor, raw_scale=raw_scale(t, raw), response=response, dt=abs(t[:, None] - t[None, :]), sigma=sigma, pi=np.clip(pi, 1e-05, 1 - 1e-05), past_mass=past_mass, past_time_moment=None if history is None else history['past_time_moment'])

class DualAxialPath(AxialPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(args[0] / 'CS_training_contexts.npz', allow_pickle=False) as z:
            self.raw_range = z['features'][:, 0].copy() * 20
        with np.load(args[0] / 'trajectory_factors.npz', allow_pickle=False) as z:
            sigma = z['cs_sigma'].copy()
        self.dual = np.asarray(self.prior['dual_CS_parameters'])
        self.dual_blocks = []
        self.raw_anchor = self.anchor.copy()
        self.raw_height2 = np.zeros(len(self.ct))
        if 'CS_raw_geometry' in self.prior:
            geometry = self.prior['CS_raw_geometry']
            self.raw_anchor += np.asarray(geometry['anchor_shifts_xy_m'])[self.anchor_index]
            self.raw_height2 = np.asarray(geometry['effective_height_squared_m2'])[self.anchor_index]
        quality = None
        if 'raw_quality_output_root' in self.prior:
            folder = Path(self.prior['raw_quality_output_root']) / args[0].parent.name / args[0].name
            with np.load(folder / 'raw_quality.npz', allow_pickle=False) as z:
                quality = {k: z[k].copy() for k in z.files}
            if not np.array_equal(quality['anchor_index'], self.anchor_index):
                raise ValueError('Raw quality anchor IDs differ')
            if not np.allclose((quality['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                raise ValueError('Raw quality event times differ')
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            if self.cs_history_metadata is not None:
                history = history_subset(self.cs_history_metadata, ix)
            else:
                history = None
            b = metadata(anchor, self.ct[ix], self.raw_range[ix], sigma[ix], self.pi[ix], self.cs_past_mass[ix], self.prior, history)
            if quality is not None:
                b['raw_sigma_pred'] = quality['sigma_m'][ix]
                b['raw_pi_pred'] = quality['outlier_probability'][ix]
            L, pi = joint_noise(self.dual, b, self.prior)
            if self.prior.get('raw_only_CS', False):
                L = L[:len(ix), :len(ix)]
                pi = pi[:len(ix)]
            self.dual_blocks.append((anchor, ix, L, pi))

    def path_objective(self, vector, mode):
        value, gradient = SensorPath.objective(self, vector, mode)
        q, _ = self.unpack(vector)
        current = self.C @ q
        delta = current - self.raw_anchor
        d_raw = np.sqrt(np.square(delta).sum(-1) + self.raw_height2 + 1e-08)
        history = np.einsum('mln,nd->mld', self.C_history, q) - self.anchor[:, None]
        history_d = np.sqrt(np.square(history).sum(-1) + 1e-08)
        d_corrected = (history_d * self.cs_response_weights).sum(-1)
        gain = math.exp(self.dual[6])
        Gmean = np.asarray(self.prior['CS_temporal_innovation_parameters'])[4:]
        r_raw = self.raw_range - gain * d_raw - self.dual[8:12][self.anchor_index]
        r_corrected = self.cs_range - d_corrected - Gmean[self.anchor_index]
        raw_derivative = np.zeros(len(self.ct))
        corrected_derivative = np.zeros(len(self.ct))
        for anchor, ix, L, pi in self.dual_blocks:
            residual = r_raw[ix] if self.prior.get('raw_only_CS', False) else np.r_[r_raw[ix], r_corrected[ix]]
            loss, score = innovation_nll(residual, L, pi, True)
            value += loss
            raw_derivative[ix] = -gain * score[:len(ix)]
            if not self.prior.get('raw_only_CS', False):
                corrected_derivative[ix] = -score[len(ix):]
        grad = self.C.T @ (raw_derivative[:, None] * delta / d_raw[:, None])
        grad += np.einsum('mln,mld,ml,m->nd', self.C_history, history / history_d[..., None], self.cs_response_weights, corrected_derivative)
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

def generator(parameters, phases):
    p = np.asarray(parameters)
    mass = softmax([0.0, p[3], p[4]])
    pi = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    rate = phases / np.exp(p[[0, 0, 1, 1, 2]])
    Q = np.zeros((5 * phases, 5 * phases))
    for state in range(5):
        for phase in range(phases):
            row = state * phases + phase
            if phase < phases - 1:
                Q[row, row + 1] = rate[state]
            else:
                Q[row, np.arange(5) * phases] = rate[state] * pi
    Q -= np.diag(Q.sum(-1))
    return (Q, np.repeat(pi / phases, phases))

def duration_motion_context(prior, speed, dt, speed_sigma=None):
    phases = int(prior['duration_motion_phases'])
    Q, pi = generator(prior['duration_motion_parameters'], phases)
    p = np.asarray(prior['velocity_state_parameters'])
    means = np.zeros((len(speed), 5, 2))
    means[:, 0, 0] = speed
    means[:, 1, 0] = -np.asarray(speed)
    means[:, 2, 1] = math.exp(p[3]) * np.asarray(speed)
    means[:, 3, 1] = -math.exp(p[3]) * np.asarray(speed)
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        extra = np.zeros_like(sigma)
        v = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, :2, 0] = v[:, None]
        extra[:, 2:4, 1] = math.exp(2 * p[3]) * v[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    transitions = []
    cache = {}
    for gap in 0.5 * (np.asarray(dt[:-1]) + np.asarray(dt[1:])):
        key = round(float(gap), 10)
        if key not in cache:
            T = expm(Q * gap)
            if T.min() < -1e-12 or abs(T.sum(-1) - 1).max() > 1e-10:
                raise ValueError('Invalid duration transition matrix')
            cache[key] = T.clip(0)
        transitions.append(cache[key])
    return dict(means=np.repeat(means, phases, axis=1), sigma=np.repeat(sigma, phases, axis=1), probability=pi, transition=transitions)

def force_context(evidence, query):
    token = np.asarray(evidence['motion'], float)
    t = np.asarray(evidence['times'], float)
    columns = token[:, :6].reshape(-1, 3, 2)
    R = np.concatenate((columns, np.cross(columns[..., 0], columns[..., 1])[:, :, None]), -1)
    valid = (token[:, 22] > 0.5) & (token[:, 21] * 5 <= 0.5)
    if not valid.any():
        return (np.zeros((len(query), 2)), np.zeros(len(query), bool))
    gravity = np.einsum('nij,nj->ni', R[valid], token[valid, 8:11]).mean(0)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-08)
    ex = np.array([1.0, 0, 0])
    ex -= gravity * (ex @ gravity)
    if np.linalg.norm(ex) < 0.2:
        ex = np.array([0.0, 1, 0])
        ex -= gravity * (ex @ gravity)
    ex /= max(float(np.linalg.norm(ex)), 1e-08)
    ey = np.cross(gravity, ex)
    acc = token[:, 11:14] * 5
    acc = np.stack((acc @ ex, acc @ ey), -1)
    dt = np.diff(np.r_[t[0] - 0.1, t]).clip(0.025, 0.5)
    weight = np.exp(-0.5 * ((query[:, None] - t[None]) / 0.35) ** 2) * valid[None] * dt[None]
    weight /= weight.sum(-1, keepdims=True).clip(1e-08)
    available = (query >= t[valid][0] + 0.7) & (query <= t[valid][-1] - 0.7)
    return (weight @ acc, available)

def force_energy(A, Y, weight, parameters, derivative=False, observation_model=False):
    gain, sigma, bias = np.exp(parameters)
    sigma2 = sigma * sigma
    mass = float(weight.sum())
    W = np.diag(weight) - np.outer(weight, weight) / (mass + sigma2 / (bias * bias))
    WA = W @ A
    WY = W @ Y
    C = A.T @ WY
    dot = float(C[0, 0] + C[1, 1])
    cross = float(C[0, 1] - C[1, 0])
    strength = math.hypot(dot, cross)
    concentration = gain * strength / sigma2
    logBessel = math.log(float(i0e(concentration))) + concentration
    square = (A * WA).sum() + gain * gain * (Y * WY).sum() if observation_model else (Y * WY).sum() + gain * gain * (A * WA).sum()
    energy = 0.5 * float(square) / sigma2 - logBessel
    energy += len(A) * math.log(2 * math.pi) + len(A) * math.log(sigma2) - np.log(weight).sum() + math.log1p(bias * bias * mass / sigma2)
    if not derivative:
        return float(energy)
    angle = math.atan2(cross, dot)
    c, s = (math.cos(angle), math.sin(angle))
    rotation = np.array([[c, s], [-s, c]])
    resultant = float(i1e(concentration) / i0e(concentration))
    gradient = ((gain * gain if observation_model else 1.0) * WY - gain * resultant * WA @ rotation) / sigma2
    return (float(energy), gradient, dict(force_rotation_concentration=concentration, force_rotation_resultant=resultant, force_frame_rotation_mode_degrees=math.degrees(angle), motion_heading_not_observed=True, force_gain=gain, force_noise_sigma=sigma, force_bias_sigma=bias, observation_model=observation_model))

class ForceDualPath(DualAxialPath):

    def __init__(self, *args, **kwargs):
        evidence = kwargs.pop('force_context')
        super().__init__(*args, **kwargs)
        A, valid = force_context(evidence, self.mid)
        self.force_ix = np.where(valid)[0]
        self.measured_force = A[valid]
        self.force_matrix = self.A[self.force_ix]
        self.force_weight = self.adt[self.force_ix]
        self.force_intervals = None
        if 'force_preintegration_minimum_span_s' in self.prior:
            intervals, self.measured_force = measured_intervals(evidence, self.prior['force_preintegration_minimum_span_s'])
            width = intervals[:, 1] - intervals[:, 0]
            self.force_matrix = (self.spline(intervals[:, 1], 1) - self.spline(intervals[:, 0], 1)) / width[:, None]
            self.force_weight = width
            self.force_intervals = intervals
            covered = ((self.mid[:, None] >= intervals[None, :, 0]) & (self.mid[:, None] <= intervals[None, :, 1])).any(-1)
            self.force_ix = np.where(covered)[0]
        self.force_parameters = np.asarray(self.prior['marginal_force_parameters'])
        if not self.prior['marginal_force_selected']:
            raise ValueError('Force evidence failed source selection')

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        if len(self.measured_force) < 5:
            return (value, gradient)
        q, _ = self.unpack(vector)
        ix = self.force_ix
        A = self.A[ix]
        weight = self.adt[ix]
        acceleration = A @ q
        old = acceleration.copy()
        scale2 = self.prior['acceleration_scale_m_s2'] ** 2
        if self.prior.get('learned_acceleration_center', False):
            old -= self.learned_acceleration[ix]
            scale2 = self.prior['learned_acceleration_sigma_m_s2'] ** 2
        square = np.square(old).sum(-1)
        balance = self.prior.get('trajectory_factor_weights', {})
        old_weight = balance.get('acceleration', 1.0)
        force_weight = balance.get('force', 1.0)
        value -= float((2.5 * np.log1p(square / (3 * scale2)) * weight).sum()) * old_weight
        g_old = A.T @ (5 * old / (3 * scale2 + square)[:, None] * weight[:, None]) * old_weight
        loss, g_force, _ = force_energy(self.measured_force, self.force_matrix @ q, self.force_weight, self.force_parameters, True, self.prior.get('marginal_force_observation_model', False))
        value += loss * force_weight
        grad = self.force_matrix.T @ g_force * force_weight - g_old
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        if len(self.measured_force) >= 5:
            q, _ = self.unpack(self.last_selected_vector[:-1])
            ix = self.force_ix
            _, _, d = force_energy(self.measured_force, self.force_matrix @ q, self.force_weight, self.force_parameters, True, self.prior.get('marginal_force_observation_model', False))
            diagnostic['force_evidence'] = d
            if self.force_intervals is not None:
                diagnostic['force_evidence'].update(nonoverlapping_actual_interval_count=len(self.force_intervals), actual_interval_bounds_s=self.force_intervals.tolist(), known_phone_heading_used=False)
        return (predictions, diagnostic)

def measured_intervals(evidence, minimum_span):
    token = np.asarray(evidence['motion'], float)
    t = np.asarray(evidence['times'], float)
    columns = token[:, :6].reshape(-1, 3, 2)
    R = np.concatenate((columns, np.cross(columns[..., 0], columns[..., 1])[:, :, None]), -1)
    valid = (token[:, 22] > 0.5) & (token[:, 21] * 5 <= 0.5) & (token[:, 23] > 0)
    if not valid.any():
        return (np.zeros((0, 2)), np.zeros((0, 2)))
    gravity = np.einsum('nij,nj->ni', R[valid], token[valid, 8:11]).mean(0)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-08)
    ex = np.array([1.0, 0, 0])
    ex -= gravity * (ex @ gravity)
    if np.linalg.norm(ex) < 0.2:
        ex = np.array([0.0, 1, 0])
        ex -= gravity * (ex @ gravity)
    ex /= max(float(np.linalg.norm(ex)), 1e-08)
    ey = np.cross(gravity, ex)
    acc = token[:, 11:14] * 5
    acc = np.stack((acc @ ex, acc @ ey), -1)
    dt = token[:, 20] * 5
    left = t - dt
    if len(t) and dt[0] <= 1e-06:
        dt[0] = t[0]
        left[0] = 0.0
    intervals = []
    means = []
    group = []
    for i in range(len(t)):
        if not valid[i] or dt[i] <= 1e-05 or dt[i] > 2.0 or (left[i] < 0.5) or (t[i] > t[-1] - 0.5):
            group = []
            continue
        if group and abs(left[i] - t[group[-1]]) > 0.05:
            group = []
        group.append(i)
        span = t[i] - left[group[0]]
        if span < minimum_span:
            continue
        indices = np.asarray(group)
        coverage = float(dt[indices].sum() / span)
        if 0.9 <= coverage <= 1.1:
            intervals.append([float(left[group[0]]), float(t[i])])
            means.append((acc[indices] * dt[indices, None]).sum(0) / dt[indices].sum())
        group = []
    return (np.asarray(intervals, float).reshape(-1, 2), np.asarray(means, float).reshape(-1, 2))

def potential_energy(xy, grid, potential, weight, bandwidth, derivative=False):
    kernel = -0.5 * np.square(grid[None] - xy[:, None]).sum(-1) / bandwidth ** 2
    ln = logsumexp(kernel + potential, axis=-1)
    ld = logsumexp(kernel, axis=-1)
    value = -float((weight * (ln - ld)).sum())
    if not derivative:
        return value
    pn = np.exp(kernel + potential - ln[:, None])
    pd = np.exp(kernel - ld[:, None])
    gradient = (pd - pn) @ grid / bandwidth ** 2 * weight[:, None]
    return (value, gradient)

def heading_marginal_context(speed, dt, parameters, axis_parameters, speed_sigma=None, hypotheses=256):
    p = np.asarray(parameters)
    dt = np.asarray(dt)
    N = len(speed)
    angle = (np.arange(hypotheses) + 0.5) * math.pi / hypotheses
    u = np.stack((np.cos(angle), np.sin(angle)), -1)
    v = np.stack((-np.sin(angle), np.cos(angle)), -1)
    ap = np.asarray(axis_parameters)
    kappa = np.exp(ap[2:4])
    mix = 1 / (1 + np.exp(-ap[4]))
    logs = np.log([mix, 1 - mix])[None] + kappa[None] * np.cos(2 * (angle[:, None] - ap[None, :2])) - math.log(math.pi) - np.log(i0e(kappa))[None] - kappa[None]
    log_prior = logsumexp(logs, -1)
    log_prior -= logsumexp(log_prior)
    means = np.zeros((N, 5, 2))
    gain = math.exp(p[3])
    means[:, 0, 0] = speed
    means[:, 1, 0] = -speed
    means[:, 2, 1] = gain * speed
    means[:, 3, 1] = -gain * speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        if speed_sigma is None:
            raise ValueError('Missing measured speed uncertainty')
        extra = np.zeros_like(sigma)
        variance = np.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    tau = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / tau)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    return dict(u=u, v=v, means=means, sigma=sigma, probability=prob, transitions=transition, log_prior=log_prior, angle=angle)

def heading_marginal_energy(velocity, ctx, derivative=True):
    vp = ctx['u'] @ np.asarray(velocity).T
    vt = ctx['v'] @ np.asarray(velocity).T
    residual = np.stack((vp, vt), -1)[:, :, None] - ctx['means'][None]
    sig = ctx['sigma'][None]
    emission = (CONST - np.log(sig) - 2 * np.log1p(residual ** 2 / (3 * sig ** 2))).sum(-1)
    P, N, K = emission.shape
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, :, None])
    alpha = np.empty_like(density)
    scale = np.empty((P, N))
    alpha[:, 0] = ctx['probability'][None] * density[:, 0]
    scale[:, 0] = alpha[:, 0].sum(-1)
    alpha[:, 0] /= scale[:, 0, None]
    for i in range(1, N):
        alpha[:, i] = alpha[:, i - 1] @ ctx['transitions'][i - 1] * density[:, i]
        scale[:, i] = alpha[:, i].sum(-1)
        alpha[:, i] /= scale[:, i, None]
    conditional_energy = -(np.log(scale) + maximum).sum(-1)
    evidence = ctx['log_prior'] - conditional_energy
    normalizer = logsumexp(evidence)
    mass = np.exp(evidence - normalizer)
    if not derivative:
        return -float(normalizer)
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[:, i] = density[:, i + 1] * backward[:, i + 1] @ ctx['transitions'][i].T / scale[:, i + 1, None]
    phase = alpha * backward
    phase /= phase.sum(-1, keepdims=True)
    score = (phase[:, :, :, None] * 4 * residual / (3 * sig ** 2 + residual ** 2)).sum(2)
    world = score[..., 0, None] * ctx['u'][:, None] + score[..., 1, None] * ctx['v'][:, None]
    return (-float(normalizer), (mass[:, None, None] * world).sum(0), mass)

class InitialDependency:

    def __init__(self, folder, pre, configuration, strength=None):
        beta = float(configuration.get('initial_position_device_dependency_strength', 0.0) if strength is None else strength)
        if not 0 <= beta <= 1:
            raise ValueError('Conditional coupling must be between zero and one')
        base = initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0))
        with np.load(folder / 'native_reference_full25.npz', allow_pickle=False) as z:
            if not np.array_equal(z['observation_ids'], pre['observation_ids']):
                raise ValueError('Native joint and cavity observation IDs differ')
            native_mean = z['means'][5].astype(float)
            native = z['covariances'][5].astype(float)
        if native_mean.shape != (8, 15):
            raise ValueError('Expected the frozen native eight-component 15D state')
        native = 0.5 * (native + native.swapaxes(-1, -2)) + np.eye(15)[None] * 0.0001
        oldL = np.linalg.cholesky(native[:, 9:15, 9:15])
        newL = np.linalg.cholesky(base['covariance'])
        transfer = np.linalg.solve(oldL.swapaxes(-1, -2), newL.swapaxes(-1, -2)).swapaxes(-1, -2)
        cross = transfer @ native[:, 9:15, :2]
        X = native[:, :2, :2]
        T = np.linalg.solve(X, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        covariance = base['covariance'] - beta ** 2 * T @ cross.swapaxes(-1, -2)
        covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2))
        np.linalg.cholesky(covariance)
        adopted = np.concatenate((np.concatenate((X, cross.swapaxes(-1, -2)), axis=-1), np.concatenate((cross, base['covariance']), axis=-1)), axis=-2)
        np.linalg.cholesky(adopted)
        transported = transfer @ native[:, 9:15, 9:15] @ transfer.swapaxes(-1, -2)
        self.marginal_covariance_difference = float(abs(transported - base['covariance']).max())
        if self.marginal_covariance_difference > 1e-08:
            raise ValueError('Native-to-cavity marginal transport failed')
        self.base = base
        self.native_position_mean = native_mean[:, :2]
        self.T = beta * T
        self.covariance = covariance
        self.precision = np.linalg.inv(covariance)
        self.strength = beta
        self.logdet = np.linalg.slogdet(covariance)[1]
        self.original_device_mean_difference = float(abs(native_mean[:, 9:15] - base['mean']).max())

    def at(self, xy):
        if self.strength == 0:
            return self.base
        mean = self.base['mean'] + np.einsum('kdz,kz->kd', self.T, np.asarray(xy)[None] - self.native_position_mean)
        return dict(mean=mean, covariance=self.covariance, precision=self.precision, vector=np.einsum('kde,ke->kd', self.precision, mean), logdet=self.logdet, log_weight=self.base['log_weight'])

    def posterior(self, z, A, p, logdet, derivative=False):
        values = [joint_marginal_posterior(z, A, p, k, logdet) for k in range(8)]
        evidence = p['log_weight'] - np.asarray([v[0] for v in values])
        normalizer = logsumexp(evidence)
        weight = np.exp(evidence - normalizer)
        means = np.asarray([v[1] for v in values])
        covs = np.asarray([v[2] for v in values])
        mean = weight @ means
        delta = means - mean
        covariance = np.einsum('k,kde->de', weight, covs + delta[:, :, None] * delta[:, None, :])
        result = (-float(normalizer), mean, covariance, z - A @ mean, weight)
        if not derivative:
            return result
        grad_mu = np.einsum('kde,ke->kd', p['precision'], p['mean'] - means)
        gradient = np.einsum('k,kdz,kd->z', weight, self.T, grad_mu)
        return result + (gradient,)

def unpack_auxiliary(parameters):
    p = np.asarray(parameters, float)
    return dict(gain=p[:2], anchor_mean=p[2:10].reshape(4, 2), alias=np.vstack((np.zeros(2), p[10:14].reshape(2, 2))), beta=p[14:22].reshape(4, 2), sigma=np.exp(p[22:30].reshape(4, 2)), rho=0.95 * np.tanh(p[30:34]), background=p[34:42].reshape(4, 2))

def conditional_numpy(residual, valid, sigma, rho, derivative=False):
    m = np.asarray(valid, bool)
    r = residual * m
    both = m.all(-1)
    k = m.sum(-1)
    s = np.broadcast_to(sigma, r.shape)
    rho = np.broadcast_to(rho, r.shape[:-1])
    den = 1 - rho * rho
    w = r / (s * s)
    w_both = np.stack(((r[..., 0] / s[..., 0] ** 2 - rho * r[..., 1] / (s[..., 0] * s[..., 1])) / den, (r[..., 1] / s[..., 1] ** 2 - rho * r[..., 0] / (s[..., 0] * s[..., 1])) / den), -1)
    w = np.where(both[..., None], w_both, w) * m
    v = (r * w).sum(-1)
    ld = (2 * np.log(s) * m).sum(-1) + np.where(both, np.log(den), 0.0)
    nll = gammaln(2) - gammaln((4 + k) / 2) + k / 2 * math.log(4 * math.pi) + 0.5 * ld + (4 + k) / 2 * np.log1p(v / 4)
    if not derivative:
        return nll
    return (nll, (4 + k)[..., None] / (4 + v)[..., None] * w)

def auxiliary_emission(distance, error, context, derivative=False):
    p = unpack_auxiliary(context['auxiliary_parameters'])
    anchor = context['anchor']
    expected = np.asarray(distance)[:, None, None] * p['gain'][None, None, :] + p['anchor_mean'][anchor][None, None, :] + p['alias'][None]
    expected = np.concatenate((expected, np.broadcast_to(p['background'][anchor], (len(distance), 1, 2))), 1)
    expected += np.asarray(context.get('auxiliary_bias_m', [0.0, 0.0]))[None, None]
    residual = context['auxiliary_values'][:, None, :] - expected - p['beta'][None] * error[:, :, None]
    valid = np.broadcast_to(context['auxiliary_valid'][:, None, :], residual.shape)
    out = conditional_numpy(residual, valid, p['sigma'][None], p['rho'][None], derivative)
    if not derivative:
        return -out
    loss, score = out
    g0 = context['parameters']['gain']
    context['last_auxiliary_score'] = score
    slope = np.concatenate((-p['gain'][None] + p['beta'][:3] * g0, np.zeros((1, 2))), 0)
    return (-loss, (score * slope[None]).sum(-1))

def null_auxiliary_energy(context):
    z = np.asarray(context['auxiliary_null_parameters'], float)
    anchor = context['anchor']
    mean = z[2:10].reshape(4, 2)[anchor] + z[:2] * (context['raw'] - context['background'])[:, None]
    return float(conditional_numpy(context['auxiliary_values'] - mean, context['auxiliary_valid'], np.exp(z[10:12]), 0.95 * math.tanh(z[12])).sum())

def initial_prior(pre, uncertainty_scale=1.0):
    if len(pre['device_mean']) <= 5:
        raise ValueError('Missing complete initial-five posterior')
    mean = pre['device_mean'][5].astype(float)
    cov = pre['device_covariance'][5].astype(float)
    cov = 0.5 * (cov + cov.swapaxes(-1, -2)) + np.eye(6)[None] * 0.0001
    cov *= float(uncertainty_scale) ** 2
    precision = np.linalg.inv(cov)
    logdet = np.linalg.slogdet(cov)[1]
    weight = pre['weights'][5].astype(float)
    weight /= weight.sum()
    return dict(mean=mean, covariance=cov, precision=precision, vector=np.einsum('kde,ke->kd', precision, mean), logdet=logdet, log_weight=np.log(weight.clip(1e-20)))

def whiten(t, mask, y, mean, basis, parameters):
    blocks = radio_blocks(t, mask, parameters)
    z = []
    A = []
    logdet = 0.0
    for ap, ix, L in blocks:
        z.append(solve_triangular(L, y[ix, ap] - mean[ix, ap], lower=True, check_finite=False))
        A.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
        logdet += 2 * np.log(np.diag(L)).sum()
    if not z:
        return (np.zeros(0), np.zeros((0, 6)), 0.0, blocks)
    return (np.concatenate(z), np.concatenate(A), float(logdet), blocks)

def joint_marginal_posterior(z, A, prior, mode, noise_logdet=0.0):
    precision = prior['precision'][mode] + A.T @ A
    L = cholesky(precision, lower=True, check_finite=False)
    covariance = cho_solve((L, True), np.eye(A.shape[1]), check_finite=False)
    mean = covariance @ (prior['vector'][mode] + A.T @ z)
    residual = z - A @ mean
    d = mean - prior['mean'][mode]
    value = 0.5 * float(residual @ residual + d @ prior['precision'][mode] @ d + noise_logdet + prior['logdet'][mode] + 2 * np.log(np.diag(L)).sum() + len(z) * math.log(2 * math.pi))
    return (value, mean, covariance, residual)

def mixture_posterior(z, A, prior, noise_logdet=0.0):
    values = [joint_marginal_posterior(z, A, prior, k, noise_logdet) for k in range(len(prior['mean']))]
    evidence = prior['log_weight'] - np.asarray([v[0] for v in values])
    normalizer = logsumexp(evidence)
    weight = np.exp(evidence - normalizer)
    means = np.asarray([v[1] for v in values])
    covariances = np.asarray([v[2] for v in values])
    mean = (weight[:, None] * means).sum(0)
    difference = means - mean
    covariance = (weight[:, None, None] * (covariances + difference[:, :, None] * difference[:, None, :])).sum(0)
    return (-float(normalizer), mean, covariance, z - A @ mean, weight)

class RawHMMPath(DualAxialPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hmm = []
        quality = None
        conditional_bias = None
        auxiliary = None
        if 'raw_HMM_context_root' in self.prior:
            with np.load(Path(self.prior['raw_HMM_context_root']) / args[0].parent.name / args[0].name / 'raw_HMM_context.npz', allow_pickle=False) as z:
                if not np.array_equal(z['anchor_index'], self.anchor_index):
                    raise ValueError('HMM context anchor mismatch')
                if not np.allclose((z['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                    raise ValueError('HMM context time mismatch')
                quality = z['component_probability'].copy()
                if 'conditional_bias_m' in z.files:
                    conditional_bias = z['conditional_bias_m'].copy()
        if self.prior.get('joint_RF_auxiliary_selected', False):
            if quality is not None or np.any(np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4))):
                raise ValueError('Joint RF density uses original IFFT states without a duplicate auxiliary gate or an extra IFFT-only bias')
            if self.prior.get('joint_RF_auxiliary_bias_prior_selected', False) and self.prior.get('joint_RF_family_mixture_selected', False):
                raise ValueError('Auxiliary bias and family-mixture experiments are separate branches')
            with np.load(Path(self.prior['joint_RF_auxiliary_query_root']) / args[0].parent.name / args[0].name / 'CS_history_queries.npz', allow_pickle=False) as z:
                if not np.array_equal(z['anchor_index'], self.anchor_index):
                    raise ValueError('Joint RF auxiliary anchor mismatch')
                if not np.allclose((z['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                    raise ValueError('Joint RF auxiliary clock mismatch')
                values = z['auxiliary_values'][:, :2].copy()
                valid = z['auxiliary_is_finite'][:, :2] & (values > 0)
                valid[:, 1] &= z['auxiliary_values'][:, 3] > 0
                auxiliary = (values, valid)
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            context = block_context(anchor, self.ct[ix], self.raw_range[ix], raw_scale(self.ct[ix], self.raw_range[ix]), self.prior['raw_HMM_parameters'], self.prior['raw_HMM_background_range_m'], self.prior['raw_HMM_temporal_selected'], None if quality is None else quality[ix], None if conditional_bias is None else conditional_bias[ix])
            if auxiliary is not None:
                context.update(auxiliary_parameters=self.prior['joint_RF_auxiliary_parameters'], auxiliary_values=auxiliary[0][ix], auxiliary_valid=auxiliary[1][ix])
                if self.prior.get('joint_RF_family_mixture_selected', False):
                    context['auxiliary_null_parameters'] = self.prior['joint_RF_null_auxiliary_parameters']
                    context['auxiliary_null_loss'] = null_auxiliary_energy(context)
                    context['auxiliary_family_probability'] = self.prior['joint_RF_family_probability'][anchor]
            self.hmm.append((ix, context))

    def auxiliary_bias(self, latent):
        if not self.prior.get('joint_RF_auxiliary_bias_prior_selected', False):
            return (None, None)
        standard = np.broadcast_to(np.asarray(self.prior.get('path_CS_bias_sigma_m', 0.5), float), (4,))[:2]
        L = np.asarray(self.prior['joint_RF_auxiliary_bias_prior_cholesky_m'], float)
        return (np.asarray(self.prior['joint_RF_auxiliary_bias_prior_mean_m']) + L @ (latent[:2] / standard), L / standard[None])

    def path_objective(self, vector, mode):
        if self.prior.get('direct_CS_policy') == 'joint_raw_corrected':
            return DualAxialPath.path_objective(self, vector, mode)
        value, gradient = SensorPath.objective(self, vector, mode)
        q, latent = self.unpack(vector)
        delta = self.C @ q - self.raw_anchor
        distance = np.sqrt(np.square(delta).sum(-1) + 1e-08)
        bias_sigma = np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4), float)
        bias = bias_sigma * latent
        auxiliary_bias, auxiliary_map = self.auxiliary_bias(latent)
        gd = np.zeros(len(distance))
        weight = self.prior.get('trajectory_factor_weights', {}).get('range', 1.0)
        for ix, context in self.hmm:
            anchor = context['anchor']
            if auxiliary_bias is not None:
                context['auxiliary_bias_m'] = auxiliary_bias
            loss, g, _ = raw_hmm_energy(distance[ix], context, True, float(bias[anchor]))
            value += loss * weight
            gd[ix] = g * weight
            gradient[-4 + anchor] += float(g.sum() / context['parameters']['gain']) * bias_sigma[anchor] * weight
            if auxiliary_bias is not None:
                gradient[-4:-2] += auxiliary_map.T @ context['last_auxiliary_bias_gradient'] * weight
        g = self.C.T @ (gd[:, None] * delta / distance[:, None])
        gradient[:2 * (self.n - 1)] += g[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        if self.prior.get('direct_CS_policy') == 'joint_raw_corrected':
            diagnostic.update(direct_CS_policy='joint_raw_corrected', raw_HMM_factor_used=False, corrected_CS_factor='Frozen preprocessed range with source-fitted temporal response and shared raw noise')
            return (predictions, diagnostic)
        q, latent = self.unpack(self.last_selected_vector[:-1])
        distance = np.linalg.norm(self.C @ q - self.raw_anchor, axis=-1)
        bias = np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4)) * latent
        auxiliary_bias, _ = self.auxiliary_bias(latent)
        posterior = []
        for ix, context in self.hmm:
            if auxiliary_bias is not None:
                context['auxiliary_bias_m'] = auxiliary_bias
            _, _, gamma = raw_hmm_energy(distance[ix], context, True, float(bias[context['anchor']]))
            posterior.extend(gamma)
        if posterior:
            posterior = np.asarray(posterior)
            diagnostic['raw_HMM_mean_component_posterior'] = posterior.mean(0).tolist()
            diagnostic['raw_HMM_high_background_fraction'] = float(np.mean(posterior[:, -1] > 0.5))
            diagnostic['raw_HMM_inferred_anchor_offsets_m'] = bias.tolist()
            if auxiliary_bias is not None:
                diagnostic['joint_RF_auxiliary_inferred_bias_m'] = auxiliary_bias.tolist()
            if self.prior.get('joint_RF_family_mixture_selected', False):
                diagnostic['joint_RF_geometry_family_posterior'] = [dict(anchor=int(context['anchor']), probability=float(context['last_auxiliary_family_posterior'])) for _, context in self.hmm]
        return (predictions, diagnostic)

class ForceRawHMMPath(ForceDualPath, RawHMMPath):
    pass

class JointMarginalPath(ForceRawHMMPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wifi_weight[:] = 0.0
        with np.load(args[0] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            self.pre = {k: z[k].copy() for k in z.files}
        self.device_prior = initial_prior(self.pre)
        self.radio_field = field(Path(self.prior['joint_runtime_base']))
        self.radio_parameters = np.asarray(self.prior['joint_marginal_RSSI_parameters'])
        self.radio_blocks = radio_blocks(self.t, self.pre['mask'], self.radio_parameters)
        self.radio_noise_logdet = sum((float(2 * np.log(np.diag(L)).sum()) for _, _, L in self.radio_blocks))

    def radio(self, q, mode):
        xy = self.B @ q
        mean, basis, dm, db = self.radio_field(xy, self.pre['motion'], True)
        zs = []
        As = []
        for ap, ix, L in self.radio_blocks:
            zs.append(solve_triangular(L, self.pre['rssi'][ix, ap] - mean[ix, ap], lower=True, check_finite=False))
            As.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
        z = np.concatenate(zs)
        A = np.concatenate(As)
        loss, eta, P, residual, component_weight = mixture_posterior(z, A, self.device_prior, self.radio_noise_logdet)
        gradient = np.zeros_like(xy)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            score = solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            uncertainty = solve_triangular(L.T, A[offset:end] @ P, lower=False, check_finite=False)
            gradient[ix] -= score[:, None] * dm[ix, ap]
            gradient[ix] += np.einsum('nd,ndz->nz', uncertainty - score[:, None] * eta[None], db[ix, ap])
            offset = end
        return (float(loss), self.B.T @ gradient, (eta, P, component_weight))

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        q, _ = self.unpack(vector)
        loss, g, _ = self.radio(q, mode)
        value += loss
        gradient[:2 * (self.n - 1)] += g[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        q, _ = self.unpack(self.last_selected_vector[:-1])
        mode = diagnostic['calibration_mode']
        _, _, (eta, P, component_weight) = self.radio(q, mode)
        xy = self.B @ q
        mean, basis = self.radio_field(xy, self.pre['motion'])
        cells = np.floor(xy / 2.0).astype(int)
        keys = [tuple(c) for c in cells]
        regions = sorted(set(keys))
        N = len(xy)
        _, initial_mean, initial_cov, _, _ = mixture_posterior(np.zeros(0), np.zeros((0, 6)), self.device_prior)
        means = np.broadcast_to(initial_mean, (N, 6)).copy()
        covariances = np.broadcast_to(initial_cov, (N, 6, 6)).copy()
        delta = np.zeros((N, 4))
        for cell in regions:
            inside = np.asarray([key == cell for key in keys])
            mask = self.pre['mask'].copy()
            mask[inside] = False
            z, A, ld, blocks = whiten(self.t, mask, self.pre['rssi'], mean, basis, self.radio_parameters)
            _, mu, cov, _, _ = mixture_posterior(z, A, self.device_prior, ld)
            selected = inside & (np.arange(N) >= 5)
            means[selected] = mu
            covariances[selected] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu
                score = cho_solve((L, True), residual, check_finite=False)
                kernel = np.exp(2 * self.radio_parameters[4 + ap]) * np.exp(-abs(self.t[selected, None] - self.t[None, ix]) / np.exp(self.radio_parameters[8]))
                delta[selected, ap] = kernel @ score
        self.calibration_forecasts = dict(device_mean=means, device_covariance=covariances, short_residual=delta, mode=0)
        diagnostic.update(joint_marginal_device_mean=eta.tolist(), joint_marginal_device_covariance=P.tolist(), initial_device_mean=initial_mean.tolist(), device_prior_PRE_index=5, initial_mixture_posterior_component_weights=component_weight.tolist(), all_eight_initial_device_hypotheses_marginalized_for_one_whole_path=True, device_uncertainty_marginalized=True, RSSI_FORECAST_timing='Offline spatial-region-cavity', rolling_spatial_regions=len(regions), initial_five_recognition_retained=True)
        return (predictions, diagnostic)

class StartConditionalObservations:

    def whiten(self, mean, basis, blocks):
        return super().whiten(mean + self.start_offset, basis - self.start_basis_offset, blocks)

    def initial_noise(self, rows, ap, eta):
        return self.start_gain[rows, ap] * (self.start_residual[ap] - self.start_basis[ap] @ eta)

def conditioned_model(original, pre, known_xy, radio, covariances, shared, population_mean, population_covariance, sigma, velocity_sigma):

    class Conditional(StartConditionalObservations, ObservationModel):
        pass
    model = object.__new__(Conditional)
    model.__dict__.update(original.__dict__)
    m, B, dm, db = radio(np.asarray(known_xy, float)[None], pre['motion'][:1], True)
    m, B, dm, db = (v[0] for v in (m, B, dm, db))
    mu = np.asarray(population_mean)
    P = np.asarray(population_covariance)
    gradient = dm + np.einsum('adz,d->az', db, mu)
    gradient_second = (gradient * gradient).sum(-1) + np.einsum('adz,de,aez->a', db, P, db)
    age = np.zeros(4)
    valid = pre['mask'][0].copy()
    for ap in range(4):
        index = np.where(model.S[ap][0] > 0)[0]
        times = np.asarray(model.query_metadata['raw_times_s_' + str(ap)])[index]
        age[ap] = max(abs(times), default=float('inf'))
        valid[ap] &= bool(len(index)) and age[ap] <= 0.5
    variance = np.array([c[0, 0] for c in covariances]) + sigma ** 2
    variance += velocity_sigma ** 2 * age.clip(0, 0.5) ** 2 * gradient_second
    start_residual = pre['rssi'][0] - m
    precision = np.linalg.inv(P)
    vector = precision @ mu
    if valid.any():
        precision += B[valid].T @ (B[valid] / variance[valid, None])
        vector += B[valid].T @ (start_residual[valid] / variance[valid])
    L = cholesky(precision, lower=True, check_finite=False)
    postcov = cho_solve((L, True), np.eye(6), check_finite=False)
    postmean = postcov @ vector
    gain = np.zeros((model.N, 4))
    conditional = []
    conditional_shared = []
    for ap in range(4):
        cross = covariances[ap][:, 0].copy() if valid[ap] else np.zeros(model.N)
        gain[:, ap] = cross / variance[ap]
        removed = np.outer(cross, cross) / variance[ap]
        conditional.append(covariances[ap] - removed)
        conditional_shared.append(shared[ap] - removed)
    model.start_gain = gain
    model.start_residual = start_residual
    model.start_basis = B
    model.start_variance = variance
    model.start_valid = valid
    model.start_offset = gain * start_residual[None]
    model.start_basis_offset = gain[..., None] * B[None]
    diagnostic = dict(first_scan_valid_APs=valid.tolist(), first_scan_max_contributor_age_s=[float(v) if np.isfinite(v) else None for v in age], first_scan_additional_noise_db=float(sigma), first_scan_motion_sigma_m_s=float(velocity_sigma), first_scan_covariance_conditioned=True, known_start_only=True, initial_radio_drift_approximation='Per-AP isotropic constant-velocity variance; source-fitted speed second moment')
    return (model, postmean, postcov, conditional, conditional_shared, diagnostic)

def mix(p, mean, covariance, weight):
    means = np.concatenate((p['mean'], np.asarray(mean)[None]))
    cov = np.concatenate((p['covariance'], np.asarray(covariance)[None]))
    precision = np.linalg.inv(cov)
    return dict(mean=means, covariance=cov, precision=precision, vector=np.einsum('kde,ke->kd', precision, means), logdet=np.linalg.slogdet(cov)[1], log_weight=np.r_[p['log_weight'] + math.log1p(-weight), math.log(weight)])

def configure(pre, configuration, known_xy, radio, observations, covariances, shared):
    model, mu, P, cov, shr, diagnostic = conditioned_model(observations, pre, known_xy, radio, covariances, shared, configuration['known_start_population_mean'], configuration['known_start_population_covariance'], configuration['known_start_extra_radio_noise_db'], configuration['known_start_velocity_sigma_m_s'])
    p = mix(initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0)), mu, P, configuration['known_start_population_mixture_weight'])
    return (model, p, cov, shr, diagnostic)

def start_features(initial):
    x, y = np.asarray(initial, float) / np.array([10.0, 5.0])
    return np.array([1.0, x, y, x * x, x * y, y * y])

def family_energy(velocity, speed, dt, old_parameters, old_temporal, speed_sigma, circular_context, probability, derivative=False):
    if not 0.0 <= probability <= 1.0:
        raise ValueError('Invalid circular motion family probability')
    old = velocity_energy(velocity, speed, dt, old_parameters, old_temporal, derivative, speed_sigma)
    new = directional_velocity_energy(velocity, speed, circular_context, derivative)
    losses = np.asarray([old[0] if derivative else old, new[0] if derivative else new])
    with np.errstate(divide='ignore'):
        logs = np.log([1 - probability, probability]) - losses
    normalizer = logsumexp(logs)
    weights = np.exp(logs - normalizer)
    circular_context['last_family_posterior'] = weights.copy()
    if not derivative:
        return -float(normalizer)
    return (-float(normalizer), weights[0] * old[1] + weights[1] * new[1], weights)
NODES, WEIGHTS = np.polynomial.hermite.hermgauss(9)
LOG_WEIGHTS = np.log(WEIGHTS) - 0.5 * math.log(math.pi)

def gain_components(mean, sigma):
    return np.exp(mean + math.sqrt(2) * sigma * NODES)

def gain_energy(velocity, speed, dt, parameters, temporal, mean, sigma, derivative=False):
    p = np.asarray(parameters, float)
    if len(p) != 9:
        raise ValueError('Motion gain experiment uses the frozen nine-parameter velocity-state model')
    gain = gain_components(mean, sigma)
    Q = len(gain)
    N = len(speed)
    transverse = math.exp(p[3])
    scaled = gain[:, None] * np.asarray(speed)[None]
    means = np.zeros((Q, N, 5, 2))
    means[:, :, 0, 0] = scaled
    means[:, :, 1, 0] = -scaled
    means[:, :, 2, 1] = transverse * scaled
    means[:, :, 3, 1] = -transverse * scaled
    sp, st, ss = np.exp(p[:3])
    scales = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])
    scales = scales[None, None] / np.sqrt(np.asarray(dt))[None, :, None, None]
    residual = np.asarray(velocity)[None, :, None] - means
    emission = (CONST - np.log(scales) - 2 * np.log1p(residual ** 2 / (3 * scales ** 2))).sum(-1)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / persistence) if temporal else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, :, None])
    alpha = np.empty_like(density)
    scale = np.empty((Q, N))
    alpha[:, 0] = prob[None] * density[:, 0]
    scale[:, 0] = alpha[:, 0].sum(-1)
    alpha[:, 0] /= scale[:, 0, None]
    for i in range(1, N):
        alpha[:, i] = alpha[:, i - 1] @ transition[i - 1] * density[:, i]
        scale[:, i] = alpha[:, i].sum(-1)
        alpha[:, i] /= scale[:, i, None]
    component_energy = -(np.log(scale) + maximum).sum(-1)
    logmass = LOG_WEIGHTS - component_energy
    normalizer = logsumexp(logmass)
    if not derivative:
        return -float(normalizer)
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[:, i] = density[:, i + 1] * backward[:, i + 1] @ transition[i].T / scale[:, i + 1, None]
    posterior = alpha * backward
    posterior /= posterior.sum(-1, keepdims=True)
    component_gradient = (posterior[:, :, :, None] * 4 * residual / (3 * scales ** 2 + residual ** 2)).sum(2)
    weight = np.exp(logmass - normalizer)
    gradient = (weight[:, None, None] * component_gradient).sum(0)
    return (-float(normalizer), gradient, weight)

def scale_posterior(z, A, prior, logdet, configuration, derivative=False):
    scales = np.asarray(configuration['RSSI_segment_scale_factors'], float)
    logprior = np.log(configuration['RSSI_segment_scale_probability'])
    values = [mixture_posterior(z / s, A / s, prior, logdet + 2 * len(z) * math.log(s)) for s in scales]
    scores = logprior - np.asarray([v[0] for v in values])
    normalizer = logsumexp(scores)
    weight = np.exp(scores - normalizer)
    means = np.asarray([v[1] for v in values])
    mean = weight @ means
    difference = means - mean
    covariance = np.einsum('k,kde->de', weight, np.asarray([v[2] for v in values]) + difference[:, :, None] * difference[:, None, :])
    components = weight @ np.asarray([v[4] for v in values])
    result = (-float(normalizer), mean, covariance, z - A @ mean, components)
    if not derivative:
        return result
    gz = np.zeros_like(z)
    gA = np.zeros_like(A)
    for w, s, v in zip(weight, scales, values):
        residual = z - A @ v[1]
        gz += w / s ** 2 * residual
        gA += w / s ** 2 * (A @ v[2] - residual[:, None] * v[1][None])
    return result + (dict(g_z=gz, g_A=gA, scale_posterior=weight),)

def unpack(parameters):
    p = np.asarray(parameters, float)
    return dict(gain=math.exp(p[0]), anchor_mean=p[1:5], alias=np.array([0.0, math.exp(p[5]), -math.exp(p[6])]), sigma=np.exp(p[7:11]), mad_coefficient=math.exp(p[11]), persistence_s=np.exp(p[12:16]), probability=softmax(np.r_[p[16:19], 0.0]))

def block_context(anchor, t, raw, scale, parameters, background, temporal=True, probabilities=None, conditional_bias_m=None):
    p = unpack(parameters)
    t = np.asarray(t, float)
    sigma = np.sqrt(p['sigma'][None, :3] ** 2 + (p['mad_coefficient'] * np.asarray(scale, float)[:, None]) ** 2)
    sigma = np.c_[sigma, np.full(len(t), p['sigma'][3])]
    log_transition = []
    probabilities = np.broadcast_to(p['probability'], (len(t), 4)) if probabilities is None else np.asarray(probabilities, float)
    for i, dt in enumerate(np.diff(t)):
        stick = np.exp(-max(float(dt), 1e-05) / p['persistence_s']) if temporal else np.zeros(4)
        transition = (1 - stick[:, None]) * probabilities[i + 1, None, :] + np.diag(stick)
        log_transition.append(np.log(transition.clip(1e-15)))
    offset = np.zeros(len(t)) if conditional_bias_m is None else np.asarray(conditional_bias_m, float)
    if offset.shape != (len(t),) or not np.isfinite(offset).all():
        raise ValueError('Invalid conditional range offset')
    return dict(anchor=anchor, raw=np.asarray(raw, float), sigma=sigma, conditional_bias_m=offset, log_transition=np.asarray(log_transition), transition=np.exp(np.asarray(log_transition)), log_initial=np.log(probabilities[0].clip(1e-15)), background=float(background[anchor]), parameters=p)

def raw_hmm_energy(distance, context, derivative=False, bias_m=0.0):
    if 'auxiliary_family_probability' in context:
        probability = float(context['auxiliary_family_probability'])
        full_context = dict(context)
        full_context.pop('auxiliary_family_probability')
        null_context = dict(full_context)
        null_context.pop('auxiliary_parameters')
        full = raw_hmm_energy(distance, full_context, derivative, bias_m)
        null = raw_hmm_energy(distance, null_context, derivative, bias_m)
        losses = np.asarray([full[0] if derivative else full, (null[0] if derivative else null) + context['auxiliary_null_loss']])
        logmass = np.log([probability, 1 - probability]) - losses
        normalizer = logsumexp(logmass)
        weight = np.exp(logmass - normalizer)
        context['last_auxiliary_family_posterior'] = float(weight[0])
        if not derivative:
            return -float(normalizer)
        return (-float(normalizer), weight[0] * full[1] + weight[1] * null[1], weight[0] * full[2] + weight[1] * null[2])
    p = context['parameters']
    raw = context['raw']
    sigma = context['sigma']
    e = raw[:, None] - p['gain'] * np.asarray(distance)[:, None] - p['anchor_mean'][context['anchor']] - p['alias'][None, :] - bias_m
    if 'conditional_bias_m' in context:
        e -= context['conditional_bias_m'][:, None]
    e = np.c_[e, raw - context['background']]
    emission = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(sigma) - 2 * np.log1p(e * e / (3 * sigma * sigma))
    auxiliary_gradient = None
    if 'auxiliary_parameters' in context:
        if derivative:
            extra, auxiliary_gradient = auxiliary_emission(distance, e, context, True)
        else:
            extra = auxiliary_emission(distance, e, context)
        emission += extra
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty_like(emission)
    normalizer = np.empty(len(e))
    alpha[0] = np.exp(context['log_initial']) * density[0]
    normalizer[0] = alpha[0].sum()
    alpha[0] /= normalizer[0]
    for i in range(1, len(e)):
        alpha[i] = alpha[i - 1] @ context['transition'][i - 1] * density[i]
        normalizer[i] = alpha[i].sum()
        alpha[i] /= normalizer[i]
    loss = -float((np.log(normalizer) + maximum).sum())
    if not derivative:
        return loss
    beta = np.ones_like(emission)
    for i in range(len(e) - 2, -1, -1):
        beta[i] = context['transition'][i] @ (density[i + 1] * beta[i + 1]) / normalizer[i + 1]
    posterior = alpha * beta
    posterior /= posterior.sum(-1, keepdims=True)
    score = 4 * e / (3 * sigma * sigma + e * e)
    gradient = -p['gain'] * (posterior[:, :3] * score[:, :3]).sum(-1)
    if auxiliary_gradient is not None:
        gradient += (posterior * auxiliary_gradient).sum(-1)
        context['last_auxiliary_bias_gradient'] = -(posterior[:, :, None] * context['last_auxiliary_score']).sum((0, 1))
    return (loss, gradient, posterior)

def augment(p, mean, covariance, weight, scale):
    if weight <= 0:
        return p
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    means = np.concatenate((p['mean'], np.broadcast_to(mean, (2, 6))))
    cov = np.concatenate((p['covariance'], np.asarray([covariance, covariance * 9]) * scale ** 2))
    precision = np.linalg.inv(cov)
    log_weight = np.r_[p['log_weight'] + math.log1p(-weight), np.log([0.9, 0.1]) + math.log(weight)]
    return dict(mean=means, covariance=cov, precision=precision, logdet=np.linalg.slogdet(cov)[1], vector=np.einsum('kde,ke->kd', precision, means), log_weight=log_weight)

def robust_initial_prior(pre, configuration):
    p = initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0))
    weight = float(configuration.get('initial_population_mixture_weight', 0.0))
    return augment(p, configuration['initial_population_mean'], configuration['initial_population_covariance'], weight, configuration['initial_population_covariance_scale']) if weight > 0 else p

def polish_axis(solver, modes):
    saved = solver.polish_parent_state
    state = saved['optimization_state']
    x = state['vector'].copy()
    if not solver.prior.get('joint_marginal_initial_mixture'):
        raise ValueError('Polishing currently requires the same eight-component joint marginal model')
    if not np.allclose(state['knots_s'], solver.knots, atol=1e-10):
        raise ValueError('Polish spline convention differs')
    mode = int(saved['diagnostic']['calibration_mode'])
    before, g0 = solver.axial_objective(x, mode)
    difference = float(abs(before - saved['diagnostic']['objective']))
    if difference > 1e-08 * (1 + abs(before)):
        raise ValueError('Polish model differs from the saved objective')
    result = minimize(solver.axial_objective, x, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=int(solver.prior.get('solver_polish_maxiter', 1000)), ftol=float(solver.prior.get('solver_polish_ftol', 1e-12)), gtol=float(solver.prior.get('solver_polish_gtol', 2e-05)), maxls=60))
    use_refined = bool(np.isfinite(result.fun) and result.fun <= before)
    selected = result.x if use_refined else x
    value, g = solver.axial_objective(selected, mode)
    q, _ = solver.unpack(selected[:-1])
    q0, _ = solver.unpack(x[:-1])
    xy = solver.B @ q
    change = np.linalg.norm(solver.B @ (q - q0), axis=-1)
    solver.last_selected_vector = selected.copy()
    predictions = [dict(p, xy_m=point.tolist(), inference_timing='OFFLINE_SAME_MODEL_REFINED_SOLVER', trajectory_calibration_mode=mode, later_position_GT_used=False) for p, point in zip(solver.predictions, xy)]
    diagnostic = dict(converged=bool(result.success), iterations=int(result.nit), objective=float(value), axis_degrees=float(np.rad2deg(selected[-1]) % 180), branch_objectives=[float(before), float(result.fun)], stopping_reason=str(result.message), optimization_starts=1, branch_stopping_reasons=[str(result.message)], calibration_mode=mode, later_GT_used=False, source_scene_motion_prior=True, phone_heading_used=False, initial_device_mixture_marginalized=True, optimization_native_seed_modes=modes, solver_polish_same_model=True, solver_polish_parent_objective=float(before), solver_polish_parent_objective_difference=difference, solver_polish_refined_selected=use_refined, solver_polish_original_position_gradient_inf=float(abs(g0[:-5]).max()), solver_polish_position_gradient_inf=float(abs(g[:-5]).max()), solver_polish_original_nuisance_gradient_inf=float(abs(g0[-5:]).max()), solver_polish_nuisance_gradient_inf=float(abs(g[-5:]).max()), solver_polish_mean_prediction_change_m=float(change.mean()), solver_polish_max_prediction_change_m=float(change.max()), solver_polish_fixed_initial_position=True, solver_polish_parameters_or_measurements_changed=False, solver_polish_ftol=solver.prior.get('solver_polish_ftol', 1e-12), solver_polish_gtol=solver.prior.get('solver_polish_gtol', 2e-05))
    return (predictions, diagnostic)

def density(points, centres, sigma, derivative=False):
    difference = np.asarray(points)[:, None] - np.asarray(centres)[None]
    squared = np.square(difference).sum(-1)
    denominator = 4 * sigma * sigma + squared
    lp = -math.log(2 * math.pi * sigma * sigma) - 3 * np.log1p(squared / (4 * sigma * sigma))
    total = logsumexp(lp, axis=-1)
    energy = math.log(len(centres)) - total
    if not derivative:
        return energy
    probability = np.exp(lp - total[:, None])
    gradient = (probability[..., None] * 6 * difference / denominator[..., None]).sum(1)
    return (energy, gradient)

class ObservationModel:

    def __init__(self, query_file, pre):
        with np.load(query_file, allow_pickle=False) as z:
            q = {k: z[k].copy() for k in z.files}
        if not np.array_equal(q['observation_ids'], pre['observation_ids']):
            raise ValueError('RSSI contributor IDs differ')
        self.pre = pre
        self.N = len(pre['rssi'])
        self.S = []
        self.raw_dt = []
        self.context = []
        self.raw_times = []
        self.ap = []
        self.bounds = [0]
        for ap in range(4):
            times = q['raw_times_s_' + str(ap)]
            indices = q['raw_index_' + str(ap)]
            S = np.zeros((self.N, len(times)))
            ctx = np.zeros(len(times), np.int64)
            for i in range(self.N):
                for j in range(2):
                    rid = int(indices[i, j])
                    if rid >= 0:
                        S[i, rid] += q['query_weights'][i, ap, j]
                        ctx[rid] = q['query_context_index'][i, ap, j]
            self.S.append(S)
            self.raw_dt.append(abs(times[:, None] - times[None, :]))
            self.raw_times.extend(times)
            self.context.extend(ctx)
            self.ap.extend([ap] * len(times))
            self.bounds.append(self.bounds[-1] + len(times))
        self.raw_times = np.asarray(self.raw_times)
        self.context = np.asarray(self.context)
        self.ap = np.asarray(self.ap)
        self.mask = pre['mask'].copy() & ~q['initial_support_raw_reused']
        self.mask[:5] = False
        self.shared = [S @ S.T for S in self.S]
        self.query_metadata = q

    def freeze_field_context(self, index=4):
        if len(self.pre['motion']) <= index:
            raise ValueError('Incomplete initial-five field context')
        self.fixed_context = np.broadcast_to(self.pre['motion'][index], (len(self.raw_times), 32)).copy()

    def raw_context(self):
        return self.fixed_context if hasattr(self, 'fixed_context') else self.pre['motion'][self.context]

    def lock_device_basis(self, known_initial_xy, radio_field):
        points = np.broadcast_to(np.asarray(known_initial_xy, float), (len(self.raw_times), 2))
        _, self.fixed_device_basis = radio_field.ap_query(points, self.raw_context(), self.ap, False)

    def temper_device_spatial_change(self, known_initial_xy, radio_field, strength):
        if not 0 <= strength <= 1:
            raise ValueError('Device spatial-change strength must be between zero and one')
        points = np.broadcast_to(np.asarray(known_initial_xy, float), (len(self.raw_times), 2))
        _, self.initial_location_basis = radio_field.ap_query(points, self.raw_context(), self.ap, False)
        self.device_spatial_strength = float(strength)

    def geometry(self, raw_xy, radio_field, derivative=False):
        if hasattr(self, 'fixed_device_basis'):
            if derivative:
                raw_mean, raw_dm = radio_field.source_ap(raw_xy, self.ap, True)
            else:
                raw_mean = radio_field.source_ap(raw_xy, self.ap)
                raw_dm = None
            raw_basis = self.fixed_device_basis.copy()
            raw_basis[:, 1] = raw_mean / 50
            raw_db = np.zeros((len(raw_xy), 6, 2)) if derivative else None
            if derivative:
                raw_db[:, 1] = raw_dm / 50
        elif hasattr(radio_field, 'ap_query'):
            outputs = radio_field.ap_query(raw_xy, self.raw_context(), self.ap, derivative)
            raw_mean, raw_basis = outputs[:2]
            raw_dm, raw_db = outputs[2:4] if derivative else (None, None)
        else:
            outputs = radio_field(raw_xy, self.raw_context(), derivative)
            index = np.arange(len(raw_xy))
            raw_mean = outputs[0][index, self.ap]
            raw_basis = outputs[1][index, self.ap]
            raw_dm, raw_db = (outputs[2][index, self.ap], outputs[3][index, self.ap]) if derivative else (None, None)
        if hasattr(self, 'initial_location_basis'):
            alpha = self.device_spatial_strength
            raw_basis = raw_basis.copy()
            raw_basis[:, 2:] = self.initial_location_basis[:, 2:] + alpha * (raw_basis[:, 2:] - self.initial_location_basis[:, 2:])
            if derivative:
                raw_db = raw_db.copy()
                raw_db[:, 2:] *= alpha
        mean = np.zeros((self.N, 4))
        basis = np.zeros((self.N, 4, 6))
        for ap in range(4):
            sl = slice(self.bounds[ap], self.bounds[ap + 1])
            mean[:, ap] = self.S[ap] @ raw_mean[sl]
            basis[:, ap] = self.S[ap] @ raw_basis[sl]
        if not derivative:
            return (mean, basis)
        return (mean, basis, raw_dm, raw_db)

    def noise(self, parameters):
        tau = math.exp(parameters[8])
        cov = []
        shared = []
        derivatives = []
        for ap in range(4):
            S = self.S[ap]
            K = np.exp(-self.raw_dt[ap] / tau)
            prop = S @ K @ S.T
            d_tau = S @ (K * self.raw_dt[ap] / tau) @ S.T
            white = math.exp(2 * parameters[ap])
            colored = math.exp(2 * parameters[4 + ap])
            raw = math.exp(2 * parameters[9 + ap])
            correlated = raw * self.shared[ap] + colored * prop
            cov.append(correlated + np.eye(self.N) * (white + 0.1 ** 2))
            shared.append(correlated)
            derivatives.append((2 * white * np.eye(self.N), 2 * colored * prop, colored * d_tau, 2 * raw * self.shared[ap]))
        return (cov, shared, derivatives)

    def blocks(self, covariances, mask=None):
        available = self.mask if mask is None else self.mask & mask
        blocks = []
        for ap in range(4):
            ix = np.where(available[:, ap])[0]
            if len(ix):
                blocks.append((ap, ix, cholesky(covariances[ap][np.ix_(ix, ix)], lower=True, check_finite=False)))
        return blocks

    def whiten(self, mean, basis, blocks):
        zs = []
        As = []
        ld = 0.0
        for ap, ix, L in blocks:
            zs.append(solve_triangular(L, self.pre['rssi'][ix, ap] - mean[ix, ap], lower=True, check_finite=False))
            As.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
            ld += 2 * np.log(np.diag(L)).sum()
        return (np.concatenate(zs), np.concatenate(As), float(ld)) if zs else (np.zeros(0), np.zeros((0, 6)), 0.0)

    def pullback(self, g_mean, g_basis, raw_dm, raw_db, raw_spline):
        gm = []
        gb = []
        for ap in range(4):
            gm.append(self.S[ap].T @ g_mean[:, ap])
            gb.append(self.S[ap].T @ g_basis[:, ap])
        gm = np.concatenate(gm)
        gb = np.concatenate(gb)
        points = gm[:, None] * raw_dm + np.einsum('nd,ndz->nz', gb, raw_db)
        return raw_spline.T @ points

class TemporalJointPath(JointMarginalPath):

    def __init__(self, *args, **kwargs):
        ForceRawHMMPath.__init__(self, *args, **kwargs)
        self.wifi_weight[:] = 0.0
        with np.load(args[0] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            self.pre = {k: z[k].copy() for k in z.files}
        if self.prior.get('field_context_override_root'):
            with np.load(Path(self.prior['field_context_override_root']) / args[0].parent.name / args[0].name / 'field_context.npz', allow_pickle=False) as z:
                if not np.array_equal(z['observation_ids'], self.pre['observation_ids']) or not np.array_equal(z['times_ns'], self.times_ns):
                    raise ValueError('Rolling field context override IDs or times differ')
                self.pre['motion'] = z['motion'].copy()
        self.device_prior = robust_initial_prior(self.pre, self.prior)
        self.radio_field = field(Path(self.prior['joint_runtime_base']), self.prior)
        if self.prior.get('RSSI_AP_scale_marginalization', False) and any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'initial_position_device_dependency_strength', 'RSSI_segment_scale_marginalization'])):
            raise ValueError('Evaluate AP noise families separately from alternative initial/noise conditioning')
        self.observations = ObservationModel(Path(self.prior['temporal_RSSI_query_root']) / args[0].parent.name / args[0].name / 'radio_queries.npz', self.pre)
        if self.prior.get('radio_field_context_mode') == 'initial_five_final':
            self.observations.freeze_field_context()
        if self.prior.get('geometry_locked_device_basis', False):
            self.observations.lock_device_basis(self.s['initial_position_xy_m'], self.radio_field)
        if self.prior.get('device_spatial_basis_strength', 1.0) < 1:
            self.observations.temper_device_spatial_change(self.s['initial_position_xy_m'], self.radio_field, self.prior['device_spatial_basis_strength'])
        self.raw_spline = self.spline(self.observations.raw_times.clip(0, self.knots[-1]))
        self.radio_covariances, self.radio_shared_covariances, _ = self.observations.noise(np.asarray(self.prior['temporal_RSSI_parameters']))
        if self.prior.get('known_start_radio_conditioning', False):
            self.observations, self.device_prior, self.radio_covariances, self.radio_shared_covariances, self.known_start_diagnostic = configure(self.pre, self.prior, self.s['initial_position_xy_m'], self.radio_field, self.observations, self.radio_covariances, self.radio_shared_covariances)
        self.radio_blocks = self.observations.blocks(self.radio_covariances)
        if self.prior.get('initial_position_device_dependency_strength', 0.0) > 0:
            if self.prior.get('known_start_radio_conditioning') or self.prior.get('RSSI_segment_scale_marginalization') or self.prior.get('initial_population_mixture_weight', 0.0) > 0:
                raise ValueError('Evaluate the initial-dependency mechanism before combining alternative initial/scale priors')
            self.initial_dependency = InitialDependency(args[0], self.pre, self.prior)
            self.initial_dependency_basis = self.spline(self.t[5])
        if self.prior.get('frozen_M3_spatial_head_scale', 0.0) > 0:
            with np.load(Path(self.prior['frozen_M3_spatial_head_root']) / args[0].parent.name / args[0].name / 'head.npz', allow_pickle=False) as z:
                if not np.array_equal(z['observation_ids'], self.pre['observation_ids']) or not np.array_equal(z['times_ns'], self.times_ns):
                    raise ValueError('Frozen M3 head observation IDs or times differ')
                self.M3_head_grid = z['grid_xy'].copy()
                self.M3_head_potential = z['potential'].copy()
                self.M3_head_weight = z['weight'].copy()

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        strength = self.prior.get('source_spatial_support_strength', 0.0)
        if strength > 0:
            q, _ = self.unpack(vector)
            B = self.spline(self.mid)
            energy, g = density(B @ q, np.asarray(self.prior['source_spatial_support_centres_m']), self.prior['source_spatial_support_sigma_m'], True)
            weight = np.diff(self.knots) * strength
            value += float(energy @ weight)
            dq = B.T @ (g * weight[:, None])
            gradient[:2 * (self.n - 1)] += dq[1:].ravel()
        strength = self.prior.get('frozen_M3_spatial_head_scale', 0.0)
        if strength > 0:
            q, _ = self.unpack(vector)
            energy, g = potential_energy(self.B @ q, self.M3_head_grid, self.M3_head_potential, self.M3_head_weight, self.prior['frozen_M3_spatial_head_bandwidth_m'], True)
            value += strength * energy
            dq = self.B.T @ g
            gradient[:2 * (self.n - 1)] += strength * dq[1:].ravel()
        return (value, gradient)

    def radio(self, q, mode):
        model = self.observations
        mean, basis, dm, db = model.geometry(self.raw_spline @ q, self.radio_field, True)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        if hasattr(self, 'initial_dependency'):
            self.device_prior = self.initial_dependency.at(self.initial_dependency_basis @ q)
        if self.prior.get('RSSI_segment_scale_marginalization', False) or self.prior.get('RSSI_AP_scale_marginalization', False):
            if self.prior.get('RSSI_segment_scale_marginalization', False) and self.prior.get('RSSI_AP_scale_marginalization', False):
                raise ValueError('Do not count both global and per-AP noise mixtures')
            if self.prior.get('RSSI_AP_scale_marginalization', False):
                loss, eta, P, residual, components, derivatives = ap_posterior(z, A, self.device_prior, ld, self.radio_blocks, self.prior, True)
            else:
                loss, eta, P, residual, components, derivatives = scale_posterior(z, A, self.device_prior, ld, self.prior, True)
            gm = np.zeros_like(mean)
            gb = np.zeros_like(basis)
            offset = 0
            for ap, ix, L in self.radio_blocks:
                end = offset + len(ix)
                gm[ix, ap] = -solve_triangular(L.T, derivatives['g_z'][offset:end], lower=False, check_finite=False)
                gb[ix, ap] = solve_triangular(L.T, derivatives['g_A'][offset:end], lower=False, check_finite=False)
                offset = end
            return (loss, model.pullback(gm, gb, dm, db, self.raw_spline), (eta, P, components))
        if hasattr(self, 'initial_dependency'):
            loss, eta, P, residual, components, g_initial = self.initial_dependency.posterior(z, A, self.device_prior, ld, True)
        else:
            loss, eta, P, residual, components = mixture_posterior(z, A, self.device_prior, ld)
        gm = np.zeros_like(mean)
        gb = np.zeros_like(basis)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            score = solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            uncertainty = solve_triangular(L.T, A[offset:end] @ P, lower=False, check_finite=False)
            gm[ix, ap] = -score
            gb[ix, ap] = uncertainty - score[:, None] * eta[None]
            offset = end
        gradient = model.pullback(gm, gb, dm, db, self.raw_spline)
        if hasattr(self, 'initial_dependency'):
            gradient += self.initial_dependency_basis[:, None] * g_initial[None]
        return (float(loss), gradient, (eta, P, components))

    def device_posterior(self, z, A, ld=0.0, blocks=None):
        if self.prior.get('RSSI_AP_scale_marginalization', False):
            if blocks is None and len(z):
                raise ValueError('Per-AP posterior requires the actual conditioning block lengths')
            return ap_posterior(z, A, self.device_prior, ld, blocks or [], self.prior)
        if self.prior.get('RSSI_segment_scale_marginalization', False):
            return scale_posterior(z, A, self.device_prior, ld, self.prior)
        return mixture_posterior(z, A, self.device_prior, ld)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = ForceRawHMMPath.solve_axis(self, alternative, modes)
        return self.calibrate_solution(predictions, diagnostic)

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        model = self.observations
        raw_xy = self.raw_spline @ q
        if hasattr(self, 'initial_dependency'):
            prefix_xy = self.initial_dependency_basis @ q
            self.device_prior = self.initial_dependency.at(prefix_xy)
            diagnostic.update(initial_position_device_dependency_strength=self.initial_dependency.strength, initial_dependency_predicted_prefix_xy_m=prefix_xy.tolist(), initial_dependency_timestamp_s=float(self.t[5]), initial_dependency_target_positions_used=False, initial_dependency_native_position_marginal_used=False, initial_dependency_recounts_initial_RSSI=False, initial_dependency_conditional_mixture_approximation=True)
        mean, basis = model.geometry(raw_xy, self.radio_field)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        _, eta, P, _, component_weight = self.device_posterior(z, A, ld, self.radio_blocks)
        if self.prior.get('RSSI_AP_scale_marginalization', False):
            details = ap_posterior(z, A, self.device_prior, ld, self.radio_blocks, self.prior, True)[-1]
            diagnostic.update(RSSI_AP_high_noise_posterior=details['AP_high_noise_posterior'].tolist(), RSSI_AP_noise_family_posterior=details['family_posterior'].tolist(), RSSI_AP_noise_families_integrated=True, RSSI_AP_noise_normalizers_retained=True, RSSI_AP_noise_scope='One scale per AP for the complete offline segment; no AP observations deleted')
        if self.prior.get('RSSI_segment_scale_marginalization', False):
            full_scale = scale_posterior(z, A, self.device_prior, ld, self.prior, True)[-1]['scale_posterior']
            diagnostic.update(RSSI_segment_scale_posterior=full_scale.tolist(), RSSI_segment_scale_factors=self.prior['RSSI_segment_scale_factors'])
        _, initial_mu, initial_P, _, _ = self.device_posterior(np.zeros(0), np.zeros((0, 6)))
        N = len(mean)
        means = np.broadcast_to(initial_mu, (N, 6)).copy()
        covariances = np.broadcast_to(initial_P, (N, 6, 6)).copy()
        delta = np.zeros((N, 4))
        raw_cells = [tuple(cell) for cell in np.floor(raw_xy / 2.0).astype(int)]
        now_cells = np.floor(self.B @ q / 2.0).astype(int)
        groups = {}
        for i in range(5, N):
            regions = {tuple(now_cells[i])}
            for ap in range(4):
                for j in np.where(model.S[ap][i] > 0)[0]:
                    regions.add(raw_cells[model.bounds[ap] + j])
            groups.setdefault(tuple(sorted(regions)), []).append(i)
        for regions, rows in groups.items():
            mask = self.pre['mask'].copy()
            region_set = set(regions)
            for ap in range(4):
                bad = np.asarray([raw_cells[j] in region_set for j in range(model.bounds[ap], model.bounds[ap + 1])])
                mask[model.S[ap][:, bad].sum(-1) > 0, ap] = False
            blocks = model.blocks(self.radio_covariances, mask)
            z, A, ld = model.whiten(mean, basis, blocks)
            _, mu, cov, _, _ = self.device_posterior(z, A, ld, blocks)
            means[rows] = mu
            covariances[rows] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu
                if hasattr(model, 'initial_noise'):
                    residual -= model.initial_noise(ix, ap, mu)
                score = cho_solve((L, True), residual, check_finite=False)
                delta[rows, ap] = self.radio_shared_covariances[ap][np.ix_(rows, ix)] @ score
            if hasattr(model, 'initial_noise'):
                for ap in range(4):
                    delta[rows, ap] += model.initial_noise(rows, ap, mu)
        self.calibration_forecasts = dict(device_mean=means, device_covariance=covariances, short_residual=delta, mode=0)
        if self.prior.get('field_context_override_root'):
            self.calibration_forecasts['field_motion'] = self.pre['motion'].copy()
            diagnostic['field_context_scope'] = 'Source-trained rolling context with immutable original initial-five device prior'
        if hasattr(model, 'fixed_context'):
            self.calibration_forecasts['field_motion'] = np.broadcast_to(self.pre['motion'][4], self.pre['motion'].shape).copy()
            diagnostic['field_context_timing'] = 'Immutable initial-five context; subsequent RSSI updates device posterior and propagation nuisance'
        diagnostic.update(joint_marginal_device_mean=eta.tolist(), joint_marginal_device_covariance=P.tolist(), initial_mixture_posterior_component_weights=component_weight.tolist(), device_prior_PRE_index=5, measured_RSSI_contributor_clock_used=True, shared_raw_RSSI_noise_modeled=True, rolling_initial_raw_sample_overlap_excluded=True, region_cavity_groups=len(groups), RSSI_FORECAST_timing='Offline raw-contributor spatial-region-cavity', known_phone_heading_used=False)
        if hasattr(model, 'initial_location_basis'):
            diagnostic.update(device_spatial_basis_strength=model.device_spatial_strength, device_spatial_scope='Initial-XY nonlinear AP correction retained with live measured context; only its additional spatial variation is tempered; common field gain remains free')
        if len(component_weight) > 8:
            diagnostic.update(initial_population_posterior_mass=float(component_weight[8:].sum()), initial_population_initial_eight_components_retained=True, source_field_common_gain_mean=float(1 + eta[1] / 50), source_field_common_gain_std=float(math.sqrt(max(P[1, 1], 0)) / 50))
        if hasattr(self, 'known_start_diagnostic'):
            diagnostic['known_start_radio_calibration'] = self.known_start_diagnostic
        if self.prior.get('geometry_locked_device_basis', False):
            diagnostic.update(device_basis_position_query='Known initial XY; measured causal context still varies', spatial_device_correction='One common source-field gain and context-dependent AP offsets', source_field_common_gain_mean=float(1 + eta[1] / 50), source_field_common_gain_std=float(math.sqrt(max(P[1, 1], 0)) / 50), position_dependent_learned_device_basis_frozen=True)
        return (predictions, diagnostic)
CONST = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi)

def velocity_energy(velocity, speed, dt, parameters, temporal=True, derivative=False, speed_sigma=None):
    p = np.asarray(parameters)
    N = len(speed)
    gain = math.exp(p[3])
    means = np.zeros((N, 5, 2))
    means[:, 0, 0] = speed
    means[:, 1, 0] = -speed
    means[:, 2, 1] = gain * speed
    means[:, 3, 1] = -gain * speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])
    sigma = sigma[None] / np.sqrt(np.asarray(dt))[:, None, None]
    if len(p) > 9:
        if speed_sigma is None:
            raise ValueError('Missing measured-context speed uncertainty')
        extra = np.zeros_like(sigma)
        variance = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    residual = np.asarray(velocity)[:, None, :] - means
    emission = (CONST - np.log(sigma) - 2 * np.log1p(residual * residual / (3 * sigma * sigma))).sum(-1)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / persistence) if temporal else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty_like(density)
    scale = np.empty(N)
    alpha[0] = prob * density[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ transition[i - 1] * density[i]
        scale[i] = alpha[i].sum()
        alpha[i] /= scale[i]
    energy = -float((np.log(scale) + maximum).sum())
    if not derivative:
        return energy
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[i] = transition[i] @ (density[i + 1] * backward[i + 1]) / scale[i + 1]
    posterior = alpha * backward
    posterior /= posterior.sum(-1, keepdims=True)
    gradient = (posterior[:, :, None] * 4 * residual / (3 * sigma * sigma + residual * residual)).sum(1)
    return (energy, gradient, posterior)
