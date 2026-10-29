import torch
import numpy as np
from torch import nn
import math
from torch.nn import functional as F

def configure_filter(configuration, parameters, anchors):
    result = dict(configuration)
    if parameters:
        result['cs_temporal_noise'] = {name: dict(tau_s=float(parameters['tau_s']), stationary_sigma_m=float(parameters['stationary_sigma_m'])) for name in anchors}
    return result

def replace_device(mean, covariance, new_mean, new_covariance):
    dimension = mean.shape[-1]
    if dimension == 15:
        old = stabilize(covariance[..., 9:, 9:])
        old_root = torch.linalg.cholesky(old)
        new_root = torch.linalg.cholesky(stabilize(new_covariance))
        transport = torch.linalg.solve_triangular(old_root.transpose(-1, -2), new_root.transpose(-1, -2), upper=True)
        cross = covariance[..., :9, 9:] @ transport
        full = torch.cat((torch.cat((covariance[..., :9, :9], cross), -1), torch.cat((cross.transpose(-1, -2), new_covariance), -1)), -2)
        return (torch.cat((mean[..., :9], new_mean), -1), stabilize(full))
    outside = list(range(9)) + list(range(15, dimension))
    order = torch.tensor(outside + list(range(9, 15)), device=mean.device)
    reverse = order.argsort()
    count = len(outside)
    permuted = covariance.index_select(-2, order).index_select(-1, order)
    old_root = torch.linalg.cholesky(stabilize(permuted[..., count:, count:]))
    new_root = torch.linalg.cholesky(stabilize(new_covariance))
    transport = torch.linalg.solve_triangular(old_root.transpose(-1, -2), new_root.transpose(-1, -2), upper=True)
    cross = permuted[..., :count, count:] @ transport
    full = torch.cat((torch.cat((permuted[..., :count, :count], cross), -1), torch.cat((cross.transpose(-1, -2), new_covariance), -1)), -2)
    updated = torch.cat((mean.index_select(-1, order)[..., :count], new_mean), -1)
    return (updated.index_select(-1, reverse), stabilize(full.index_select(-2, reverse).index_select(-1, reverse)))

def radio_tokens(rssi, mask, times):
    previous = np.r_[np.zeros_like(rssi[:1]), rssi[:-1]]
    prev_mask = np.r_[np.zeros_like(mask[:1]), mask[:-1]]
    dt = np.r_[0.0, np.diff(times)]
    delta = np.where(mask & prev_mask, rssi - previous, 0.0)
    age = np.zeros_like(rssi)
    last = np.full(rssi.shape[-1], -1000000000.0)
    for i in range(len(rssi)):
        age[i] = np.minimum(30.0, times[i] - last)
        last = np.where(mask[i], times[i], last)
    token = np.stack((rssi / 100, delta / 20, abs(delta) / 20, np.broadcast_to(dt[:, None] / 5, rssi.shape), age / 10, mask, prev_mask, np.broadcast_to(((dt > 0) & (dt < 10))[:, None], rssi.shape)), -1)
    return token.astype(np.float32)

class InitialEvidencePosterior(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.ap = nn.Sequential(nn.Linear(21, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(60, 32), nn.SiLU(), nn.Linear(32, 12))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, rssi, mask, field, noise_variance, mean, covariance, ap_embedding):
        basis = field['basis']
        valid = mask.float().expand_as(field['mean'])
        residual = rssi - field['mean']
        posterior_residual = residual - (basis @ mean[..., None]).squeeze(-1)
        scalars = torch.cat((torch.stack((residual / 10, posterior_residual / 10, noise_variance.clamp_min(0.25).log() / 5), -1), basis, field['gradient'] / 10, (field['mean'] / 50)[..., None], valid[..., None]), -1)
        h = self.ap(torch.cat((scalars, ap_embedding.expand(*scalars.shape[:-1], 8)), -1))
        weight = valid[..., None]
        den = weight.sum(-2).clamp_min(1)
        average = (h * weight).sum(-2) / den
        spread = (((h - average[..., None, :]).square() * weight).sum(-2) / den + 1e-06).sqrt()
        raw = self.readout(torch.cat((average, spread, mean / 10, covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).log1p() / 5), -1)).float()
        shift = 2 * raw[..., :6].tanh()
        scale = (0.6931471805599453 * raw[..., 6:].tanh()).exp() if self.mode == 'full' else torch.ones_like(shift)
        root = torch.linalg.cholesky(covariance)
        mapped_mean = mean + (root @ shift[..., None]).squeeze(-1)
        mapped_root = root * scale[..., None, :]
        mapped_covariance = mapped_root @ mapped_root.transpose(-1, -2)
        self.last_shift = shift.detach()
        self.last_scale = scale.detach()
        return (mapped_mean, mapped_covariance)

    def window(self, rssi, mask, field, noise_variance, mean, covariance, ap_embedding, policy):
        basis = field['basis']
        valid = mask.float().expand_as(field['mean'])
        residual = rssi - field['mean']
        posterior_residual = residual - torch.einsum('...tad,...d->...ta', basis, mean)
        scalars = torch.cat((torch.stack((residual / 10, posterior_residual / 10, noise_variance.clamp_min(0.25).log() / 5), -1), basis, field['gradient'] / 10, (field['mean'] / 50)[..., None], valid[..., None]), -1)
        inputs = torch.cat((scalars, ap_embedding.expand(*scalars.shape[:-1], 8)), -1)
        if policy == 'time_average':
            den = valid.sum(-2).clamp_min(1)
            inputs = (inputs * valid[..., None]).sum(-3) / den[..., None]
            valid = (valid.sum(-2) > 0).float()
        elif policy != 'evidence_set':
            raise ValueError('Unknown initial evidence pool')
        h = self.ap(inputs)
        if policy == 'evidence_set':
            h = h.flatten(-3, -2)
            valid = valid.flatten(-2, -1)
        weight = valid[..., None]
        den = weight.sum(-2).clamp_min(1)
        average = (h * weight).sum(-2) / den
        spread = (((h - average[..., None, :]).square() * weight).sum(-2) / den + 1e-06).sqrt()
        raw = self.readout(torch.cat((average, spread, mean / 10, covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).log1p() / 5), -1)).float()
        shift = 2 * raw[..., :6].tanh()
        scale = (0.6931471805599453 * raw[..., 6:].tanh()).exp() if self.mode == 'full' else torch.ones_like(shift)
        root = torch.linalg.cholesky(covariance)
        mapped_mean = mean + (root @ shift[..., None]).squeeze(-1)
        mapped_root = root * scale[..., None, :]
        mapped_covariance = mapped_root @ mapped_root.transpose(-1, -2)
        self.last_shift = shift.detach()
        self.last_scale = scale.detach()
        return (mapped_mean, mapped_covariance)

def window_posterior(calibration, rssi, mask, field, noise, policy, ap_embedding):
    information, vector, _, _, _ = calibration.block_information(field['basis'], rssi - field['mean'], noise, mask)
    count = mask.any(-1).sum(-1).clamp_min(1).float()
    information = information / count[..., None, None]
    vector = vector / count[..., None]
    mean, covariance = calibration.posterior(information, vector)
    posterior = calibration.initial_encoder.window(rssi, mask, field, noise.diagonal(dim1=-2, dim2=-1), mean, covariance, ap_embedding, policy)
    return (information, vector, posterior)

class SpatialIntegrationGrid(nn.Module):

    def __init__(self, field, cache, kind='legacy'):
        super().__init__()
        self.kind = kind
        self.cell_size = 0.6
        if kind not in ('legacy', 'cells', 'contrast32'):
            raise ValueError('Unknown spatial integration grid')
        if kind == 'legacy':
            xy = field.xy.detach().clone()
            area = torch.ones(len(xy))
            variance = torch.full((len(xy),), 0.6 ** 2 / 12)
            parent = torch.arange(len(xy))
            self.selected = []
        else:
            cells = torch.as_tensor(cache['cells'], dtype=torch.float32)
            centres = (cells + 0.5) * self.cell_size
            parent = torch.arange(len(cells))
            xy = centres
            area = torch.ones(len(xy))
            variance = torch.full((len(xy),), 0.6 ** 2 / 12)
            self.selected = []
            if kind == 'contrast32':
                offsets = torch.tensor([[-0.15, -0.15], [-0.15, 0.15], [0.15, -0.15], [0.15, 0.15]])
                children = centres[:, None] + offsets
                with torch.no_grad():
                    values = field.source(children)['mean']
                    noise = field.source(centres)['variance'].clamp_min(0.25)
                    contrast = (values.var(1, unbiased=False) / noise).sum(-1)
                    selected = contrast.topk(min(32, len(cells))).indices
                keep = torch.ones(len(cells), dtype=torch.bool)
                keep[selected] = False
                xy = torch.cat((centres[keep], children[selected].reshape(-1, 2)))
                area = torch.cat((torch.ones(int(keep.sum())), torch.full((4 * len(selected),), 0.25)))
                variance = torch.cat((torch.full((int(keep.sum()),), 0.6 ** 2 / 12), torch.full((4 * len(selected),), 0.3 ** 2 / 12)))
                parent = torch.cat((parent[keep], selected.repeat_interleave(4)))
                self.selected = selected.tolist()
        for name, value in [('xy', xy), ('log_area', area.log()), ('variance', variance), ('parent', parent)]:
            self.register_buffer(name, value, persistent=False)

    def description(self):
        return dict(kind=self.kind, nodes=len(self.xy), source_cell_size_m=self.cell_size, selected_source_cells=self.selected, total_relative_area=float(self.log_area.exp().sum()))

def basin_probability(points, radio_score, prior_mean, prior_covariance, posterior, policy, cell_size=0.6, max_peaks=4, log_prominence=3.0):
    if policy not in ('density_mode', 'mass_mode'):
        raise ValueError('Unknown spatial basin projection')
    n = points.shape[-2]
    prefix = posterior.shape[:-1]
    with torch.no_grad():
        location = points.detach()
        difference = location - prior_mean.detach().unsqueeze(-2)
        precision = torch.linalg.inv(prior_covariance.detach())
        density = radio_score.detach() - 0.5 * torch.einsum('...nd,...de,...ne->...n', difference, precision, difference)
        density = density.masked_fill(posterior.detach() <= 0, -torch.inf)
        distances = torch.cdist(location, location, p=2, compute_mode='donot_use_mm_for_euclid_dist').square()
        neighbours = distances.topk(min(9, n), largest=False, dim=-1).indices
        nearby = torch.gather(density.unsqueeze(-2).expand(*prefix, n, n), -1, neighbours)
        local_maximum = density >= nearby.max(-1).values
        remaining = density.masked_fill(~local_maximum, -torch.inf)
        highest = density.max(-1).values
        columns = []
        available = []
        for _ in range(max_peaks):
            value, index = remaining.max(-1)
            valid = torch.isfinite(value) & (value >= highest - log_prominence)
            column = distances.gather(-1, index[..., None, None].expand(*prefix, n, 1)).squeeze(-1)
            columns.append(column)
            available.append(valid)
            remaining = remaining.masked_fill(column < (1.5 * cell_size) ** 2, -torch.inf)
        valid = torch.stack(available, -1)
        distance_to_peaks = torch.stack(columns, -1).masked_fill(~valid.unsqueeze(-2), torch.inf)
        nearest = distance_to_peaks.argmin(-1)
        mass = torch.zeros(*prefix, max_peaks, device=posterior.device, dtype=posterior.dtype)
        mass.scatter_add_(-1, nearest, posterior.detach())
        chosen = torch.zeros_like(highest, dtype=torch.long) if policy == 'density_mode' else mass.argmax(-1)
        selected = nearest == chosen.unsqueeze(-1)
        selected_mass = (posterior.detach() * selected).sum(-1)
        informative = radio_score.detach().max(-1).values - radio_score.detach().min(-1).values > 1e-05
        use = informative & (selected_mass > 1e-08)
        count = torch.where(informative, valid.sum(-1), torch.ones_like(chosen))
    restricted = posterior * selected
    restricted = restricted / restricted.sum(-1, keepdim=True).clamp_min(1e-30)
    probability = torch.where(use[..., None], restricted, posterior)
    return (probability, dict(peak_count=count, selected_basin_probability=torch.where(use, selected_mass, torch.ones_like(selected_mass))))

def mlp(*sizes, last_zero=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    result = nn.Sequential(*layers)
    if last_zero:
        nn.init.zeros_(result[-1].weight)
        nn.init.zeros_(result[-1].bias)
    return result

def gaussian_log_prob(error, covariance, return_factor=False):
    covariance = covariance.float()
    error = error.float()
    chol = torch.linalg.cholesky(covariance)
    white = torch.linalg.solve_triangular(chol, error.unsqueeze(-1), upper=False)
    value = -0.5 * (white.square().sum((-2, -1)) + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) + error.shape[-1] * math.log(2 * math.pi))
    return (value, chol) if return_factor else value

def stabilize(matrix, floor=1e-05):
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    return matrix + floor * torch.eye(matrix.shape[-1], device=matrix.device)

class CausalTCN(nn.Module):

    def __init__(self, input_dim, hidden, dilations, depthwise=False):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.dilations = tuple(dilations)
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for d in dilations:
            conv = nn.Conv1d(hidden, hidden, 3, dilation=d, groups=hidden if depthwise else 1)
            self.convolutions.append(nn.Sequential(conv, nn.Conv1d(hidden, hidden, 1)) if depthwise else conv)
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, inputs):
        prefix, time = (inputs.shape[:-2], inputs.shape[-2])
        h = F.silu(self.input(inputs)).reshape(-1, time, self.input.out_features)
        for d, conv, norm in zip(self.dilations, self.convolutions, self.norms):
            update = conv(F.pad(h.transpose(1, 2), (2 * d, 0))).transpose(1, 2)
            h = norm(h + F.silu(update))
        return h.reshape(*prefix, time, self.input.out_features)

class PhysicalRadioField(nn.Module):

    def __init__(self, cache, geometry, orientation_basis=False, ap_basis_prior=False, fixed_ap_basis=False, source_model='idw', source_inducing=32, source_lengthscale=1.2):
        super().__init__()
        for key in ('xy', 'trend', 'residual', 'var', 'visibility', 'mask', 'band'):
            value = torch.as_tensor(cache[key])
            self.register_buffer(key, value.bool() if key == 'mask' else value.long() if key == 'band' else value.float())
        for key in ('centres_m', 'exponents'):
            self.register_buffer(key, torch.as_tensor(geometry[key], dtype=torch.float32))
        self.ap_embedding = nn.Embedding(len(cache['band']), 8)
        self.band_embedding = nn.Embedding(3, 2)
        self.basis_net = mlp(49, 64, 32, 4)
        self.map_noise_logit = nn.Parameter(torch.zeros(len(cache['band'])))
        self.orientation_basis = bool(orientation_basis)
        if self.orientation_basis:
            self.orientation_strength = nn.Parameter(torch.zeros(4))
        self.ap_basis_prior = bool(ap_basis_prior)
        self.fixed_ap_basis = bool(fixed_ap_basis)
        if source_model not in ('idw', 'physical_sparse_gp'):
            raise ValueError('Unknown source wireless field')
        self.source_model = source_model
        self.source_inducing = int(source_inducing)
        self.source_lengthscale = float(source_lengthscale)
        if source_model == 'physical_sparse_gp':
            self.source_gp = SparsePhysicalSource(cache, geometry, source_inducing, source_lengthscale)
        if self.ap_basis_prior:
            if len(cache['band']) != 4:
                raise ValueError('The current AP-bias basis is defined for the four-map-AP experiment')
            self.ap_basis_strength = nn.Parameter(torch.zeros(()), requires_grad=not self.fixed_ap_basis)

    def AP_strength(self):
        return torch.ones_like(self.ap_basis_strength) if self.fixed_ap_basis else self.ap_basis_strength.tanh()

    def source(self, points):
        if self.source_model == 'physical_sparse_gp':
            return self.source_gp(points, getattr(self, 'calibrated_geometry', False))
        offset = points[..., None, :] - self.xy
        d2 = offset.square().sum(-1).clamp_min(1e-05)
        weight = d2.reciprocal()[..., None] * self.mask
        denominator = weight.sum(-2).clamp_min(1e-08)
        fields = []
        for v in (self.trend, self.residual, self.var, self.visibility):
            fields.append((weight * v).sum(-2) / denominator)
        mean = fields[0] + fields[1]
        gradient = []
        bounded_source_gradient = getattr(self, 'finite_source_backward', False) and points.requires_grad
        calibrated_geometry = getattr(self, 'calibrated_geometry', False)
        need_components = bounded_source_gradient or calibrated_geometry
        component_gradients = []
        total = self.trend + self.residual
        for axis in range(2):
            shift = torch.zeros_like(points)
            shift[..., axis] = 0.15
            local = []
            components = []
            for query in (points + shift, points - shift):
                distance = (query[..., None, :] - self.xy).square().sum(-1).clamp_min(1e-05)
                w = distance.reciprocal()[..., None] * self.mask
                local.append((w * total).sum(-2) / w.sum(-2).clamp_min(1e-08))
                if need_components:
                    components.append(torch.stack([(w * value).sum(-2) / w.sum(-2).clamp_min(1e-08) for value in (self.trend, self.residual, self.var, self.visibility)], -1))
            gradient.append((local[0] - local[1]) / 0.3)
            if need_components:
                component_gradients.append((components[0] - components[1]) / 0.3)
        gradient = torch.stack(gradient, -1)
        if bounded_source_gradient:
            slopes = torch.stack(component_gradients, -1).detach()
            displacement = (points - points.detach())[..., None, :]
            fields = [value.detach() + (displacement * slopes[..., j, :]).sum(-1) for j, value in enumerate(fields)]
            mean = fields[0] + fields[1]
            gradient = gradient.detach()
        result = dict(trend=fields[0], residual=fields[1], variance=fields[2], visibility=fields[3].clamp(0.01, 0.99), mean=mean, gradient=gradient)
        if calibrated_geometry:
            components = torch.stack(component_gradients, -1)
            result['component_gradient'] = components.detach() if bounded_source_gradient else components
        return result

    def condition_heading(self, fields, points, heading):
        if not self.orientation_basis:
            return fields
        raw = fields.get('unoriented_basis', fields['basis'])
        toward = self.centres_m - points[..., None, :]
        angle = torch.atan2(toward[..., 1], toward[..., 0]) - heading[..., None]
        harmonics = torch.stack((angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos()), -1)
        modulation = 1 + self.orientation_strength.tanh() * harmonics
        response = raw[..., 2:]
        if self.ap_basis_prior:
            constant = self.AP_strength() * torch.eye(4, device=points.device)
            response = constant + (response - constant) * modulation
        else:
            response = response * modulation
        basis = torch.cat((raw[..., :2], response), -1)
        return dict(fields, basis=basis, unoriented_basis=raw)

    def forward(self, points, context, fields=None, heading=None):
        fields = self.source(points) if fields is None else fields
        a = self.ap_embedding.num_embeddings
        shape = points.shape[:-1]
        pos = (points / 10)[..., None, :].expand(*shape, a, 2)
        radius2 = ((points[..., None, :] - self.centres_m) ** 2).sum(-1) + 1
        log_distance = 0.5 * radius2.log10()
        ap = self.ap_embedding.weight.expand(*shape, a, 8)
        band = self.band_embedding(self.band).expand(*shape, a, 2)
        context = context[..., None, :].expand(*shape, a, 32)
        scalar = torch.stack((log_distance, fields['trend'] / 50, fields['residual'] / 10, fields['variance'].log1p() / 5, fields['visibility']), -1)
        inputs = torch.cat((pos, scalar, ap, band, context), -1)
        if getattr(self, 'calibrated_geometry', False):
            if self.orientation_basis:
                raise ValueError('Calibrated geometry currently preserves the selected non-harmonic field')
            component = fields['component_gradient']
            log_gradient = (points[..., None, :] - self.centres_m) / (math.log(10) * radius2[..., None])
            scalar_gradient = torch.stack((log_gradient, component[..., 0, :] / 50, component[..., 1, :] / 10, component[..., 2, :] / (1 + fields['variance'])[..., None] / 5, component[..., 3, :]), -2)
            position_gradient = (torch.eye(2, device=points.device) / 10).expand(*shape, a, 2, 2)
            feature_gradient = torch.cat((position_gradient, scalar_gradient, torch.zeros(*shape, a, 42, 2, device=points.device)), -2)
            learned, learned_gradient = mlp_with_directions(self.basis_net, inputs, feature_gradient)
        else:
            learned = torch.tanh(self.basis_net(inputs)).float()
        if self.ap_basis_prior:
            strength = self.AP_strength()
            learned = (1 - 0.85 * strength.abs()) * learned + strength * torch.eye(a, 4, device=points.device)
            if getattr(self, 'calibrated_geometry', False):
                learned_gradient = (1 - 0.85 * strength.abs()) * learned_gradient
        basis = torch.cat((torch.ones_like(log_distance)[..., None], (fields['mean'] / 50)[..., None], learned), -1)
        scale = 0.1 + 0.9 * self.map_noise_logit.sigmoid()
        result = dict(**fields, basis=basis, map_variance=fields['variance'] * scale + 0.25)
        if getattr(self, 'calibrated_geometry', False):
            result['basis_gradient'] = torch.cat((torch.zeros_like(fields['gradient'])[..., None, :], (fields['gradient'] / 50)[..., None, :], learned_gradient), -2)
        if self.orientation_basis:
            result['unoriented_basis'] = basis
        return self.condition_heading(result, points, heading) if heading is not None else result

class model_CounterfactualCalibration(nn.Module):

    def __init__(self, initial_posterior='none'):
        super().__init__()
        self.radio_encoder = CausalTCN(8, 24, (1, 2), depthwise=True)
        self.motion_encoder = CausalTCN(41, 32, (1, 2, 4))
        self.set_context = mlp(48, 24)
        self.radio_quality = mlp(80, 32, 2)
        self.innovation_gate = mlp(97, 64, 4, last_zero=True)
        self.cs_encoder = mlp(20, 32, 32)
        self.cs_quality = mlp(64, 32, 2)
        self.prior_mean = nn.Parameter(torch.zeros(6))
        self.prior_log_std = nn.Parameter(torch.tensor([6.0, 6.0, 3.0, 3.0, 3.0, 3.0]).log())
        self.register_buffer('timescales', torch.tensor([0.25, 0.75, 2.0]))
        self.position_difference_context = False
        self.initial_posterior_mode = initial_posterior
        if initial_posterior != 'none':
            self.initial_encoder = InitialEvidencePosterior(initial_posterior)

    def prior(self, shape):
        mean = self.prior_mean.expand(*shape, 6)
        variance = self.prior_log_std.clamp(-2.0, 3.0).exp().square()
        covariance = torch.diag_embed(variance.expand(*shape, 6))
        return (mean, covariance)

    def encode_history(self, radio_history, motion_history, initial_history, mask):
        radio = self.radio_encoder(radio_history)[..., -1, :]
        spatial_radio = radio
        if self.position_difference_context:
            valid = radio_history[..., 5]
            centre = (radio_history[..., 0] * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            relative = radio_history.clone()
            relative[..., 0] = (radio_history[..., 0] - centre[..., None]) * valid * 5
            spatial_radio = self.radio_encoder(relative)[..., -1, :]
        initial = self.radio_encoder(initial_history)[..., -1, :]
        motion = torch.tanh(self.motion_encoder(motion_history)[..., -1, :])
        spatial_motion = motion
        if self.position_difference_context:
            relative_motion = motion_history.clone()
            relative_motion[..., 25::4] = 0
            spatial_motion = torch.tanh(self.motion_encoder(relative_motion)[..., -1, :])
        weights = mask.float()[..., None]
        den = weights.sum(-2).clamp_min(1.0)
        average = (radio * weights).sum(-2) / den
        variance = ((radio - average[..., None, :]).square() * weights).sum(-2) / den
        global_context = self.set_context(torch.cat((average, (variance + 1e-06).sqrt()), -1))
        global_context = global_context[:, None].expand_as(radio)
        raw = self.radio_quality(torch.cat((radio, global_context, motion[:, None].expand(-1, radio.shape[1], -1)), -1)).float()
        sigma = 0.5 + F.softplus(raw[..., 0])
        pi = 0.001 + 0.998 * raw[..., 1].sigmoid()
        return dict(radio=radio, spatial_radio=spatial_radio, initial=initial, motion=motion, spatial_motion=spatial_motion, sigma=sigma, outlier=pi)

    def cs_parameters(self, features, motion):
        h = self.cs_encoder(features)
        raw = self.cs_quality(torch.cat((h, motion), -1)).float()
        return (0.1 + getattr(self, 'cs_white_sigma_scale', 1.0) * F.softplus(raw[..., 0]), 0.001 + 0.998 * raw[..., 1].sigmoid())

    def short_residual(self, encoded, ap_embedding, innovations, ages, valid, pose_confidence, spatial_weight, rotation_weight, scan_gap, cs_presence):
        w = torch.exp(-ages[..., None] / self.timescales)
        w = w * spatial_weight[..., None] * rotation_weight[..., None]
        w = w[:, :, None] * valid[..., None]
        mass = w.sum(-2)
        average = (w * innovations[..., None]).sum(-2) / mass.clamp_min(1e-06)
        support = 1 - (-mass).exp()
        b, k, a = average.shape[:3]
        fixed = encoded['initial'][:, None].expand(b, k, a, 24)
        recent = encoded['radio'][:, None].expand(b, k, a, 24)
        motion = encoded['motion'][:, None, None].expand(b, k, a, 32)
        ap = ap_embedding.expand(b, k, a, 8)
        shared = torch.stack((pose_confidence, scan_gap[:, None].expand(b, k) / 5, cs_presence[:, None].expand(b, k)), -1)
        shared = shared[:, :, None].expand(b, k, a, 3)
        raw = self.innovation_gate(torch.cat((fixed, recent, motion, ap, average / 10, support, shared), -1)).float()
        mix, gain = (raw[..., :3].softmax(-1), 2 * raw[..., 3].tanh())
        correction = gain * (mix * average * support).sum(-1) * pose_confidence[..., None]
        return (8 * (correction / 8).tanh(), (mix * support).sum(-1))

    def posterior(self, information, vector, prior_mean=None, prior_covariance=None):
        shape = vector.shape[:-1]
        if prior_mean is None:
            mean0, covariance0 = self.prior(shape)
            precision0 = torch.diag_embed(covariance0.diagonal(dim1=-2, dim2=-1).reciprocal())
        else:
            mean0, covariance0 = (prior_mean, prior_covariance)
            precision0 = torch.linalg.inv(stabilize(covariance0))
        precision = stabilize(precision0 + information)
        vector = vector + (precision0 @ mean0[..., None]).squeeze(-1)
        chol = torch.linalg.cholesky(precision)
        mean = torch.cholesky_solve(vector[..., None], chol).squeeze(-1)
        covariance = torch.cholesky_inverse(chol)
        return (mean, covariance)

    def cavity(self, information, vector, region_mask, retired_information=None, retired_vector=None):
        selected = region_mask[..., None, None, None]
        outside_information = (information * selected).sum(1)
        outside_vector = (vector * region_mask[..., None, None]).sum(1)
        if retired_information is not None:
            outside_information = outside_information + retired_information
            outside_vector = outside_vector + retired_vector
        return self.posterior(outside_information, outside_vector)

    def predictive_score(self, rssi, mask, source_mean, basis, noise, device_mean, device_covariance):
        prediction = source_mean + (basis @ device_mean[..., None]).squeeze(-1)
        covariance = basis @ device_covariance @ basis.transpose(-1, -2)
        covariance = covariance + torch.diag_embed(noise)
        valid = mask.float()
        covariance = covariance * valid[..., :, None] * valid[..., None, :]
        covariance = covariance + torch.diag_embed(1 - valid)
        score = gaussian_log_prob((rssi - prediction) * valid, stabilize(covariance))
        score = score + 0.5 * (1 - valid).sum(-1) * math.log(2 * math.pi)
        return (score, prediction, covariance)

    def predictive_diagonal_score(self, rssi, mask, source_mean, basis, noise, device_mean, device_covariance):
        prediction = source_mean + (basis @ device_mean[..., None]).squeeze(-1)
        error = rssi - prediction
        inv = mask.float() / noise.clamp_min(0.25)
        prior_precision = torch.linalg.inv(stabilize(device_covariance))
        precision = stabilize(prior_precision + basis.transpose(-1, -2) @ (basis * inv[..., None]))
        rhs = basis.transpose(-1, -2) @ (inv * error)[..., None]
        chol = torch.linalg.cholesky(precision)
        quadratic = (error.square() * inv).sum(-1) - (rhs.transpose(-1, -2) @ torch.cholesky_solve(rhs, chol)).squeeze(-1).squeeze(-1)
        logdet = (noise.clamp_min(0.25).log() * mask).sum(-1)
        logdet = logdet + torch.linalg.slogdet(stabilize(device_covariance))[1]
        logdet = logdet + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (quadratic + logdet + mask.sum(-1) * math.log(2 * math.pi))

    def block_information(self, basis, error, covariance, mask):
        valid = mask.float()
        noise = covariance * valid[..., None] * valid[..., None, :] + torch.diag_embed(1 - valid)
        chol = torch.linalg.cholesky(stabilize(noise))
        design = basis * valid[..., None]
        residual = error * valid
        solved_design = torch.cholesky_solve(design, chol)
        solved_error = torch.cholesky_solve(residual[..., None], chol).squeeze(-1)
        information = (design.transpose(-1, -2) @ solved_design).sum(-3)
        vector = (design.transpose(-1, -2) @ solved_error[..., None]).squeeze(-1).sum(-2)
        quadratic = (residual * solved_error).sum((-1, -2))
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum((-1, -2))
        return (information, vector, quadratic, logdet, valid.sum((-1, -2)))

    def predictive_block_score(self, rssi, mask, source_mean, basis, noise, device_mean, device_covariance):
        prediction = source_mean + (basis @ device_mean[..., None, :, None]).squeeze(-1)
        information, vector, quadratic, logdet, count = self.block_information(basis, rssi - prediction, noise, mask)
        prior_precision = torch.linalg.inv(stabilize(device_covariance))
        chol = torch.linalg.cholesky(stabilize(prior_precision + information))
        quadratic = quadratic - (vector[..., None, :] @ torch.cholesky_solve(vector[..., None], chol)).squeeze(-1).squeeze(-1)
        logdet = logdet + torch.linalg.slogdet(stabilize(device_covariance))[1] + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (quadratic + logdet + count * math.log(2 * math.pi))

    def messages(self, source_mean, basis, rssi, mask, noise, geometry_log_prior, transition_log_prob=None, rounds=2, initial_information=None, initial_vector=None, causal=True):
        b, regions, modes, aps = source_mean.shape
        inverse_noise = mask[:, :, None].float() / noise.clamp_min(0.25)
        information = torch.einsum('brkad,brka,brkae->brkde', basis, inverse_noise, basis)
        vector = torch.einsum('brkad,brka,brka->brkd', basis, inverse_noise, rssi[:, :, None] - source_mean)
        geometry = geometry_log_prior.softmax(-1)
        independent_info = torch.einsum('brk,brkde->brde', geometry, information)
        independent_vector = torch.einsum('brk,brkd->brd', geometry, vector)
        cavity_scores, cavity_means, cavity_covariances = ([], [], [])
        for heldout in range(regions):
            keep = (torch.arange(regions, device=rssi.device) < heldout).float() if causal else torch.ones(regions, device=rssi.device)
            keep[heldout] = 0
            info = (independent_info * keep[None, :, None, None]).sum(1)
            rhs = (independent_vector * keep[None, :, None]).sum(1)
            if initial_information is not None:
                info = info + initial_information
                rhs = rhs + initial_vector
            mu, cov = self.posterior(info, rhs)
            score = self.predictive_diagonal_score(rssi[:, heldout, None], mask[:, heldout, None], source_mean[:, heldout], basis[:, heldout], noise[:, heldout], mu[:, None], cov[:, None])
            cavity_scores.append(score)
            cavity_means.append(mu)
            cavity_covariances.append(cov)
        heldout_scores = torch.stack(cavity_scores, 1)
        beliefs = geometry_log_prior
        for _ in range(rounds):
            logits = geometry_log_prior + heldout_scores
            if transition_log_prob is not None:
                forward = [logits[:, 0]]
                for r in range(1, regions):
                    forward.append(logits[:, r] + torch.logsumexp(forward[-1][..., :, None] + transition_log_prob[:, r - 1], -2))
                beliefs = torch.stack(forward, 1).log_softmax(-1)
            else:
                beliefs = logits.log_softmax(-1)
        final_info = torch.einsum('brk,brkde->bde', beliefs.exp(), information)
        final_vector = torch.einsum('brk,brkd->bd', beliefs.exp(), vector)
        if initial_information is not None:
            final_info = final_info + initial_information
            final_vector = final_vector + initial_vector
        final_mean, final_cov = self.posterior(final_info, final_vector)
        conditional_mean, conditional_cov = self.posterior(information, vector)
        return dict(log_weights=beliefs, heldout_scores=heldout_scores, cavity_mean=torch.stack(cavity_means, 1), cavity_covariance=torch.stack(cavity_covariances, 1), conditional_mean=conditional_mean, conditional_covariance=conditional_cov, shared_mean=final_mean, shared_covariance=final_cov)

    def path_messages(self, source_mean, basis, rssi, mask, noise, geometry_log_prior, initial_information, initial_vector, prefix_mean=None, prefix_covariance=None, support_offset=None, forecast_offset=None, block_covariance=None):
        b, regions, k, _ = source_mean.shape
        inv = mask[:, :, None].float() / noise.clamp_min(0.25)
        information = torch.einsum('brkad,brka,brkae->brkde', basis, inv, basis)
        support_mean = source_mean if support_offset is None else source_mean + support_offset
        forecast_mean = source_mean if forecast_offset is None else source_mean + forecast_offset
        vector = torch.einsum('brkad,brka,brka->brkd', basis, inv, rssi[:, :, None] - support_mean)
        if block_covariance is not None:
            time, aps = block_covariance.shape[-3:-1]
            block_basis = basis.reshape(b, regions, k, time, aps, 6)
            block_mask = mask.reshape(b, regions, 1, time, aps).expand(b, regions, k, time, aps)
            block_support = support_mean.reshape(b, regions, k, time, aps)
            block_rssi = rssi.reshape(b, regions, 1, time, aps)
            information, vector, _, _, _ = self.block_information(block_basis, block_rssi - block_support, block_covariance, block_mask)
        info = initial_information[:, None].expand(b, k, 6, 6)
        rhs = initial_vector[:, None].expand(b, k, 6)
        if prefix_mean is not None:
            prefix_precision = torch.linalg.inv(stabilize(prefix_covariance))
            info = torch.zeros_like(info)
            rhs = torch.zeros_like(rhs)

        def posterior(info, rhs):
            if prefix_mean is None:
                return self.posterior(info, rhs)
            precision = stabilize(prefix_precision + info)
            vector = (prefix_precision @ prefix_mean[..., None]).squeeze(-1) + rhs
            chol = torch.linalg.cholesky(precision)
            return (torch.cholesky_solve(vector[..., None], chol).squeeze(-1), torch.cholesky_inverse(chol))
        means = []
        covariances = []
        scores = []
        for region in range(regions):
            mu, cov = posterior(info, rhs)
            if block_covariance is None:
                score = self.predictive_diagonal_score(rssi[:, region, None], mask[:, region, None], forecast_mean[:, region], basis[:, region], noise[:, region], mu, cov)
            else:
                score = self.predictive_block_score(block_rssi[:, region], block_mask[:, region], forecast_mean[:, region].reshape(b, k, time, aps), block_basis[:, region], block_covariance[:, region], mu, cov)
            means.append(mu)
            covariances.append(cov)
            scores.append(score)
            info = info + information[:, region]
            rhs = rhs + vector[:, region]
        return dict(cavity_mean=torch.stack(means, 1), cavity_covariance=torch.stack(covariances, 1), heldout_scores=torch.stack(scores, 1), conditional_mean=posterior(info, rhs)[0])

class SpatialRSSIHead(nn.Module):

    def __init__(self, variant='set', history_context=False, geometry_head=False, temporal_contrast=0):
        super().__init__()
        self.variant = variant
        self.history_context = history_context
        self.geometry_head = bool(geometry_head)
        self.temporal_contrast = int(temporal_contrast)
        if geometry_head:
            self.geometry_potential = GeometryRSSIPotential()
        self.ap = mlp((72 if history_context else 16) + 9 * self.temporal_contrast, 64, 32)
        self.readout = mlp(64, 32, 1, last_zero=True)
        if variant == 'pair':
            self.pair = mlp(21 + 4 * self.temporal_contrast, 48, 24)
            self.pair_readout = mlp(48, 32, 1, last_zero=True)

    def forward(self, rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence=None, motion=None, observation_covariance=None, observation_cholesky=None, temporal=None):
        error = (rssi - prediction) / variance.clamp_min(0.25).sqrt()
        scalars = torch.stack((error, variance.clamp_min(0.25).log() / 5, field['mean'] / 50, field['gradient'][..., 0] / 10, field['gradient'][..., 1] / 10, device_variance / 100, field['visibility'], mask.float()), -1)
        features = [scalars, ap_embedding.expand(*scalars.shape[:-1], 8)]
        if self.history_context:
            if evidence is None or motion is None:
                raise ValueError('Temporal spatial head requires causal radio and motion context')
            features.extend((evidence, motion[..., None, :].expand(*scalars.shape[:-1], 32)))
        if self.temporal_contrast:
            if temporal is None:
                raise ValueError('Temporal contrast head needs a causal query-side context')
            features.append(temporal[0])
        hidden = self.ap(torch.cat(features, -1))
        weight = mask[..., None].float()
        den = weight.sum(-2).clamp_min(1.0)
        average = (hidden * weight).sum(-2) / den
        variance_h = ((hidden - average[..., None, :]).square() * weight).sum(-2) / den
        value = self.readout(torch.cat((average, (variance_h + 1e-06).sqrt()), -1)).squeeze(-1)
        if self.variant == 'pair':
            a = prediction.shape[-1]
            left, right = torch.triu_indices(a, a, 1, device=prediction.device)
            pair_mask = mask[..., left] & mask[..., right]
            source = field['mean'][..., left] - field['mean'][..., right]
            residual = (rssi[..., left] - rssi[..., right] - (prediction[..., left] - prediction[..., right])) / 10
            gradient = field['gradient'][..., left, :] - field['gradient'][..., right, :]
            pair_scalar = torch.cat((residual[..., None], (source / 20)[..., None], gradient / 10, pair_mask[..., None].float()), -1)
            embedding = ap_embedding.expand(*scalars.shape[:-1], 8)
            pair_features = [pair_scalar, embedding[..., left, :], embedding[..., right, :]]
            if self.temporal_contrast:
                pair_features.append(temporal[1])
            pair_h = self.pair(torch.cat(pair_features, -1))
            pw = pair_mask[..., None].float()
            pn = pw.sum(-2).clamp_min(1.0)
            pm = (pair_h * pw).sum(-2) / pn
            pv = ((pair_h - pm[..., None, :]).square() * pw).sum(-2) / pn
            value = value + self.pair_readout(torch.cat((pm, (pv + 1e-06).sqrt()), -1)).squeeze(-1)
        if self.geometry_head:
            if observation_covariance is None or observation_cholesky is None:
                raise ValueError('Geometry head needs full predictive AP covariance')
            value = value + self.geometry_potential(rssi, mask, prediction, field, observation_covariance, observation_cholesky, device_variance)
        return 4 * (value.float() / 4).tanh()

class SpatialPrecisionHead(nn.Module):

    def __init__(self, variant='set', history_context=True):
        super().__init__()
        self.variant = variant
        self.history_context = history_context
        self.ap = mlp(72 if history_context else 16, 48, 24)
        self.readout = mlp(48, 32, 2, last_zero=True)
        if variant == 'pair':
            self.pair = mlp(21, 32, 16)
            self.pair_readout = mlp(32, 24, 2, last_zero=True)

    def forward(self, rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence=None, motion=None, observation_covariance=None, observation_cholesky=None):
        if observation_covariance is None:
            raise ValueError('Spatial precision needs the full predictive covariance')
        scalars = torch.stack((variance.clamp_min(0.25).log() / 5, field['mean'] / 50, field['gradient'][..., 0] / 10, field['gradient'][..., 1] / 10, device_variance / 100, field['visibility'], mask.float(), field['gradient'].square().sum(-1).sqrt() / 10), -1)
        features = [scalars, ap_embedding.expand(*scalars.shape[:-1], 8)]
        if self.history_context:
            if evidence is None or motion is None:
                raise ValueError('Missing causal precision context')
            features.extend((evidence, motion[..., None, :].expand(*scalars.shape[:-1], 32)))
        h = self.ap(torch.cat(features, -1))
        w = mask[..., None].float()
        den = w.sum(-2).clamp_min(1)
        average = (h * w).sum(-2) / den
        spread = (((h - average[..., None, :]).square() * w).sum(-2) / den + 1e-06).sqrt()
        raw = self.readout(torch.cat((average, spread), -1))
        if self.variant == 'pair':
            a = prediction.shape[-1]
            left, right = torch.triu_indices(a, a, 1, device=prediction.device)
            pmask = mask[..., left] & mask[..., right]
            gradient = field['gradient'][..., left, :] - field['gradient'][..., right, :]
            pair_variance = variance[..., left] + variance[..., right] - 2 * observation_covariance[..., left, right]
            values = torch.cat(((field['mean'][..., left] - field['mean'][..., right])[..., None] / 20, gradient / 10, pair_variance.clamp_min(0.25).log()[..., None] / 5, pmask[..., None].float(), ap_embedding[left].expand(*pmask.shape, 8), ap_embedding[right].expand(*pmask.shape, 8)), -1)
            ph = self.pair(values)
            pw = pmask[..., None].float()
            pn = pw.sum(-2).clamp_min(1)
            pa = (ph * pw).sum(-2) / pn
            ps = (((ph - pa[..., None, :]).square() * pw).sum(-2) / pn + 1e-06).sqrt()
            raw = raw + self.pair_readout(torch.cat((pa, ps), -1))
        gain = (math.log(4.0) * raw.float().tanh()).exp()
        self.last_gain = gain.detach()
        chol = observation_cholesky if observation_cholesky is not None else torch.linalg.cholesky(stabilize(observation_covariance))
        difference = (rssi - prediction) * mask
        white = torch.linalg.solve_triangular(chol, difference[..., None], upper=False).squeeze(-1)
        gradient = field['gradient'] * mask[..., None]
        g = torch.linalg.solve_triangular(chol, gradient, upper=False)
        gram = stabilize(g.transpose(-1, -2) @ g, floor=0.0001)
        root = torch.linalg.cholesky(gram)
        u = torch.linalg.solve_triangular(root, g.transpose(-1, -2), upper=False).transpose(-1, -2)
        direction_gram = u.transpose(-1, -2) @ u
        small = torch.eye(2, device=prediction.device) + (gain - 1)[..., :, None] * direction_gram
        determinant = small[..., 0, 0] * small[..., 1, 1] - small[..., 0, 1] * small[..., 1, 0]
        logdet = determinant.clamp_min(1e-07).log()
        projected = (u.transpose(-1, -2) @ white[..., None]).squeeze(-1)
        energy = ((gain - 1) * projected.square()).sum(-1)
        return 0.5 * (logdet - energy)

class ThreeModuleNetwork(nn.Module):

    def __init__(self, cache, geometry, variant='set', history_context=False, precision_head=False, robust_radio=False, initial_context_scans=1, aligned_motion=False, full_pose_covariance=False, project_innovations=False, orientation_basis=False, mode_evidence='local', initial_posterior='none', ap_basis_prior=False, fixed_ap_basis=False, robust_transfer=False, calibrated_geometry=False, geometry_head=False, source_model='idw', source_inducing=32, source_lengthscale=1.2, online_device_policy='cavity', memory_recenter=False, trajectory_mode_weight=0.0, cs_temporal_noise=None, mode_spatial_evidence=False, position_integration='legacy', region_reliability='none', temporal_contrast=0, initial_recognition='first', field_tangent_weight=0.0):
        super().__init__()
        self.variant = variant
        self.field = PhysicalRadioField(cache, geometry, orientation_basis, ap_basis_prior, fixed_ap_basis, source_model, source_inducing, source_lengthscale)
        self.calibrated_geometry = bool(calibrated_geometry)
        self.geometry_head = bool(geometry_head)
        if geometry_head and (not calibrated_geometry or precision_head):
            raise ValueError('Geometry potential requires calibrated geometry and the selected AP-pair head')
        self.field.calibrated_geometry = self.calibrated_geometry
        self.integration_grid = SpatialIntegrationGrid(self.field, cache, position_integration)
        self.ap_basis_prior = bool(ap_basis_prior)
        if initial_posterior not in ('none', 'mean', 'full'):
            raise ValueError('Unknown initial recognition policy')
        self.calibration = model_CounterfactualCalibration(initial_posterior)
        if region_reliability not in ('none', 'prequential', 'crossfit'):
            raise ValueError('Unknown regional reliability')
        self.region_reliability = region_reliability
        if region_reliability != 'none':
            if initial_posterior == 'none' or online_device_policy != 'cavity':
                raise ValueError('Regional reliability requires an independent initial posterior and held-out current region')
            self.calibration.region_logit_prior = nn.Parameter(torch.tensor(math.log(9.0)))
        self.cs_temporal_noise = cs_temporal_noise
        self.calibration.cs_white_sigma_scale = float(cs_temporal_noise.get('white_sigma_scale', 1.0)) if cs_temporal_noise else 1.0
        self.initial_posterior = initial_posterior
        if initial_recognition not in ('first', 'time_average', 'evidence_set'):
            raise ValueError('Unknown initial recognition context')
        if initial_recognition != 'first' and (initial_posterior == 'none' or initial_context_scans != 5 or (not full_pose_covariance)):
            raise ValueError('Window recognition requires an independent full-covariance initial-five channel')
        self.initial_recognition = initial_recognition
        if field_tangent_weight < 0 or (field_tangent_weight and (not calibrated_geometry)):
            raise ValueError('Field tangent loss requires calibrated geometry')
        self.field_tangent_weight = float(field_tangent_weight)
        if online_device_policy not in ('cavity', 'all'):
            raise ValueError('Unknown causal device posterior policy')
        self.online_device_policy = online_device_policy
        self.memory_recenter = bool(memory_recenter)
        self.trajectory_mode_weight = float(trajectory_mode_weight)
        self.precision_head = precision_head
        if temporal_contrast not in (0, 1, 3) or (temporal_contrast and precision_head):
            raise ValueError('Temporal contrasts require the spatial potential head')
        self.temporal_contrast = int(temporal_contrast)
        self.robust_radio = robust_radio
        self.robust_transfer = bool(robust_transfer)
        self.initial_context_scans = int(initial_context_scans)
        self.aligned_motion = bool(aligned_motion)
        self.full_pose_covariance = bool(full_pose_covariance)
        self.project_innovations = bool(project_innovations)
        self.orientation_basis = bool(orientation_basis)
        if mode_evidence not in ('local', 'rolling', 'frozen'):
            raise ValueError('Unknown regional evidence policy')
        self.mode_evidence = mode_evidence
        self.mode_spatial_evidence = bool(mode_spatial_evidence)
        if self.mode_spatial_evidence and (mode_evidence == 'local' or precision_head):
            raise ValueError('Head partition ratio requires CF mode evidence and a spatial potential head')
        self.calibration.position_difference_context = precision_head
        self.localization = SpatialPrecisionHead(variant, history_context) if precision_head else SpatialRSSIHead(variant, history_context, geometry_head, temporal_contrast)

    def position_nodes(self):
        return self.integration_grid.xy

    def configuration(self):
        return dict(variant=self.variant, device_parameters=6, field_basis=[49, 64, 32, 4], radio_tcn=[8, 24, 1, 2], motion_tcn=[41, 32, 1, 2, 4], radio_quality=[80, 32, 2], innovation_gate=[97, 64, 4], cs_encoder=[20, 32, 32], cs_quality=[64, 32, 2], localization_ap=[(72 if self.localization.history_context else 16) + 9 * self.temporal_contrast, 48 if self.precision_head else 64, 24 if self.precision_head else 32], localization_context=self.localization.history_context, localization_kind='precision' if self.precision_head else 'potential', precision_gain_bounds=[0.25, 4.0] if self.precision_head else None, radio_likelihood='gaussian_student_t_mixture' if self.robust_radio else 'gaussian', initial_context_scans=self.initial_context_scans, motion_history_policy='wifi_only_transient_cs_v2' if self.aligned_motion else 'legacy_mixed_callbacks', calibration_geometry='pre_pose_full_AP_covariance' if self.full_pose_covariance else 'legacy_post_pose_diagonal', innovation_memory='conditional_nuisance' if self.project_innovations else 'raw_native', orientation_basis='relative_AP_harmonics' if self.orientation_basis else 'none', mode_evidence=self.mode_evidence, robust_transfer=self.robust_transfer, calibrated_geometry=self.calibrated_geometry, localization_geometry='whitened_tangent_and_orthogonal' if self.geometry_head else 'none', geometry_potential=[16, 32, 16, 1] if self.geometry_head else None, source_field=self.field.source_model, source_inducing=self.field.source_inducing, source_lengthscale_m=self.field.source_lengthscale, source_reference_fit=self.field.source_gp.fit_summary if self.field.source_model == 'physical_sparse_gp' else None, online_device_policy=self.online_device_policy, memory_recenter=self.memory_recenter, trajectory_mode_weight=self.trajectory_mode_weight, cs_temporal_noise=self.cs_temporal_noise, mode_spatial_evidence='head_partition_ratio' if self.mode_spatial_evidence else 'none', position_integration=self.integration_grid.kind, region_reliability=self.region_reliability, temporal_contrast=self.temporal_contrast, temporal_lags_s=[0.9] if self.temporal_contrast == 1 else [0.35, 0.9, 1.8] if self.temporal_contrast else [], integration_grid=self.integration_grid.description(), initial_posterior=self.initial_posterior, initial_recognition=self.initial_recognition, field_tangent_weight=self.field_tangent_weight, ap_basis_prior=self.ap_basis_prior, fixed_ap_basis=self.field.fixed_ap_basis, localization_pool=[48, 32, 2] if self.precision_head else [64, 32, 1], trainable_parameters=sum((p.numel() for p in self.parameters() if p.requires_grad)))
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(5)
_GH_Z = np.stack(np.meshgrid(_GH_X, _GH_X, indexing='ij'), -1).reshape(-1, 2) * math.sqrt(2.0)
_GH_LOG_W = np.log(np.outer(_GH_W, _GH_W).reshape(-1) / math.pi)

def position_support(mean, covariance, source_xy, rule='mixed', log_area=None):
    shape = mean.shape[:-1]
    g = len(source_xy)
    chol = torch.linalg.cholesky(covariance)
    global_points = source_xy.expand(*shape, g, 2)
    if rule == 'hermite':
        z = torch.as_tensor(_GH_Z, device=mean.device, dtype=mean.dtype)
        local = mean[..., None, :] + torch.einsum('...de,ne->...nd', chol, z)
        points = torch.cat((global_points, local), -2)
        local_prior = torch.as_tensor(_GH_LOG_W, device=mean.device, dtype=mean.dtype).expand(*shape, 25)
        prior = torch.cat((torch.full((*shape, g), -torch.inf, device=mean.device, dtype=mean.dtype), local_prior), -1)
    elif rule in ('mixed', 'balanced'):
        difference = global_points - mean[..., None, :]
        precision = torch.linalg.inv(covariance)
        density = -0.5 * torch.einsum('...nd,...de,...ne->...n', difference, precision, difference)
        if log_area is not None:
            density = density + log_area
        grid_prior = density.log_softmax(-1)
        fraction = ((covariance.diagonal(dim1=-2, dim2=-1).mean(-1).sqrt() - 0.15) / 0.45).clamp(0.0, 0.75)
        local_mean = mean
        local_chol = chol
        if rule == 'balanced':
            probability = grid_prior.exp()
            grid_mean = (probability[..., None] * global_points).sum(-2)
            centred = global_points - grid_mean[..., None, :]
            grid_cov = torch.einsum('...n,...nd,...ne->...de', probability, centred, centred)
            shift = grid_mean - mean
            white_shift = torch.linalg.solve_triangular(chol, shift[..., None], upper=False).squeeze(-1)
            left = torch.linalg.solve_triangular(chol, grid_cov, upper=False)
            white_cov = torch.linalg.solve_triangular(chol, left.transpose(-1, -2), upper=False).transpose(-1, -2)
            white_cov = 0.5 * (white_cov + white_cov.transpose(-1, -2))
            largest = 0.5 * (white_cov[..., 0, 0] + white_cov[..., 1, 1]) + (0.25 * (white_cov[..., 0, 0] - white_cov[..., 1, 1]).square() + white_cov[..., 0, 1].square()).clamp_min(1e-16).sqrt()
            bound = 0.9
            value = largest + white_shift.square().sum(-1) + bound
            root = 2 * bound / (value + (value.square() - 4 * largest * bound).clamp_min(0).sqrt()).clamp_min(1e-12)
            fraction = torch.minimum(fraction, 0.95 * root)
            remainder = (1 - fraction).clamp_min(1e-06)
            local_mean = mean - fraction[..., None] / remainder[..., None] * shift
            outer = shift[..., None] * shift[..., None, :]
            residual = (covariance - fraction[..., None, None] * grid_cov - (fraction / remainder)[..., None, None] * outer) / remainder[..., None, None]
            residual = 0.5 * (residual + residual.transpose(-1, -2))
            local_chol = torch.linalg.cholesky(residual)
        local = torch.stack((torch.zeros_like(mean), 3 ** 0.5 * local_chol[..., 0], -3 ** 0.5 * local_chol[..., 0], 3 ** 0.5 * local_chol[..., 1], -3 ** 0.5 * local_chol[..., 1]), -2) + local_mean[..., None, :]
        points = torch.cat((global_points, local), -2)
        local_prior = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], device=mean.device, dtype=mean.dtype).log().expand(*shape, 5)
        prior = torch.cat((grid_prior + fraction.clamp_min(1e-20).log()[..., None], local_prior + torch.log1p(-fraction)[..., None]), -1)
    else:
        raise ValueError('Unknown position rule: ' + rule)
    return (points, prior, g)

def inducing_indices(xy, count):
    count = min(int(count), len(xy))
    selected = [int(((xy - xy.mean(0)) ** 2).sum(-1).argmin())]
    distance = np.full(len(xy), np.inf)
    while len(selected) < count:
        distance = np.minimum(distance, ((xy - xy[selected[-1]]) ** 2).sum(-1))
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    return np.asarray(selected)

def matern32(left, right, lengthscale):
    radius = np.linalg.norm(left[:, None] - right[None], axis=-1)
    scaled = math.sqrt(3) * radius / lengthscale
    return (1 + scaled) * np.exp(-scaled)

class SparsePhysicalSource(nn.Module):

    def __init__(self, cache, geometry, inducing_count=32, lengthscale=1.2):
        super().__init__()
        self.lengthscale = float(lengthscale)
        xy = np.asarray(cache['xy'], float)
        mask = np.asarray(cache['mask'], bool)
        mean = np.asarray(cache['trend'], float) + np.asarray(cache['residual'], float)
        variance = np.asarray(cache['var'], float)
        centres = np.asarray(geometry['centres_m'], float)
        exponent = np.asarray(geometry['exponents'], float)
        exponent = np.clip(exponent, 0.2, 6.0)
        log_distance = 0.5 * np.log10(((xy[:, None] - centres) ** 2).sum(-1) + 1.0)
        power = np.asarray([np.mean(np.asarray(cache['trend'])[mask[:, a], a] + 10 * exponent[a] * log_distance[mask[:, a], a]) for a in range(mean.shape[-1])])
        trend = power - 10 * exponent * log_distance
        indices = inducing_indices(xy, inducing_count)
        inducing = xy[indices]
        m = len(inducing)
        base_uu = matern32(inducing, inducing, self.lengthscale)
        base_xu = matern32(xy, inducing, self.lengthscale)
        roots = []
        coefficients = []
        adjustments = []
        amplitudes = []
        noise = []
        fit_error = []
        for a in range(mean.shape[-1]):
            valid = mask[:, a] & np.isfinite(mean[:, a]) & np.isfinite(variance[:, a])
            if valid.sum() < 4:
                raise ValueError('OPPO map has insufficient observations for an AP')
            residual = mean[valid, a] - trend[valid, a]
            amplitude = max(2.0, float(np.std(residual)))
            amp2 = amplitude ** 2
            kuu = amp2 * base_uu + np.eye(m) * (amp2 * 0.0001 + 1e-06)
            root = np.linalg.cholesky(kuu)
            inverse_root_transpose = np.linalg.solve(root, np.eye(m)).T
            feature = amp2 * base_xu[valid] @ inverse_root_transpose
            measurement = np.maximum(variance[valid, a], 0.25)
            diagonal = measurement + np.maximum(amp2 - (feature ** 2).sum(-1), 0.0)
            precision = np.eye(m) + feature.T @ (feature / diagonal[:, None])
            posterior = np.linalg.inv(precision)
            posterior = 0.5 * (posterior + posterior.T)
            coefficient = posterior @ (feature.T @ (residual / diagonal))
            roots.append(inverse_root_transpose)
            coefficients.append(coefficient)
            adjustments.append(posterior - np.eye(m))
            amplitudes.append(amp2)
            noise.append(float(np.median(measurement)))
            fit_error.append(float(np.mean(abs(feature @ coefficient - residual))))
        values = dict(inducing=inducing, centres=centres, exponent=exponent, power=power, inverse_root_transpose=np.asarray(roots), coefficient=np.asarray(coefficients), posterior_adjustment=np.asarray(adjustments), amplitude_squared=np.asarray(amplitudes), measurement_noise=np.asarray(noise), inducing_visibility=np.asarray(cache['visibility'])[indices].T)
        for name, value in values.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.fit_summary = dict(source='OPPO reference map only', inducing_count=m, lengthscale_m=self.lengthscale, kernel='Matern_3_2', residual='Sparse FITC Gaussian process', AP_fit_MAE_db=fit_error, source_measurement_variance_db2=noise, target_inputs=False)

    def forward(self, points, component_gradient=False):
        offset = points[..., None, :] - self.inducing
        radius = offset.square().sum(-1).clamp_min(1e-12).sqrt()
        scaled = math.sqrt(3) * radius / self.lengthscale
        decay = (-scaled).exp()
        base = (1 + scaled) * decay
        kernel = base[..., None, :] * self.amplitude_squared[:, None]
        feature = torch.einsum('...am,amn->...an', kernel, self.inverse_root_transpose)
        derivative = -3 / self.lengthscale ** 2 * offset * decay[..., None]
        kernel_derivative = derivative[..., None, :, :] * self.amplitude_squared[:, None, None]
        feature_derivative = torch.einsum('...amz,amn->...anz', kernel_derivative, self.inverse_root_transpose)
        residual = torch.einsum('...am,am->...a', feature, self.coefficient)
        residual_gradient = torch.einsum('...amz,am->...az', feature_derivative, self.coefficient)
        radio_offset = points[..., None, :] - self.centres
        radius2 = radio_offset.square().sum(-1) + 1.0
        trend = self.power - 5 * self.exponent * radius2.log10()
        trend_gradient = -10 / math.log(10) * self.exponent[..., None] * radio_offset / radius2[..., None]
        raw_variance = self.amplitude_squared + self.measurement_noise + torch.einsum('...am,amn,...an->...a', feature, self.posterior_adjustment, feature)
        variance = raw_variance.clamp_min(0.25)
        variance_gradient = 2 * torch.einsum('...amz,amn,...an->...az', feature_derivative, self.posterior_adjustment, feature)
        variance_gradient = variance_gradient * (raw_variance > 0.25)[..., None]
        total = base.sum(-1).clamp_min(1e-08)
        visibility = torch.einsum('...m,am->...a', base, self.inducing_visibility) / total[..., None]
        visibility_gradient = torch.einsum('...mz,am->...az', derivative, self.inducing_visibility) / total[..., None, None] - visibility[..., None] * derivative.sum(-2)[..., None, :] / total[..., None, None]
        visibility_gradient = visibility_gradient * ((visibility > 0.01) & (visibility < 0.99))[..., None]
        result = dict(trend=trend, residual=residual, variance=variance, visibility=visibility.clamp(0.01, 0.99), mean=trend + residual, gradient=trend_gradient + residual_gradient)
        if component_gradient:
            result['component_gradient'] = torch.stack((trend_gradient, residual_gradient, variance_gradient, visibility_gradient), -2)
        return result

def robust_radio_score(nominal_score, rssi, mask, prediction, covariance, outlier_probability, head_potential=None):
    valid = mask.float()
    probability = (1 - (1 - outlier_probability * valid).prod(-1)).clamp(1e-06, 1 - 1e-06)
    wide = stabilize(covariance + torch.diag_embed(100.0 * valid))
    chol = torch.linalg.cholesky(wide)
    residual = (rssi - prediction) * valid
    white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
    quadratic = white.square().sum(-1)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    dimensions = valid.sum(-1)
    nu = 3.0
    tail = torch.lgamma((nu + dimensions) / 2) - math.lgamma(nu / 2) - 0.5 * (dimensions * math.log(nu * math.pi) + logdet) - (nu + dimensions) / 2 * torch.log1p(quadratic / nu)
    normal_component = torch.log1p(-probability) + nominal_score
    tail_component = probability.log() + tail
    mixture = torch.logaddexp(normal_component, tail_component)
    responsibility = (normal_component - mixture).exp()
    if head_potential is not None:
        baseline = torch.logaddexp(normal_component - head_potential, tail_component)
        return (mixture, responsibility, baseline)
    return (mixture, responsibility)

def log_integral(information, vector, mean, covariance):
    information = information.double()
    vector = vector.double()
    mean = mean.double()
    covariance = covariance.double()
    root = torch.linalg.cholesky(0.5 * (covariance + covariance.transpose(-1, -2)) + torch.eye(6, device=mean.device) * 1e-09)
    whitened = root.transpose(-1, -2) @ information @ root
    precision = torch.eye(6, device=mean.device) + 0.5 * (whitened + whitened.transpose(-1, -2))
    factor = torch.linalg.cholesky(precision)
    h = (root.transpose(-1, -2) @ (vector - (information @ mean[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    solution = torch.cholesky_solve(h[..., None], factor).squeeze(-1)
    return (vector * mean).sum(-1) - 0.5 * torch.einsum('...d,...de,...e->...', mean, information, mean) + 0.5 * (h * solution).sum(-1) - factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)

def shared_probability(calibration, information, vector, reference_mean, reference_covariance, base_mean, base_covariance):
    shared = log_integral(information, vector, reference_mean, reference_covariance)
    independent = log_integral(information, vector, base_mean, base_covariance)
    bayes_factor = shared - independent
    probability = torch.sigmoid(calibration.region_logit_prior.double() + bayes_factor).to(vector.dtype)
    return (probability, bayes_factor.to(vector.dtype))

def leave_one_out_references(information, vector, retired_information, retired_vector):
    count = vector.shape[-2]
    other = torch.ones(count, count, device=vector.device) - torch.eye(count, device=vector.device)
    return (retired_information[..., None, :, :] + torch.einsum('rs,...sde->...rde', other, information), retired_vector[..., None, :] + torch.einsum('rs,...sd->...rd', other, vector))

def closed_factors(calibration, mode, information, vector, original_probability, retired_information, retired_vector, base_mean, base_covariance):
    if mode == 'crossfit':
        reference_information, reference_vector = leave_one_out_references(information, vector, retired_information, retired_vector)
        mean, covariance = calibration.posterior(reference_information, reference_vector, base_mean[..., None, :], base_covariance[..., None, :, :])
        probability, _ = shared_probability(calibration, information, vector, mean, covariance, base_mean[..., None, :], base_covariance[..., None, :, :])
    elif mode == 'prequential':
        probability = original_probability
    else:
        raise ValueError('Unknown region evidence reliability mode')
    return (retired_information + (information * probability[..., None, None]).sum(-3), retired_vector + (vector * probability[..., None]).sum(-2), probability)

def mlp_with_directions(network, inputs, directions):
    value = inputs
    jacobian = directions
    for layer in network:
        if isinstance(layer, nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
            jacobian = torch.einsum('...id,oi->...od', jacobian, layer.weight)
        elif isinstance(layer, nn.SiLU):
            probability = value.sigmoid()
            jacobian = jacobian * (probability * (1 + value * (1 - probability)))[..., None]
            value = F.silu(value)
        else:
            raise TypeError('Unsupported physical-basis activation: ' + type(layer).__name__)
    result = value.tanh().float()
    return (result, jacobian * (1 - result.square())[..., None])

def calibrated_field(field, device_mean):
    if 'basis_gradient' not in field:
        return field
    source = field.get('source_gradient', field['gradient'])
    gradient = source + torch.einsum('...adz,...d->...az', field['basis_gradient'], device_mean)
    return dict(field, source_gradient=source, gradient=gradient)

class GeometryRSSIPotential(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, rssi, mask, prediction, field, covariance, cholesky, device_variance):
        valid = mask.float()
        residual = (rssi - prediction) * valid
        white = torch.linalg.solve_triangular(cholesky, residual[..., None], upper=False).squeeze(-1)
        gradient = field['gradient'] * valid[..., None]
        design = torch.linalg.solve_triangular(cholesky, gradient, upper=False)
        gram = design.transpose(-1, -2) @ design
        rhs = (design.transpose(-1, -2) @ white[..., None]).squeeze(-1)
        displacement = torch.linalg.solve(gram + 0.01 * torch.eye(2, device=rssi.device), rhs[..., None]).squeeze(-1)
        tangent = (design @ displacement[..., None]).squeeze(-1)
        orthogonal = white - tangent
        source = field.get('source_gradient', field['gradient']) * valid[..., None]
        source = torch.linalg.solve_triangular(cholesky, source, upper=False)
        alignment = (source * design).sum((-1, -2)) / (source.square().sum((-1, -2)) * design.square().sum((-1, -2))).clamp_min(1e-08).sqrt()
        attenuation = (design.square().sum((-1, -2)) + 0.01) / (source.square().sum((-1, -2)) + 0.01)
        scalars = torch.stack((orthogonal.square().sum(-1).log1p(), tangent.square().sum(-1).log1p(), gram[..., 0, 0].log1p(), gram[..., 1, 1].log1p(), gram[..., 0, 1] / (1 + gram.diagonal(dim1=-2, dim2=-1).sum(-1)), alignment, attenuation.log().clamp(-5, 5), device_variance.mean(-1).log1p() / 5), -1)
        features = torch.cat((white.clamp(-20, 20) / 5, valid, scalars), -1)
        if features.shape[-1] != 16:
            raise ValueError('Geometry head expects the four-map-AP protocol')
        return 4 * self.network(features).squeeze(-1).tanh()

def temporal_contrast_features(gradient, rssi, mask, delta, context, count):
    lags = [0.9] if count == 1 else [0.35, 0.9, 1.8]
    b, k, q, a, _ = gradient.shape
    age = context['now_time'][:, None] - context['times']
    target = age.new_tensor(lags)
    time_weight = torch.exp(-0.5 * ((age[..., None] - target) / (0.1 + 0.2 * target)).square())
    time_weight = time_weight * ((age >= 0) & (age <= 3.0))[..., None]
    weight = time_weight[:, :, None] * context['mask'][..., None]
    mass = weight.sum(1)
    weight = weight / mass.clamp_min(1e-08)[:, None]
    past_rssi = torch.einsum('bhal,bha->bal', weight, context['rssi'])
    past_delta = torch.einsum('bhal,bhka->bkal', weight, context['delta'])
    past_path = torch.einsum('bhal,bhkz->bkalz', weight, context['path'])
    past_variance = torch.einsum('bhal,bhk->bkal', weight, context['variance'])
    selected_age = torch.einsum('bhal,bh->bal', weight, age)
    valid = (mass > 1e-08) & mask[..., None]
    movement = context['now_path'][:, :, None, None] - past_path
    expected = torch.einsum('bkqaz,bkalz->bkqal', gradient, movement)
    raw = rssi[:, None, None, :, None] - past_rssi[:, None, None]
    corrected = raw - delta[:, :, None, :, None] + past_delta[:, :, None]
    residual = corrected - expected
    old_axis = context['axes']
    now_axis = context['now_axis'][:, None]
    cosine = (old_axis * now_axis).sum(-1) / (old_axis.norm(dim=-1) * now_axis.norm(dim=-1)).clamp_min(1e-06)
    cosine = torch.einsum('bhal,bh->bal', weight, cosine)
    uncertainty = (context['now_variance'][..., None, None] + past_variance).clamp_min(1e-06).sqrt()
    shape = expected.shape
    channels = [raw.clamp(-30, 30) / 10, corrected.clamp(-30, 30) / 10, expected.clamp(-30, 30) / 10, residual.clamp(-30, 30) / 10, selected_age[:, None, None] / 2, cosine[:, None, None], movement.norm(dim=-1)[:, :, None].clamp_max(10) / 2, uncertainty[:, :, None] / 4, valid[:, None, None].float()]
    ap = torch.stack([value.expand(shape) for value in channels], -1) * valid[:, None, None, ..., None]
    ap = ap.reshape(b, k, q, a, 9 * len(lags))
    left, right = torch.triu_indices(a, a, 1, device=gradient.device)
    pair_mask = (valid[..., left, :] & valid[..., right, :])[:, None, None]
    raw = raw.expand(shape)
    pair = torch.stack(((raw[..., left, :] - raw[..., right, :]).clamp(-30, 30) / 10, (expected[..., left, :] - expected[..., right, :]).clamp(-30, 30) / 10, (residual[..., left, :] - residual[..., right, :]).clamp(-30, 30) / 10, pair_mask.expand(b, k, q, len(left), len(lags)).float()), -1)
    pair = pair * pair_mask[..., None]
    return (ap, pair.reshape(b, k, q, len(left), 4 * len(lags)))

def predictive_transfer(rssi, mask, field, device_mean, device_covariance, offset, sigma, outlier, pose_covariance, update=False, robust=False):
    basis = field['basis']
    valid = mask.float()
    noise = torch.diag_embed(field['map_variance'] + sigma.square()) + field['gradient'] @ pose_covariance @ field['gradient'].transpose(-1, -2)
    if not robust:
        scale = (1 - outlier).clamp_min(0.05).rsqrt()
        noise = noise * scale[..., None] * scale[..., None, :]
    design = basis * valid[..., None]
    noise = noise * valid[..., None] * valid[..., None, :] + torch.diag_embed(1 - valid)
    prediction = field['mean'] + (basis @ device_mean[..., None]).squeeze(-1) + offset
    residual = (rssi - prediction) * valid
    covariance = stabilize(design @ device_covariance @ design.transpose(-1, -2) + noise)
    score = gaussian_log_prob(residual, covariance) + 0.5 * (1 - valid).sum(-1) * math.log(2 * math.pi)
    responsibility = None
    if robust:
        score, responsibility = robust_radio_score(score, rssi, mask, prediction, covariance, outlier)
    if not update:
        return score
    if responsibility is not None:
        noise = noise / responsibility.clamp_min(0.001)[..., None, None]
        covariance = stabilize(design @ device_covariance @ design.transpose(-1, -2) + noise)
    cross = device_covariance @ design.transpose(-1, -2)
    gain = torch.linalg.solve(covariance, cross.transpose(-1, -2)).transpose(-1, -2)
    mean = device_mean + (gain @ residual[..., None]).squeeze(-1)
    transform = torch.eye(6, device=basis.device) - gain @ design
    covariance = stabilize(transform @ device_covariance @ transform.transpose(-1, -2) + gain @ noise @ gain.transpose(-1, -2))
    return (score, mean, covariance)
