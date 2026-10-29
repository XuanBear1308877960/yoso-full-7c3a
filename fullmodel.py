import math
import torch
from torch import nn
from torch.nn import functional as F

class CausalBlock(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.depth = nn.Conv1d(channels, channels, 3, dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(channels, 2 * channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        y = F.glu(self.mix(self.depth(F.pad(x, (self.left, 0)))), dim=1)
        return self.norm((x + y).transpose(1, 2)).transpose(1, 2)

class RadioReliability(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(12, 24)
        self.blocks = nn.Sequential(*[CausalBlock(24, d) for d in (1, 2, 4)])
        self.set_context = nn.Sequential(nn.Linear(48, 24), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(48, 16), nn.SiLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.constant_(self.output[-1].bias, -2.0)

    def forward(self, x, mask):
        k, t, a, _ = x.shape
        h = self.input(x * mask[..., None]).permute(0, 2, 3, 1)
        h = self.blocks(h.reshape(k * a, 24, t))
        h = h.reshape(k, a, 24, t).permute(0, 3, 1, 2)
        w = mask[..., None].to(h.dtype)
        n = w.sum(2).clamp_min(1.0)
        mean = (h * w).sum(2) / n
        var = ((h - mean[:, :, None]).square() * w).sum(2) / n
        context = self.set_context(torch.cat((mean, (var + 1e-06).sqrt()), -1))
        z = self.output(torch.cat((h, context[:, :, None].expand_as(h)), -1))
        return (0.5 + 7.5 * z[..., 0].sigmoid(), 0.01 + 0.39 * z[..., 1].sigmoid())

def posterior(y, mean, basis, valid, variance, prior, whitening=None, return_evidence=False):
    if whitening is None:
        B = basis.double().flatten(1, 2)
        z = (y[None] - mean).double().flatten(1)
        w = (valid[None] / variance.clamp_min(0.25)).double().flatten(1)
        noise_logdet = (variance.double().clamp_min(0.25).log() * valid[None]).sum((1, 2))
    else:
        z, B, row_valid, noise_logdet, gain = whitening
        z, B = (z.double(), B.double())
        w = row_valid[None].double() / gain.double().clamp_min(0.01)
        noise_logdet = noise_logdet.double() + (gain.double().clamp_min(0.01).log() * row_valid[None]).sum(-1)
    if B.shape[1] == 0:
        pm, pc, pl = prior
        k = len(B)
        result = (pm[None].expand(k, -1, -1), pc[None].expand(k, -1, -1, -1), pl.log_softmax(-1)[None].expand(k, -1), torch.zeros((k, 6, 6), device=B.device, dtype=pm.dtype))
        return result + (torch.zeros(k, device=B.device, dtype=pm.dtype),) if return_evidence else result
    J = torch.einsum('kn,knd,kne->kde', w, B, B)
    b = torch.einsum('kn,knd,kn->kd', w, B, z)
    eye = torch.eye(6, device=B.device, dtype=B.dtype)
    gate = torch.linalg.solve(J + 0.02 * eye, J)
    raw_information = J
    J = J @ gate
    J = 0.5 * (J + J.transpose(-1, -2))
    b = torch.einsum('kde,ke->kd', gate.transpose(-1, -2), b)
    pm, pc, pl = prior
    pm, pc, pl = (pm.double(), pc.double(), pl.double())
    precision0 = torch.linalg.inv(pc)
    precision = precision0[None] + J[:, None]
    L = torch.linalg.cholesky(precision + 1e-08 * eye)
    covariance = torch.cholesky_inverse(L)
    rhs = torch.einsum('cde,ce->cd', precision0, pm)[None] + b[:, None]
    mu = torch.cholesky_solve(rhs[..., None], L).squeeze(-1)
    residual = z[:, None] - torch.einsum('knd,kcd->kcn', B, mu)
    delta = mu - pm[None]
    cost = 0.5 * ((residual.square() * w[:, None]).sum(-1) + torch.einsum('kcd,cde,kce->kc', delta, precision0, delta) + torch.linalg.slogdet(pc)[1][None] + 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1) + torch.einsum('kcde,kced->kc', covariance, precision0[None] + raw_information[:, None]) - 6 + noise_logdet[:, None])
    component_mass = pl[None] - cost
    logw = component_mass.log_softmax(-1)
    result = (mu.float(), covariance.float(), logw.float(), J.float())
    return result + (component_mass.logsumexp(-1).float(),) if return_evidence else result

def predictive_log_probability(y, mean, basis, variance, post):
    mu, cov, logw = post[:3]
    prediction = mean[..., None] + torch.einsum('...d,cd->...c', basis, mu)
    v = variance[..., None] + torch.einsum('...d,cde,...e->...c', basis, cov, basis)
    v = v.clamp_min(0.25)
    lp = -0.5 * ((y[..., None] - prediction).square() / v + v.log() + math.log(2 * math.pi))
    return torch.logsumexp(lp + logw, -1)

class CounterfactualCalibration(nn.Module):

    def __init__(self):
        super().__init__()
        self.reliability = RadioReliability()

    def forward(self, d):
        sigma, pi = self.reliability(d['encoder'], d['support_mask'][None].expand(d['encoder'].shape[:-1]))
        variance = d['support_variance'] + sigma.square() + 64 * pi
        prior = (d['prior_mean'], d['prior_covariance'], d['prior_log_weight'])

        def get_posterior(mask, region=4, evidence=False):
            whitening = None
            if 'white_z' in d:
                scan, ap = (d['white_scan'][region], d['white_AP'][region])
                gain = 1 + (sigma[:, scan, ap].square() + 64 * pi[:, scan, ap]) / d['support_marginal_variance'][scan, ap][None].clamp_min(0.25)
                whitening = (d['white_z'][:, region], d['white_basis'][:, region], d['white_valid'][region], d['white_logdet'][:, region], gain)
            return posterior(d['support_y'], d['support_mean'], d['support_basis'], mask, variance, prior, whitening, evidence)
        regions = torch.unique(d['region'][d['support_mask'].any(-1)])
        posts, masks = ([], [])
        for region in regions:
            mask = d['support_mask'] * (d['region'] == region)[:, None]
            if 'message_mask' in d:
                mask = d['message_mask'][int(region)].to(mask.dtype)
            posts.append(get_posterior(mask, int(region)))
            masks.append(mask)
        k = d['support_mean'].shape[0]
        r = len(posts)
        if r:
            edges = torch.zeros((r, r, k, k), device=variance.device)
            for i in range(r):
                for j in range(i + 1, r):
                    forward, reverse = ([], [])
                    for h in range(k):
                        mi, bi, vi = (d['support_mean'], d['support_basis'], variance)
                        mj, bj, vj = (mi, bi, vi)
                        if 'message_offset' in d:
                            ri, rj = (int(regions[i]), int(regions[j]))
                            mi = mi + d['message_offset'][ri, rj, h][None]
                            bi = bi - d['message_basis_offset'][ri, rj, h][None]
                            vi = d['message_variance'][ri, rj, h] + sigma.square() + 64 * pi
                            mj = mj + d['message_offset'][rj, ri, h][None]
                            bj = bj - d['message_basis_offset'][rj, ri, h][None]
                            vj = d['message_variance'][rj, ri, h] + sigma.square() + 64 * pi
                        lp = predictive_log_probability(d['support_y'][None], mi, bi, vi, tuple((x[h] for x in posts[i][:3])))
                        forward.append((lp * masks[j][None]).sum((1, 2)) / masks[j].sum().clamp_min(1))
                        lp = predictive_log_probability(d['support_y'][None], mj, bj, vj, tuple((x[h] for x in posts[j][:3])))
                        reverse.append((lp * masks[i][None]).sum((1, 2)) / masks[i].sum().clamp_min(1))
                    edge = 0.5 * (torch.stack(forward) + torch.stack(reverse).T)
                    edge = edge - d['continuity_cost']
                    edges[i, j] = edge
                    edges[j, i] = edge.T
            node = d['geometry_log_weight'][None].expand(r, -1) / max(r, 1)
            messages = torch.zeros_like(edges[..., 0])
            for _ in range(2):
                new = torch.zeros_like(messages)
                for i in range(r):
                    for j in range(r):
                        if i == j:
                            continue
                        cavity = node[i] + messages[:, i].sum(0) - messages[j, i]
                        new[i, j] = torch.logsumexp(cavity[:, None] + edges[i, j], 0).log_softmax(-1)
                messages = new
            belief = (node + messages.sum(0)).log_softmax(-1)
            path_logw = belief.mean(0).log_softmax(-1)
        else:
            path_logw = d['geometry_log_weight'].log_softmax(-1)
        mu, cov, logw, information = get_posterior(d['support_mask'])
        weight = (logw + path_logw[:, None]).flatten()
        selected = weight.topk(min(8, len(weight))).indices
        weight = weight[selected].log_softmax(-1)
        return ((mu.flatten(0, 1)[selected], cov.flatten(0, 1)[selected], weight), dict(sigma=sigma, outlier=pi, path_probability=path_logw.exp(), information=information, query_noise=(sigma.square() + 64 * pi).mean((0, 1)), regions=r, message_rounds=2, posterior_hypothesis=selected // logw.shape[-1]))

class SpatialReadout(nn.Module):

    def __init__(self):
        super().__init__()
        self.ap = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.normal_(self.readout[-1].weight, std=0.01)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, x, mask):
        h = self.ap(x)
        w = mask[:, None, :, None].to(h.dtype)
        n = w.sum(-2).clamp_min(1.0)
        mean = (h * w).sum(-2) / n
        var = ((h - mean[..., None, :]).square() * w).sum(-2) / n
        value = self.readout(torch.cat((mean, (var + 1e-06).sqrt()), -1)).squeeze(-1)
        return 4 * torch.tanh(value / 4)

class FullModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.calibration = CounterfactualCalibration()
        self.positioning = SpatialReadout()

    def forward(self, d):
        post, diagnostics = self.calibration(d)
        mu, cov, logw = post
        m, B = (d['query_mean'], d['query_basis'])
        effective_B = B[..., None, :].expand(B.shape[:-1] + (len(mu), 6))
        offset = 0.0
        if 'query_noise_offset' in d:
            h = diagnostics['posterior_hypothesis']
            offset = d['query_noise_offset'][h].permute(1, 2, 0)[:, None]
            effective_B = effective_B - d['query_basis_offset'][h].permute(1, 2, 0, 3)[:, None]
        prediction = m[..., None] + offset + torch.einsum('tqacd,cd->tqac', effective_B, mu)
        mapping_var = torch.einsum('tqacd,cde,tqace->tqac', effective_B, cov, effective_B)
        query_variance = d.get('query_conditional_variance', d['query_variance'])
        if 'query_variance_by_hypothesis' in d:
            h = diagnostics['posterior_hypothesis']
            query_variance = d['query_variance_by_hypothesis'][h].permute(1, 2, 0)[:, None]
        else:
            query_variance = query_variance[:, None, :, None]
        variance = query_variance + diagnostics['query_noise'][None, None, :, None] + mapping_var
        variance = variance.clamp_min(0.25)
        residual = d['query_y'][:, None, :, None] - prediction
        ap_lp = -0.5 * (residual.square() / variance + variance.log() + math.log(2 * math.pi))
        mask = d['query_mask']
        gaussian = torch.logsumexp((ap_lp * mask[:, None, :, None]).sum(-2) + logw, -1)
        weight = logw.exp()
        predicted = (prediction * weight).sum(-1)
        mapped = (mapping_var * weight).sum(-1)
        total = (variance * weight).sum(-1) + ((prediction - predicted[..., None]).square() * weight).sum(-1)
        gradient = d['query_gradient'] + torch.einsum('tqadz,cd,c->tqaz', d['query_basis_gradient'], mu, weight)
        scalar = torch.stack(((d['query_y'][:, None] - predicted) / total.sqrt(), total.log() / 5, m / 50, gradient[..., 0] / 10, gradient[..., 1] / 10, mapped / 100, d['query_visibility'], torch.ones_like(m)), -1)
        emb = d['ap_embedding'][None, None].expand(m.shape + (8,))
        correction = self.positioning(torch.cat((scalar, emb), -1), mask)
        potential = gaussian + correction
        diagnostics.update(correction=correction, gaussian=gaussian, predicted_RSSI=predicted, posterior_mean=mu, posterior_covariance=cov, posterior_log_weight=logw)
        return (potential, diagnostics)
