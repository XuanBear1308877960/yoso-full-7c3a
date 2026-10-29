from collections import deque
import numpy as np
import torch

class RegionalCalibration:

    def __init__(self, model, prior, embedding, window=48):
        self.model = model
        self.prior = tuple((t.detach().double().clone() for t in prior))
        self.embedding = embedding
        self.window = window
        self.rows = deque()
        self.history = deque(maxlen=15)
        self.J = torch.zeros(4, 8, 6, 6, dtype=torch.float64)
        self.b = torch.zeros(4, 8, 6, dtype=torch.float64)
        self.c = torch.zeros(4, 8, dtype=torch.float64)
        self.ld = torch.zeros(4, 8, dtype=torch.float64)
        self.region_count = np.zeros(4, int)
        self.updates = 0
        self.evictions = 0
        self.last_sigma = None
        self.last_pi = None

    def _change(self, row, sign):
        r = row['region']
        self.J[r] += sign * row['J']
        self.b[r] += sign * row['b']
        self.c[r] += sign * row['c']
        self.ld[r] += sign * row['ld']
        self.region_count[r] += sign * int(bool(row['mask'].any()))

    def add(self, row):
        self.history.append(row['encoder'].float())
        x = torch.stack(list(self.history), 1)
        if not hasattr(self, 'mask_history'):
            self.mask_history = deque(maxlen=15)
        self.mask_history.append(row['mask'])
        mask = torch.stack(list(self.mask_history), 0)
        sigma, pi = self.model.calibration.reliability(x, mask[None].expand(8, -1, -1).to(x.dtype))
        sigma = sigma[:, -1].double()
        pi = pi[:, -1].double()
        self.last_sigma = sigma
        self.last_pi = pi
        variance = (row['variance'].double() + sigma.square() + 64 * pi).clamp_min(0.25)
        w = row['mask'][None].double() / variance
        B = row['basis'].double()
        z = row['y'][None].double() - row['mean'].double()
        row.update(J=torch.einsum('ka,kad,kae->kde', w, B, B), b=torch.einsum('ka,kad,ka->kd', w, B, z), c=(w * z.square()).sum(-1), ld=(row['mask'][None] * variance.log()).sum(-1), variance=variance)
        self._change(row, 1)
        self.rows.append(row)
        self.updates += 1
        if len(self.rows) > self.window:
            self._change(self.rows.popleft(), -1)
            self.evictions += 1

    def posterior(self, regions):
        pm, pc, pl = self.prior
        eye = torch.eye(6, dtype=torch.float64)
        P0 = torch.linalg.inv(pc)
        J0 = self.J[regions].sum(0)
        b0 = self.b[regions].sum(0)
        c = self.c[regions].sum(0)
        ld = self.ld[regions].sum(0)
        gate = torch.linalg.solve(J0 + 0.02 * eye, J0)
        J = J0 @ gate
        J = (J + J.transpose(-1, -2)) / 2
        b = (gate.transpose(-1, -2) @ b0[..., None])[..., 0]
        precision = P0[None] + J[:, None]
        L = torch.linalg.cholesky(precision + 1e-08 * eye)
        covariance = torch.cholesky_inverse(L)
        rhs = (P0 @ pm[..., None])[..., 0][None] + b[:, None]
        mu = torch.cholesky_solve(rhs[..., None], L)[..., 0]
        delta = mu - pm[None]
        residual = c[:, None] - 2 * (mu * b0[:, None]).sum(-1) + torch.einsum('kcd,kde,kce->kc', mu, J0, mu)
        cost = 0.5 * (residual + torch.einsum('kcd,cde,kce->kc', delta, P0, delta) + torch.linalg.slogdet(pc)[1][None] + 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1) + torch.einsum('kcde,kced->kc', covariance, P0[None] + J0[:, None]) - 6 + ld[:, None])
        return (mu, covariance, (pl[None] - cost).log_softmax(-1))

    def cavity(self, region, geometry_log_weight):
        from fullmodel import predictive_log_probability
        regions = [i for i in range(4) if i != region and self.region_count[i] > 0]
        if not regions:
            return (tuple((v.float() for v in self.prior)), dict(other_regions=0, message_rounds=0))
        posts = [self.posterior([r]) for r in regions]
        R = len(regions)
        edges = torch.zeros(R, R, 8, 8, dtype=torch.float64)
        for i in range(R):
            for j in range(i + 1, R):
                a = [v for v in self.rows if v['region'] == regions[j]]
                b = [v for v in self.rows if v['region'] == regions[i]]

                def predict(post, rr):
                    y = torch.stack([v['y'] for v in rr])
                    m = torch.stack([v['mean'] for v in rr], 1)
                    B = torch.stack([v['basis'] for v in rr], 1)
                    var = torch.stack([v['variance'] for v in rr], 1)
                    mask = torch.stack([v['mask'] for v in rr])
                    values = []
                    for h in range(8):
                        lp = predictive_log_probability(y[None], m, B, var, tuple((v[h] for v in post)))
                        values.append((lp * mask[None]).sum((1, 2)) / mask.sum().clamp_min(1))
                    return torch.stack(values)
                edge = 0.5 * (predict(posts[i], a) + predict(posts[j], b).T)
                edges[i, j] = edge
                edges[j, i] = edge.T
        node = geometry_log_weight.double().log_softmax(-1)[None].expand(R, -1) / R
        messages = torch.zeros(R, R, 8, dtype=torch.float64)
        for _ in range(2):
            new = torch.zeros_like(messages)
            for i in range(R):
                for j in range(R):
                    if i != j:
                        new[i, j] = torch.logsumexp((node[i] + messages[:, i].sum(0) - messages[j, i])[:, None] + edges[i, j], 0).log_softmax(-1)
            messages = new
        path = (node + messages.sum(0)).log_softmax(-1).mean(0).log_softmax(-1)
        mu, cov, lw = self.posterior(regions)
        mass = (lw + path[:, None]).flatten()
        ix = mass.topk(8).indices
        return ((mu.flatten(0, 1)[ix].float(), cov.flatten(0, 1)[ix].float(), mass[ix].log_softmax(-1).float()), dict(other_regions=R, message_rounds=2))
