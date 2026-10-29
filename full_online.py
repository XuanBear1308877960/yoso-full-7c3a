import copy, json, math, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from online_m2 import RegionalCalibration

def load_assets(root):
    from radio_field import temporal_radio_load_actor as load_actor
    from radio_field import PARENT_RELATIVE
    from radio_field import field
    from fullmodel import FullModel
    project = Path(root) / 'project' if (Path(root) / 'project').is_dir() else Path(root)
    start = time.perf_counter_ns()
    net, cfg, aps, _, _ = load_actor(project, project / PARENT_RELATIVE)
    net.eval().requires_grad_(False)
    prior = json.loads((project / 'runs/direction_v80/trajectory_incumbent_M2_selected_v1/prediction_complete.json').read_text())['source_prior']
    radio = field(project, prior)
    model = FullModel().eval()
    ck = torch.load(project / 'experiments/full_A_20261020/training_v5/trained.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.requires_grad_(False)
    from data_utils import select_indices
    grid = radio.xy[select_indices(radio.xy, 6)].copy()
    from filtering import SpeedHead
    speed_saved = torch.load(project / 'runs/direction_v80/speed_source_v1/trained.pt', map_location='cpu', weights_only=False)
    speed_head = SpeedHead().eval().requires_grad_(False)
    speed_head.load_state_dict({k[len('motion_speed_head.'):]: v for k, v in speed_saved['state_dict'].items() if k.startswith('motion_speed_head.')}, strict=True)
    return dict(net=net, cfg=cfg, aps=aps, prior=prior, radio=radio, model=model, grid=grid, speed_head=speed_head, load_ms=(time.perf_counter_ns() - start) / 1000000.0)

def create_engine(assets, initial_xy, initial_ns):
    from filtering import RadioDirectionEngine
    from encoders import stabilize
    from encoders import radio_tokens

    class Geometry(RadioDirectionEngine):

        def _consume_wifi(self, e):
            if len(self.outputs) < 5:
                return super()._consume_wifi(e)
            output = dict(observation_id=e['observation_id'], timestamp_ns=self.timestamp, xy_m=(self.log_weight.softmax(-1)[:, None] * self.mean[:, :2]).sum(0).tolist())
            self.forecasts.append(dict(observation_id=e['observation_id'], timestamp_ns=self.timestamp))
            self.outputs.append(output)
            return output

    class Online(RadioDirectionEngine):

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.filter_cfg = dict(self.filter_cfg, cs_radio_feedback_policy='calibrated_past_only')
            self.region_centres = np.median(assets['radio'].xy, axis=0)
            self.calibrator = None
            self.steps = 0
            self.raw_seen = [-1] * 4
            self.online_diagnostics = []
            self.is_cavity = False
            self.cavities = []

        def _consume_wifi(self, e):
            if self.steps < 5:
                for ap in e['payload']['ap_observations']:
                    i = self.lookup.get(ap['ap_id_hash'])
                    if i is not None:
                        self.raw_seen[i] = max(self.raw_seen[i], int(ap.get('measurement_timestamp_ns', self.timestamp)))
                out = super()._consume_wifi(e)
                self.steps += 1
                if self.steps == 5:
                    mu, P = self.geometry.initial_device_posterior
                    prior = (mu.clone(), P.clone(), self.geometry.log_weight.clone())
                    self.calibrator = RegionalCalibration(assets['model'], prior, self.net.field.ap_embedding.weight, 48)
                return out
            y = np.zeros(4)
            mask = np.zeros(4, bool)
            budget = np.zeros(4)
            for ap in e['payload']['ap_observations']:
                i = self.lookup.get(ap['ap_id_hash'])
                if i is None:
                    continue
                y[i] = float(ap['rssi_dbm'])
                key = (i, int(ap.get('measurement_timestamp_ns', self.timestamp)))
                if key[1] > self.raw_seen[i]:
                    mask[i] = True
                    self.raw_seen[i] = key[1]
                    budget[i] = 1 / max(1, int(ap.get('rssi_preprocessing', {}).get('history_n', 1)))
            encoded = self._encoded(torch.tensor(mask))
            motion = encoded['motion'][0].detach().numpy()
            geom = self if self.is_cavity else self.geometry
            positions = geom.mean[:, :2].detach().numpy().copy()
            partition = self.geometry
            origin = (partition.log_weight.softmax(-1)[:, None] * partition.mean[:, :2]).sum(0).numpy()
            cell = np.floor(origin / 2).astype(int)
            region = int((int(cell[0]) * 73856093 ^ int(cell[1]) * 19349663) % 4)
            calibration = self.calibrator if self.is_cavity else self.cavities[region].calibrator
            t0 = time.perf_counter_ns()
            post, diag = calibration.cavity(region, geom.log_weight)
            m2_predict_ms = (time.perf_counter_ns() - t0) / 1000000.0
            variance = self.base_variance.copy()
            if calibration.last_sigma is not None:
                variance += np.mean(calibration.last_sigma.numpy() ** 2 + 64 * calibration.last_pi.numpy(), axis=0)
            else:
                variance += 8.0
            t0 = time.perf_counter_ns()
            updated = self.localize(y, mask, budget, motion, post, variance)
            m3_ms = (time.perf_counter_ns() - t0) / 1000000.0
            gm, gB, gg, _ = self.query(positions, geom.current_motion.numpy(), True)
            Pxy = geom.cov[:, :2, :2].numpy()
            projected = np.einsum('kad,kde,kae->ka', gg, Pxy, gg).clip(0)
            pm, pc, pl = self.calibrator.prior
            mu0 = (pl.softmax(-1)[:, None] * pm).sum(0).numpy()
            predicted = gm + np.einsum('kad,d->ka', gB, mu0)
            dr = np.zeros((8, 4)) if not hasattr(self, 'last_y') else np.broadcast_to(y - self.last_y, (8, 4))
            dp = np.zeros_like(predicted) if not hasattr(self, 'last_prediction') else predicted - self.last_prediction
            dt = 0.0 if not hasattr(self, 'last_ns') else (self.timestamp - self.last_ns) / 1000000000.0
            band = np.unique(assets['radio'].band, axis=0, return_inverse=True)[1]
            encoder = np.stack((np.broadcast_to(y / 50, (8, 4)), predicted / 50, (y - predicted) / 10, dr / 10, dp / 10, np.broadcast_to(np.log1p(self.base_variance) / 5, (8, 4)), gg[..., 0] / 10, gg[..., 1] / 10, projected / 100, np.broadcast_to(band, (8, 4)), np.full((8, 4), dt / 10), np.broadcast_to(budget, (8, 4))), -1)
            t0 = time.perf_counter_ns()
            if self.is_cavity:
                self.calibrator.add(dict(region=region, y=torch.tensor(y), mean=torch.tensor(gm), basis=torch.tensor(gB), variance=torch.tensor(self.base_variance[None] + projected), mask=torch.tensor(budget), encoder=torch.tensor(encoder), oid=e['observation_id']))
            m2_update_ms = (time.perf_counter_ns() - t0) / 1000000.0
            self.last_y = y.copy()
            self.last_prediction = predicted.copy()
            self.last_ns = self.timestamp
            now = self.timestamp / 1000000000.0
            if not hasattr(self, 'token_last_seen'):
                self.token_last_seen = np.full(4, -1000000000.0)
                for mm, tt in zip(self.mask_history, self.times):
                    self.token_last_seen = np.where(mm, tt, self.token_last_seen)
            prev = self.rssi_history[-1] if self.rssi_history else np.zeros(4)
            prevmask = self.mask_history[-1] if self.mask_history else np.zeros(4, bool)
            gap = now - self.times[-1] if self.times else 0.0
            delta = np.where(mask & prevmask, y - prev, 0.0)
            age = np.minimum(30.0, now - self.token_last_seen)
            token = np.stack((y / 100, delta / 20, abs(delta) / 20, np.full(4, gap / 5), age / 10, mask, prevmask, np.full(4, float(0 < gap < 10))), -1)
            self.token_last_seen = np.where(mask, now, self.token_last_seen)
            self.rssi_history.append(y)
            self.mask_history.append(mask)
            self.times.append(now)
            self.radio.append(token.astype(np.float32))
            self.steps += 1
            weight = self.log_weight.softmax(-1)
            xy = (weight[:, None] * self.mean[:, :2]).sum(0)
            out = dict(observation_id=e['observation_id'], timestamp_ns=self.timestamp, xy_m=xy.tolist(), known_initial_position_only=True, mode_weights=weight.tolist(), online=True)
            self.outputs.append(out)
            self.forecasts.append(dict(observation_id=e['observation_id'], timestamp_ns=self.timestamp))
            self.online_diagnostics.append(dict(observation_id=e['observation_id'], M2_predict_ms=m2_predict_ms, M2_increment_ms=m2_update_ms, M3_ms=m3_ms, context_rows=len(self.calibrator.rows), context_updates=self.calibrator.updates, excluded_region=region, **diag, M3_nodes=6, pose_modes=8, solver_iterations=updated))
            return out

        def query(self, xy, motion, derivative=False):
            points = np.repeat(np.asarray(xy), 4, axis=0)
            ctx = np.broadcast_to(motion, (len(points), 32))
            ap = np.tile(np.arange(4), len(xy))
            result = assets['radio'].ap_query(points, ctx, ap, derivative)
            shapes = [(len(xy), 4), (len(xy), 4, 6), (len(xy), 4, 2), (len(xy), 4, 6, 2)]
            return tuple((np.asarray(v).reshape(shape) for v, shape in zip(result, shapes)))

        def evaluate(self, xy, y, mask, budget, motion, post, noise, correction=None):
            m, B, dm, db = self.query(xy, motion, True)
            mu, P, lw = [v.numpy().astype(float) for v in post]
            prediction = m[..., None] + np.einsum('nad,cd->nac', B, mu)
            v = (noise[None, :, None] + np.einsum('nad,cde,nae->nac', B, P, B)).clip(0.25)
            residual = y[None, :, None] - prediction
            ap = -0.5 * (residual ** 2 / v + np.log(v) + math.log(2 * math.pi))
            w = budget * mask
            component = (ap * w[None, :, None]).sum(1) + lw
            lp = logsumexp(component, axis=-1)
            prob = np.exp(component - lp[:, None])
            dp = dm[:, :, None, :] + np.einsum('nadz,cd->nacz', db, mu)
            dv = 2 * np.einsum('nadz,cde,nae->nacz', db, P, B)
            g = (residual[..., None] / v[..., None] * dp + 0.5 * (residual ** 2 / v ** 2 - 1 / v)[..., None] * dv) * w[None, :, None, None]
            grad = np.einsum('nc,nacz->nz', prob, g)
            H = np.einsum('nc,a,nacx,nacy,nac->nxy', prob, w, dp, dp, 1 / v)
            if correction is not None:
                grid, values = correction
                diff = xy[:, None] - grid[None]
                log = -np.square(diff).sum(-1) / (2 * 0.35 ** 2)
                weights = np.exp(log - log.max(-1, keepdims=True))
                weights /= weights.sum(-1, keepdims=True)
                value = weights @ values
                gradient = np.einsum('ng,ng,ngz->nz', weights, values[None] - value[:, None], -diff / 0.35 ** 2)
                lp += value
                grad += gradient
            return (lp, grad, H, (m, B, dm, db, prediction, v, prob))

        def localize(self, y, mask, budget, motion, post, noise):
            old = self.mean.numpy().astype(float)
            cov = self.cov.numpy().astype(float)
            prior = old[:, :2].copy()
            P = cov[:, :2, :2] + np.eye(2) * 1e-05
            precision = np.linalg.inv(P)
            grid = assets['grid']
            _, _, _, f = self.evaluate(grid, y, mask, budget, motion, post, noise)
            m, B, dm, db, prediction, v, prob = f
            mu, C, lw = [z.numpy().astype(float) for z in post]
            weights = np.exp(lw - logsumexp(lw))
            pred = (prediction * weights).sum(-1)
            mapped = np.einsum('nad,cde,nae,c->na', B, C, B, weights)
            total = (v * weights).sum(-1) + ((prediction - pred[..., None]) ** 2 * weights).sum(-1)
            gradient = dm + np.einsum('nadz,cd,c->naz', db, mu, weights)
            ap = np.tile(np.arange(4), 6)
            points = np.repeat(grid, 4, axis=0)
            visibility = assets['radio'].descriptor.basis_inputs_ap(points, np.broadcast_to(motion, (24, 32)), ap)[1][:, 6].reshape(6, 4)
            scalar = np.stack(((y - pred) / np.sqrt(total), np.log(total) / 5, m / 50, gradient[..., 0] / 10, gradient[..., 1] / 10, mapped / 100, visibility, np.ones_like(m)), -1)
            embedding = self.net.field.ap_embedding.weight.detach().numpy()
            features = np.concatenate((scalar, np.broadcast_to(embedding, (6, 4, 8))), -1)
            correction = assets['model'].positioning(torch.tensor(features[None], dtype=torch.float32), torch.tensor(mask[None])).numpy()[0]

            def objective(x):
                lp, g, H, _ = self.evaluate(x, y, mask, budget, motion, post, noise, (grid, correction))
                delta = x - prior
                return (0.5 * np.einsum('nd,nde,ne->n', delta, precision, delta) - 2 * lp, np.einsum('nde,ne->nd', precision, delta) - 2 * g, precision + 2 * H)
            x = prior.copy()
            cost, g, H = objective(x)
            reference = assets['radio'].xy
            reference_lp, _, _, _ = self.evaluate(reference, y, mask, budget, motion, post, noise, (grid, correction))
            difference = reference[None] - prior[:, None]
            global_cost = 0.5 * np.einsum('knd,kde,kne->kn', difference, precision, difference) - 2 * reference_lp[None]
            ix = np.argmin(global_cost, axis=1)
            best = global_cost[np.arange(8), ix]
            take = best < cost
            x[take] = reference[ix[take]]
            cost, g, H = objective(x)
            iterations = 0
            for _ in range(8):
                iterations += 1
                step = np.linalg.solve(H + np.eye(2) * 1e-05, -g[..., None])[..., 0]
                scale = np.minimum(1.0, 2 / np.linalg.norm(step, axis=-1).clip(1e-08))
                step *= scale[:, None]
                if np.linalg.norm(step, axis=-1).max() < 0.001:
                    break
                active = np.ones(8, bool)
                accepted = np.zeros(8, bool)
                for factor in [1.0, 0.5, 0.25, 0.125]:
                    trial = x + factor * step
                    value, _, _ = objective(trial)
                    take = active & (value < cost)
                    x[take] = trial[take]
                    cost[take] = value[take]
                    accepted |= take
                    active &= ~take
                    if not active.any():
                        break
                if not accepted.any():
                    break
                cost, g, H = objective(x)
            _, _, H = objective(x)
            newP = np.linalg.inv(H + np.eye(2) * 1e-05)
            gain = cov[:, :, :2] @ precision
            change = np.einsum('kdz,kz->kd', gain, x - prior)
            mean = old + change
            newcov = cov + gain @ (newP - P) @ gain.transpose(0, 2, 1)
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.cov = stabilize(torch.tensor(newcov, dtype=torch.float32))
            lp, _, _, _ = self.evaluate(x, y, mask, budget, motion, post, noise, (grid, correction))
            self.log_weight = (self.log_weight + torch.tensor(2 * lp, dtype=torch.float32)).log_softmax(-1)
            angle = np.arange(6) * math.pi / 3
            unit = np.column_stack((np.cos(angle), np.sin(angle)))
            root = np.linalg.cholesky(newP + np.eye(2) * 1e-05)
            points = x[:, None] + np.sqrt(2) * np.einsum('kdz,qz->kqd', root, unit)
            self.last_wifi_distribution = dict(time=self.timestamp / 1000000000.0, points=torch.tensor(points, dtype=torch.float32), probability=torch.ones(8, 6) / 6)
            return iterations
    cfg = copy.deepcopy(assets['cfg'])
    cfg['filter'].update(direction_policy='radio_frame_independent_cs', export_trajectory_radio_factors=False)
    args = (assets['net'], assets['aps'], cfg['anchors'], initial_xy, initial_ns, cfg['filter'], False, 1, 'mixed')
    geom = Geometry(*args)
    main = Online(*args)
    main.geometry = geom
    params = np.asarray(assets['prior']['temporal_RSSI_parameters'])
    main.base_variance = np.exp(2 * params[:4]) + np.exp(2 * params[4:8]) + np.exp(2 * params[9:13])
    return (main, geom)

class Stream:

    def __init__(self, assets, initial_xy, initial_ns):
        factory = create_engine
        t = time.perf_counter_ns()
        self.main, self.geometry = factory(assets, initial_xy, initial_ns)
        self.initialization_ms = (time.perf_counter_ns() - t) / 1000000.0
        self.last_ns = initial_ns
        self.count = 0
        self.assets = assets
        self.recent_ids = []
        self.workers = int(os.environ.get('YOMO_MOBILE_ONLINE_CONTEXT_WORKERS', '4'))
        self.pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    @torch.no_grad()
    def consume(self, event):
        ns = int(event['monotonic_timestamp_ns'])
        if ns < self.last_ns:
            raise ValueError('Out-of-order event')
        self.last_ns = ns
        if any((k in event.get('payload', {}) for k in ['gt_xy_m', 'true_position', 'true_velocity'])):
            raise ValueError('Truth forbidden')
        if int(event.get('payload', {}).get('observation_available_ns', ns)) > ns:
            raise ValueError('Future observation availability')
        row = event
        if event['kind'] == 'wifi' and self.count >= 5:
            row = dict(event, payload=dict(event['payload'], ap_observations=[]))
        self.geometry.consume(row)
        if self.count >= 5:
            origin = (self.geometry.log_weight.softmax(-1)[:, None] * self.geometry.mean[:, :2]).sum(0).numpy()
            cell = np.floor(origin / 2).astype(int)
            region = int((int(cell[0]) * 73856093 ^ int(cell[1]) * 19349663) % 4)
            pending = []
            for j, cavity in enumerate(self.main.cavities):
                selected = event
                if event['kind'] == 'wifi' and j == region:
                    selected = dict(event, payload=dict(event['payload'], ap_observations=[]))
                if self.pool:
                    pending.append(self.pool.submit(cavity.consume, selected))
                else:
                    cavity.consume(selected)
            for future in pending:
                future.result()
        result = self.main.consume(event)
        if event['kind'] == 'wifi':
            self.count += 1
            if self.count == 5:
                memo = {id(self.assets): self.assets, id(self.main.net): self.main.net, id(self.assets['net']): self.assets['net'], id(self.assets['model']): self.assets['model'], id(self.geometry): self.geometry}
                for tensor in list(self.assets['net'].parameters()) + list(self.assets['net'].buffers()):
                    memo[id(tensor)] = tensor
                for _ in range(4):
                    cavity = copy.deepcopy(self.main, memo.copy())
                    cavity.is_cavity = True
                    cavity.cavities = []
                    if hasattr(cavity, 'fork_runtime'):
                        cavity.fork_runtime()
                    self.main.cavities.append(cavity)
        self.recent_ids.append(event['observation_id'])
        if len(self.recent_ids) > 4096:
            old = self.recent_ids.pop(0)
            for engine in [self.main, self.geometry] + self.main.cavities:
                engine.ledger.discard(old)
        for engine in [self.main, self.geometry] + self.main.cavities:
            for name, limit in [('outputs', 16), ('forecasts', 16), ('motion_tokens', 16), ('rssi_history', 17), ('mask_history', 17), ('times', 17), ('cs_diagnostics', 16), ('radio_diagnostics', 16), ('cs_training_rows', 16), ('online_diagnostics', 16), ('innovations', 16), ('innovation_designs', 16), ('innovation_device_means', 16), ('predicted_poses', 16), ('pose_variances', 16), ('wifi_axes', 16), ('motion_paths', 16), ('delta_history', 16), ('solver_diagnostics', 16)]:
                value = getattr(engine, name, None)
                if value is not None and len(value) > limit:
                    setattr(engine, name, value[-limit:])
            if len(engine.radio) > 21:
                engine.radio = engine.radio[:5] + engine.radio[-16:]
            for anchor in engine.cs_history:
                engine.cs_history[anchor] = engine.cs_history[anchor][-16:]
        return result

    def close(self):
        if self.pool:
            self.pool.shutdown()
            self.pool = None
