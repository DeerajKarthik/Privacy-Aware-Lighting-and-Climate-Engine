"""Federated training engine: central (B2) / local (B3) / FedAvg (B4) / DP-FedAvg (B5) / DP-FedPer (PALACE). Day-level DP-SGD via Opacus,
one persistent PrivacyEngine+RDP accountant per client across ALL rounds. Privacy unit = one site-day record (see RESULTS notes on 48h-context overlap)."""
from __future__ import annotations
import copy, math, time
from pathlib import Path
import numpy as np, torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from palace.fl.data import build_records, DayDataset
from palace.models.occupancy_tcn import OccTCN, loss_fn, n_params
from palace.dp.accountant import delta_for, cumulative_epsilon

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEAD_KEYS = ("head.", "arr.")

def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_model(cfg):
    m = cfg["model"]; return OccTCN(ch=m["channels"], k=m["kernel"], dil=tuple(m["dilations"])).to(DEV)

class Client:
    def __init__(self, site, cfg, root, eps, seed, rounds):
        self.site, self.cfg = site, cfg; d = cfg["data"]; t = cfg["train"]
        self.train = build_records(root, site, "train", d["max_train_days_per_client"]); self.val = build_records(root, site, "val", d["max_eval_days"]); self.test = build_records(root, site, "test", d["max_eval_days"])
        self.ds = DayDataset(self.train, DEV); self.n = len(self.ds); self.q = min(1.0, t["batch_days"] / max(self.n, 1)); self.eps_target = eps
        self.model = None; self.opt = None; self.engine = None; self.loader = None; self.sigma = None; self.delta = delta_for(max(self.n, 1)); self.seed = seed
        self.steps_per_epoch = max(1, int(round(1 / self.q))); self.total_steps = rounds * t["local_epochs"] * self.steps_per_epoch

    def setup(self, model, dp: bool):
        t = self.cfg["train"]; self.model = model; self.opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
        loader = torch.utils.data.DataLoader(self.ds, batch_size=max(1, int(round(self.q * self.n))), shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(self.seed))
        if dp:
            self.engine = PrivacyEngine(accountant="rdp")
            self.sigma = get_noise_multiplier(target_epsilon=self.eps_target, target_delta=self.delta, sample_rate=self.q, steps=self.total_steps, accountant="rdp")
            self.model, self.opt, self.loader = self.engine.make_private(module=model, optimizer=self.opt, data_loader=loader, noise_multiplier=self.sigma,
                                                                         max_grad_norm=t["max_grad_norm"], poisson_sampling=True)
        else: self.loader = loader

    def raw(self): return self.model._module if hasattr(self.model, "_module") else self.model

    def local_train(self):
        t = self.cfg["train"]; self.model.train(); tot = 0.0; nb = 0
        for _ in range(t["local_epochs"]):
            for X, Y, M, A in self.loader:
                if len(X) == 0: continue
                self.opt.zero_grad(); loss = loss_fn(self.model(X), Y, M, A, t["focal_weight"], t["focal_gamma"]); loss.backward(); self.opt.step(); tot += float(loss); nb += 1
        return tot / max(nb, 1)

    def eps_spent(self):
        return float("inf") if self.engine is None else cumulative_epsilon(self.engine.accountant, self.delta)

def fedavg(states, weights, keys):
    w = np.array(weights, float) / np.sum(weights); out = {}
    for k in keys: out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states))
    return out

def run_fl(sites, cfg, root, mode, eps=float("inf"), seed=0, rounds=None, personalise=False, local_head_finetune=False, verbose=False):
    """mode in {central, local, fedavg, dpfedavg, dpfedper}. Returns dict(models per site, eps per site, stats)."""
    seed_all(seed); rounds = rounds or cfg["train"]["rounds"]; dp = mode.startswith("dp")
    cl = [Client(s, cfg, root, eps, seed + i, rounds) for i, s in enumerate(sites)]
    g = make_model(cfg); stats = dict(params=n_params(g), mode=mode, eps_target=eps, rounds=rounds, n_clients=len(cl))
    torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None; t0 = time.time()
    if mode == "central":
        pool = torch.utils.data.ConcatDataset([c.ds for c in cl]); opt = torch.optim.Adam(g.parameters(), lr=cfg["train"]["lr"]); n = len(pool); bs = cfg["train"]["batch_days"]
        steps = rounds * cfg["train"]["local_epochs"] * int(math.ceil(n / bs)); g.train(); ld = torch.utils.data.DataLoader(pool, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed)); it = iter(ld)
        for _ in range(steps // len(cl) * 1 if False else int(rounds * cfg["train"]["local_epochs"] * math.ceil(n / bs))):
            try: X, Y, M, A = next(it)
            except StopIteration: it = iter(ld); X, Y, M, A = next(it)
            opt.zero_grad(); loss_fn(g(X), Y, M, A, cfg["train"]["focal_weight"], cfg["train"]["focal_gamma"]).backward(); opt.step()
        models = {c.site: g for c in cl}; stats["comm_mb"] = 0.0
    elif mode == "local":
        models = {}
        for c in cl:
            m = copy.deepcopy(g); c.setup(m, False)
            for _ in range(rounds): c.local_train()
            models[c.site] = c.raw()
        stats["comm_mb"] = 0.0
    else:
        shared_keys = [k for k in g.state_dict() if not (mode == "dpfedper" and k.startswith(HEAD_KEYS))]
        for c in cl: c.setup(copy.deepcopy(g), dp)
        gstate = copy.deepcopy(g.state_dict()); nb = sum(g.state_dict()[k].numel() for k in shared_keys) * 4
        for r in range(rounds):
            states, ws = [], []
            for c in cl:
                c.raw().load_state_dict({**c.raw().state_dict(), **{k: gstate[k] for k in shared_keys}}); loss = c.local_train()
                states.append({k: v.detach().clone() for k, v in c.raw().state_dict().items()}); ws.append(c.n)
            gstate.update(fedavg(states, ws, shared_keys))
            if verbose: print(f"round {r+1}/{rounds}", flush=True)
        for c in cl: c.raw().load_state_dict({**c.raw().state_dict(), **{k: gstate[k] for k in shared_keys}})
        models = {c.site: c.raw() for c in cl}; stats["comm_mb"] = 2 * nb * len(cl) * rounds / 1e6
        stats["sigma"] = {c.site: c.sigma for c in cl}; stats["delta"] = {c.site: c.delta for c in cl}
        stats["eps_spent"] = {c.site: c.eps_spent() for c in cl}; stats["q"] = {c.site: c.q for c in cl}; stats["steps"] = {c.site: c.total_steps for c in cl}
    stats["train_wall_s"] = time.time() - t0
    stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else 0.0
    return dict(models=models, clients={c.site: c for c in cl}, stats=stats)
