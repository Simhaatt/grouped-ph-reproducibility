"""Training loop for DiscreteHazardKAN."""

from __future__ import annotations

import copy

import numpy as np
import torch

from .data import SurvData
from .likelihood import make_targets, nll


def to_tensors(d: SurvData, device="cpu"):
    X = torch.as_tensor(d.X, dtype=torch.float32, device=device)
    mask, y = make_targets(d.bin_idx, d.event, d.n_bins, d.entry_idx)
    return X, mask.to(device), y.to(device)


def fit(
    model,
    data: SurvData,
    *,
    epochs: int = 400,
    lr: float = 0.02,
    weight_decay: float = 0.0,
    batch_size: int | None = None,
    val_frac: float = 0.2,
    patience: int = 40,
    l1: float = 0.0,
    entropy: float = 1.0,
    smooth: float = 0.0,
    grid_update_epochs: tuple[int, ...] = (),
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
):
    """Adam on the exact discrete likelihood, early-stopped on validation NLL.

    ``grid_update_epochs`` refits each KAN layer's knots to the empirical
    distribution of its inputs (``KAN.update_grids``), which least-squares
    refits the spline coefficients so the fitted function is preserved.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)

    X, mask, y = to_tensors(data, device)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]
    va = torch.as_tensor(va, device=device)
    tr = torch.as_tensor(tr, device=device)

    model.set_standardization(X[tr])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_state, bad = float("inf"), None, 0
    history = {"train": [], "val": []}

    for ep in range(epochs):
        if ep in grid_update_epochs:
            model.refit_grids(X[tr])

        model.train()
        idx_batches = (
            [tr] if batch_size is None
            else torch.split(tr[torch.randperm(len(tr), device=device)], batch_size)
        )
        tot = 0.0
        for b in idx_batches:
            opt.zero_grad()
            loss = nll(model(X[b]), mask[b], y[b], model.link)
            obj = loss + model.penalty(X[b], l1=l1, entropy=entropy, smooth=smooth)
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        train_nll = tot / len(tr)

        model.eval()
        with torch.no_grad():
            val_nll = float(nll(model(X[va]), mask[va], y[va], model.link)) if n_val else train_nll
        history["train"].append(train_nll)
        history["val"].append(val_nll)

        if val_nll < best - 1e-6:
            best, bad = val_nll, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 50 == 0:
            print(f"  ep {ep:>4}  train {train_nll:.5f}  val {val_nll:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best
    history["epochs_run"] = ep + 1
    return model, history
