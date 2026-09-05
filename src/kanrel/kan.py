"""Kolmogorov-Arnold Network layers with B-spline edge functions.

Self-contained (no pykan / efficient-kan dependency). The layer keeps the
per-edge activations phi_{o,i}(x_i) addressable so they can be regularized
and plotted -- that interpretability is the reason a KAN is worth using here
instead of an MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """y_o = sum_i phi_{o,i}(x_i),  phi = w_base * silu(x) + w_spline . B(x)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
        scale_base: float = 1.0,
        scale_noise: float = 0.1,
        grid_eps: float = 0.02,
        clamp_inputs: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_eps = grid_eps
        self.clamp_inputs = clamp_inputs

        a, b = grid_range
        h = (b - a) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + a
        )
        # [in_features, grid_size + 2*spline_order + 1]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        self.base_weight.data.mul_(scale_base)
        with torch.no_grad():
            self.spline_weight.normal_(0.0, scale_noise / (grid_size + spline_order))

    # ------------------------------------------------------------------ basis
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in] -> [B, in, grid_size + spline_order] (Cox-de Boor)."""
        grid = self.grid  # [in, M]
        x = x.unsqueeze(-1)  # [B, in, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def inner_range(self):
        """The knot interval the spline basis actually partitions.

        ``grid`` carries ``spline_order`` padding knots on each side, and the
        B-spline bases only form a partition of unity between the padding.
        After ``update_grid`` this interval IS the training range of x.
        """
        k = self.spline_order
        return self.grid[:, k], self.grid[:, -(k + 1)]

    def edge_activations(self, x: torch.Tensor) -> torch.Tensor:
        """phi_{o,i}(x_i) for every edge. -> [B, out, in]

        With ``clamp_inputs`` (default), x is first clamped to the knot range,
        so every edge function is extended as a CONSTANT outside the range seen
        in training.

        This is a bug fix, not a tuning knob.  Un-clamped, the spline basis is
        identically zero beyond the knots, so out-of-range inputs are handled by
        the SiLU base term ALONE -- an unbounded, essentially unconstrained
        linear ramp fitted on data that never reached there.  That is what
        produced valung's `log(1-h) = -12,561` (section 4.8) and, on flchain,
        a single test row with per-unit NLL 59.8 that carried 99.7% of a
        "significant" mean difference over 2,362 rows.  A model whose predictions
        are unbounded off the training support cannot be scored by mean NLL.

        Clamping costs nothing inside the training range, where the clamp is
        the identity, so it changes fitted values only where the old behaviour
        was indefensible.  Pass ``clamp_inputs=False`` to reproduce the old
        results.
        """
        if self.clamp_inputs:
            lo, hi = self.inner_range()
            x = torch.clamp(x, min=lo, max=hi)
        bases = self.b_splines(x)  # [B, in, C]
        spline = torch.einsum("bic,oic->boi", bases, self.spline_weight)
        base = F.silu(x).unsqueeze(1) * self.base_weight.unsqueeze(0)
        return base + spline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_activations(x).sum(dim=-1)

    # ------------------------------------------------------- regularization
    def regularization(self, x: torch.Tensor, entropy_weight: float = 1.0):
        """Activation-magnitude L1 + edge entropy (KAN-style sparsification)."""
        act = self.edge_activations(x).abs().mean(dim=0)  # [out, in]
        total = act.sum()
        p = act / (total + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum()
        return total + entropy_weight * entropy

    def edge_importance(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.edge_activations(x).abs().mean(dim=0)

    # ------------------------------------------------------------ grid update
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """Refit the knot vector to the empirical distribution of x, then
        least-squares refit the coefficients so the function is preserved."""
        batch = x.size(0)
        if batch < self.grid_size + self.spline_order + 1:
            return
        bases = self.b_splines(x)                              # [B, in, C]
        y = torch.einsum("bic,oic->bio", bases, self.spline_weight)  # [B, in, out]

        x_sorted = torch.sort(x, dim=0).values
        idx = torch.linspace(0, batch - 1, self.grid_size + 1, device=x.device).long()
        grid_adaptive = x_sorted[idx]                          # [G+1, in]

        lo, hi = x_sorted[0] - margin, x_sorted[-1] + margin
        step = (hi - lo) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=x.device, dtype=x.dtype).unsqueeze(-1)
            * step
            + lo
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        step = (grid[-1] - grid[0]) / self.grid_size
        pre = grid[:1] - step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(-1)
        post = grid[-1:] + step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(-1)
        self.grid.copy_(torch.cat([pre, grid, post], dim=0).T)

        A = self.b_splines(x).transpose(0, 1)                  # [in, B, C]
        B = y.transpose(0, 1)                                  # [in, B, out]
        coef = torch.linalg.lstsq(A, B).solution               # [in, C, out]
        self.spline_weight.copy_(coef.permute(2, 0, 1))


class KAN(nn.Module):
    """Stack of KANLayers."""

    def __init__(self, widths: list[int], **layer_kwargs):
        super().__init__()
        self.widths = list(widths)
        self.layers = nn.ModuleList(
            KANLayer(widths[i], widths[i + 1], **layer_kwargs)
            for i in range(len(widths) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization(self, x: torch.Tensor, entropy_weight: float = 1.0):
        reg = x.new_zeros(())
        for layer in self.layers:
            reg = reg + layer.regularization(x, entropy_weight)
            x = layer(x)
        return reg

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor):
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)

    @torch.no_grad()
    def input_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate |activation| of the first layer, per input feature."""
        return self.layers[0].edge_importance(x).sum(dim=0)
