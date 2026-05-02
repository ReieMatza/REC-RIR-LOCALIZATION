# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: loss functions.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSE_loss_complex(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, y, x):
        return (x - y).abs().pow(2).mean()


class RIMag_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ret = (
            (output.real - target.real).abs()
            + (output.imag - target.imag).abs()
            + (output.abs() - target.abs()).abs()
        )
        return ret.mean()


class GaussianLabelSmoothingLoss(nn.Module):
    """Cross-entropy against a truncated-Gaussian soft target over an ordinal class axis.

    Motivation:
        For ordinal class targets (angle bin, radius bin), a Dirac one-hot target
        penalizes "close but not exact" predictions just as hard as "wildly wrong"
        ones. A Gaussian-shaped target whose width matches the physical resolvability
        of the quantity gives the network a graded signal and stops it from wasting
        capacity trying to spike at a single bin it cannot resolve.

    The class axis is treated as open-ended (NOT circular): the Gaussian is
    truncated at [0, num_classes-1] and renormalized, so samples near the edges
    get half-Gaussian targets that still integrate to 1.

    Args:
        num_classes: total number of ordinal classes, C.
        sigma:       Gaussian standard deviation in CLASS units (not physical units).
                     e.g. for a 1 deg/class angle axis, sigma=3 means ~3 deg width.

    Shape:
        logits:      (N, C) - raw scores from the head.
        class_idx:   (N,)   - integer class labels in [0, C-1]. Pre-filter invalids
                              (class_idx < 0) before calling forward.
    """

    def __init__(self, num_classes: int, sigma: float) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if sigma <= 0:
            raise ValueError(
                f"sigma must be positive; for hard CE use nn.CrossEntropyLoss directly"
            )

        self.num_classes = int(num_classes)
        self.sigma = float(sigma)

        # Precompute a (C, C) soft-target table: row k = truncated Gaussian
        # centered at class k, normalized to sum to 1.
        #   table[k, c] = exp(-((c - k)^2) / (2 * sigma^2)) / Z_k
        centers = torch.arange(self.num_classes, dtype=torch.float32).unsqueeze(1)  # (C, 1)
        positions = torch.arange(self.num_classes, dtype=torch.float32).unsqueeze(0)  # (1, C)
        unnormalized = torch.exp(
            -((positions - centers) ** 2) / (2.0 * self.sigma**2)
        )  # (C, C)
        table = unnormalized / unnormalized.sum(dim=1, keepdim=True)

        # Per-row entropy (nats) of the soft target distribution. Used by the trainer
        # to convert soft-CE back into KL divergence (cross-run comparable, zero floor).
        row_entropy = -(table * torch.log(table.clamp_min(1e-12))).sum(dim=1)  # (C,)

        # Register as buffers so they move with .to(device) / DDP replication.
        self.register_buffer("soft_target_table", table, persistent=False)
        self.register_buffer("soft_target_entropy", row_entropy, persistent=False)

    def soft_target(self, class_idx: torch.Tensor) -> torch.Tensor:
        """Return the smoothed (N, C) target distribution for integer labels."""
        return self.soft_target_table.index_select(0, class_idx.long())

    def target_entropy(self, class_idx: torch.Tensor) -> torch.Tensor:
        """Return the per-sample H(soft_target) in nats, shape (N,)."""
        return self.soft_target_entropy.index_select(0, class_idx.long())

    def forward(
        self, logits: torch.Tensor, class_idx: torch.Tensor
    ) -> torch.Tensor:
        target = self.soft_target(class_idx)
        # F.cross_entropy with soft-label target returns mean over batch by default.
        return F.cross_entropy(logits, target)
