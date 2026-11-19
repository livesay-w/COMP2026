"""Convolutional normalizing flow (RealNVP-style affine couplings, checkerboard masks) for lattice fields.

    z ~ N(0, 1) on every site   --flow-->   phi = f(z)
    log q(phi) = log r(z) - log|det df/dz|

Each coupling layer freezes one checkerboard colour and updates the other:
    phi_active <- exp(s) * phi_active + t,      (s, t) = CNN(phi_frozen)
The CNN only sees frozen sites, so the Jacobian is triangular and log|det| = sum of s over active sites.
The CNN kernels (3x3, circular padding = periodic lattice) are the trainable parameters.
"""
import math

import torch
import torch.nn as nn


def checkerboard(L, parity):
    """1 on sites with (i + j) % 2 == parity (the frozen sites), 0 elsewhere."""
    i, j = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    return ((i + j) % 2 == parity).to(torch.get_default_dtype())


class ConvNet(nn.Module):
    """(B, L, L) frozen field -> (s, t), each (B, L, L).

    Three 3x3 conv layers with tanh in between. Circular padding makes every layer
    translation-equivariant on the periodic lattice. The last layer starts at zero,
    so s = t = 0 and every coupling (hence the whole flow) starts as the identity.
    """

    def __init__(self, hidden, kernel_size=3):
        super().__init__()

        def conv(c_in, c_out):
            return nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2, padding_mode="circular")

        self.net = nn.Sequential(conv(1, hidden), nn.Tanh(), conv(hidden, hidden), nn.Tanh(), conv(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x[:, None])  # add channel axis: (B, 1, L, L) -> (B, 2, L, L)
        return out[:, 0], out[:, 1]


class AffineCoupling(nn.Module):
    def __init__(self, L, parity, hidden):
        super().__init__()
        self.register_buffer("frozen", checkerboard(L, parity))
        self.net = ConvNet(hidden)

    def forward(self, x):
        """x -> y, log|det dy/dx| per sample."""
        s, t = self.net(self.frozen * x)
        active = 1 - self.frozen
        y = self.frozen * x + active * (x * torch.exp(s) + t)
        return y, (active * s).sum(dim=(1, 2))

    def reverse(self, y):
        """y -> x, log|det dx/dy| per sample. Frozen sites are unchanged, so the CNN sees the same input."""
        s, t = self.net(self.frozen * y)
        active = 1 - self.frozen
        x = self.frozen * y + active * (y - t) * torch.exp(-s)
        return x, -(active * s).sum(dim=(1, 2))


def log_prior(z):
    """log r(z) for independent N(0, 1) on every site. z: (B, L, L) -> (B,)."""
    V = z.shape[1] * z.shape[2]
    return -0.5 * (z**2).sum(dim=(1, 2)) - 0.5 * V * math.log(2 * math.pi)


class Flow(nn.Module):
    """Stack of affine couplings with alternating checkerboard parity."""

    def __init__(self, L, n_layers, hidden):
        super().__init__()
        assert L % 2 == 0, "checkerboard needs even L with periodic boundaries"
        assert n_layers % 2 == 0, "use an even number of layers so both colours are updated equally"
        self.L = L
        self.layers = nn.ModuleList(AffineCoupling(L, i % 2, hidden) for i in range(n_layers))

    def forward(self, z):
        """z -> phi, log|det dphi/dz|."""
        logdet = torch.zeros(z.shape[0], dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet

    def reverse(self, phi):
        """phi -> z, log|det dz/dphi|."""
        logdet = torch.zeros(phi.shape[0], dtype=phi.dtype)
        for layer in reversed(self.layers):
            phi, ld = layer.reverse(phi)
            logdet = logdet + ld
        return phi, logdet

    def sample(self, batch_size):
        """Draw phi ~ q. Returns (phi, log q(phi))."""
        z = torch.randn(batch_size, self.L, self.L, dtype=self.layers[0].frozen.dtype)
        phi, logdet = self(z)
        return phi, log_prior(z) - logdet

    def log_prob(self, phi):
        """log q(phi) for given fields, by running the flow backwards."""
        z, logdet = self.reverse(phi)
        return log_prior(z) + logdet
