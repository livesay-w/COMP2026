"""Step 7d: embedded-Ising cluster updates (Brower & Tamayo 1989), to beat critical slowing down.

Write phi_x = s_x |phi_x| with s_x = +-1. The potential (m2 and lam terms) is even in phi, and each kinetic link is
    1/2 (phi_x - phi_y)^2 = 1/2 phi_x^2 + 1/2 phi_y^2 - |phi_x| |phi_y| s_x s_y,
so at fixed |phi| the signs are a ferromagnetic Ising model with link couplings J_xy = |phi_x phi_y|:
    p(s | |phi|)  ∝  exp( sum_<xy> J_xy s_x s_y ).
Swendsen-Wang for that Ising model: bond each link whose ends have equal sign (phi_x phi_y > 0) with probability
1 - exp(-2 J_xy) = 1 - exp(-2 phi_x phi_y), find the connected clusters, and flip each cluster's sign with probability 1/2.

This leaves p(phi) = exp(-S)/Z invariant, but it never changes |phi|. So it alternates with the local Metropolis sweep of
phi4.py, which updates the magnitudes. One "update" = 1 Metropolis sweep + 1 SW step. The SW step already flips the sign of
whole clusters, so the Metropolis sweep runs without the global flip.

Lattice units (a = 1), periodic L x L, L even (checkerboard).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from phi4 import checkerboard_np, sweep_once


def bonds(phi, rng):
    """SW bonds for both axes: [bond_0, bond_1], each (L, L) bool, True where the link x -> x + e_axis is bonded.
    p = 1 - exp(-2 phi_x phi_y) on links with phi_x phi_y > 0 and 0 otherwise (clip: J = 0 -> p = 0 -> never bonded)."""
    return [rng.random(phi.shape) < -np.expm1(-2 * np.clip(phi * np.roll(phi, -1, axis=ax), 0, None)) for ax in (0, 1)]


def clusters(bond):
    """Connected components of the bond graph (periodic). Returns (labels (L, L) with values 0 .. n - 1, n)."""
    L = bond[0].shape[0]
    idx = np.arange(L * L).reshape(L, L)
    src = np.concatenate([idx[b] for b in bond])
    dst = np.concatenate([np.roll(idx, -1, axis=ax)[b] for ax, b in enumerate(bond)])
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(L * L, L * L))
    n, labels = connected_components(graph, directed=False)
    return labels.reshape(L, L), n


def sw_step(phi, rng):
    """One Swendsen-Wang step on the signs of phi. Returns (new phi, cluster labels)."""
    labels, n = clusters(bonds(phi, rng))
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return phi * sign[labels], labels


def cluster_chain(L, m2, lam, n_updates, n_therm, step, rng, measure):
    """Markov chain of updates (Metropolis sweep, then SW step), starting from a random field.

    measure(phi) -> array is recorded after every update past thermalisation.
    Returns (records (n_updates, *shape of measure), final configuration (L, L), Metropolis acceptance rate)."""
    assert L % 2 == 0, "checkerboard needs even L with periodic boundaries"
    phi = rng.normal(size=(L, L))
    masks = [checkerboard_np(L, 0), checkerboard_np(L, 1)]
    records = np.empty((n_updates,) + np.shape(measure(phi)))
    n_accept = 0
    for t in range(n_therm + n_updates):
        phi, n_acc = sweep_once(phi, masks, m2, lam, step, rng, flip=False)
        phi, _ = sw_step(phi, rng)
        if t >= n_therm:
            records[t - n_therm] = measure(phi)
            n_accept += n_acc
    return records, phi, n_accept / (n_updates * L * L)


def m_and_phi2(phi):
    """[M, sum_x phi_x^2]. M gives U4 and chi; sum phi^2 = 2 dS/dm2 is kept so the chains can be reweighted in m2 later."""
    return np.array([phi.mean(), np.sum(phi**2)])
