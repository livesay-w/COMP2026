"""
Edwards-Anderson Ising spin glass: lattice, couplings, energy, sampling.

    H = -sum_<ij> J_ij s_i s_j,   s_i = +-1

Natural units: k_B = 1 and the coupling distribution has unit variance
(<J^2> = 1), so T is measured in units of the coupling scale.

Couplings are stored as J[k], an array with the same shape as the spins, where
J[k][x] is the bond between site x and its neighbour at x + e_k.  Every bond
therefore appears exactly once, which makes double-counting impossible by
construction rather than by remembering a factor of 1/2.

Dimension d is a parameter throughout: d=3 is the physics target (T_c ~ 0.95
for Gaussian bonds), d=2 is the fast testbed and the negative control (T_c = 0).
"""

import numpy as np


def random_spins(L, d, rng):
    """Random +-1 configuration on an L^d lattice."""
    return (2 * rng.integers(0, 2, size=(L,) * d) - 1).astype(np.int8)


def make_couplings(L, d, rng, dist="gaussian", boundary="periodic"):
    """Quenched couplings, shape (d, L, ..., L).  Drawn once and held frozen.

    "gaussian" -> N(0, 1), unique ground state, cleanest P(q)
    "pm"       -> +-1, integer energies, allows the gauge-invariance check
    "ferro"    -> all +1, the unfrustrated reference used in step 0

    boundary="open" drops the wrap-around bonds by setting them to zero rather
    than by branching anywhere downstream: a zero coupling contributes nothing
    to energy() or local_fields(), so both keep the single validated code path.
    It also removes the L=2 trap -- under periodic boundaries the +e_k and -e_k
    neighbours of a site coincide at L=2 and bonds get counted twice, but with
    the wrap bond zeroed each line carries exactly one bond, which is what makes
    the 8-spin exact enumeration in step 2 legitimate.
    """
    shape = (d,) + (L,) * d
    if dist == "gaussian":
        J = rng.normal(0.0, 1.0, size=shape)
    elif dist == "pm":
        J = 2.0 * rng.integers(0, 2, size=shape) - 1.0
    elif dist == "ferro":
        J = np.ones(shape)
    else:
        raise ValueError(f"unknown coupling distribution: {dist!r}")

    if boundary == "open":
        for k in range(d):
            idx = [k] + [slice(None)] * d
            idx[1 + k] = L - 1
            J[tuple(idx)] = 0.0
    elif boundary != "periodic":
        raise ValueError(f"unknown boundary condition: {boundary!r}")
    return J


def energy(s, J):
    """Total energy.  Sums each bond exactly once via the +e_k rolls."""
    e = 0.0
    for k in range(J.shape[0]):
        e -= np.sum(J[k] * s * np.roll(s, -1, axis=k))
    return float(e)


def local_fields(s, J):
    """h_i = sum_j J_ij s_j over all 2d neighbours, so flipping i costs
    dE = 2 s_i h_i.  The -e_k term needs the couplings rolled too, since the
    bond behind site x is stored at x - e_k."""
    h = np.zeros(s.shape)
    for k in range(J.shape[0]):
        h += J[k] * np.roll(s, -1, axis=k)                     # +e_k neighbour
        h += np.roll(J[k], 1, axis=k) * np.roll(s, 1, axis=k)   # -e_k neighbour
    return h


def sublattice_parity(L, d):
    """(x + y + z) % 2 for every site.  Under periodic boundaries the cubic
    lattice is bipartite only for even L -- odd L wraps into odd-length cycles
    -- so the checkerboard update below requires even L."""
    assert L % 2 == 0, f"checkerboard needs even L (L={L} is not bipartite under PBC)"
    return (np.indices((L,) * d).sum(axis=0) % 2).astype(np.int8)


def metropolis_sweep(s, J, T, rng, parity):
    """One sweep: a flip proposed at every site, one sublattice at a time.

    No two same-parity sites are neighbours, so an entire sublattice can be
    updated simultaneously and still reproduce single-spin-flip Metropolis
    exactly.  This is what makes L=8 in 3D affordable in numpy.

    Modifies s in place.  Returns (acceptance fraction, total energy change).

    The returned dE is *exact*, not approximate: because no two flipped spins
    are neighbours, the energy change of flipping a whole sublattice at once is
    exactly the sum of the individual 2 s_i h_i terms.  That identity holds
    only if the decomposition really is bipartite, so tracking dE and
    comparing it against a full recomputation (step 1) tests the checkerboard
    itself, not just the arithmetic.
    """
    accepted = 0
    dE_total = 0.0
    for p in (0, 1):
        dE = 2.0 * s * local_fields(s, J)
        # exp(-max(dE,0)/T) equals 1 where dE <= 0 (rng.random() < 1 always),
        # and cannot overflow for dE < 0.
        accept = rng.random(s.shape) < np.exp(-np.maximum(dE, 0.0) / T)
        flip = accept & (parity == p)
        dE_total += float(dE[flip].sum())
        s[flip] *= -1
        accepted += int(flip.sum())
    return accepted / s.size, dE_total


def frustrated_fraction(J):
    """Fraction of elementary square plaquettes whose four bonds have a
    negative product.

    A negative product means no assignment of the four spins can satisfy all
    four bonds at once.  That is frustration -- the defining feature of a spin
    glass, and the one thing step 0's ferromagnet does not have.

    For any symmetric coupling distribution the four bond signs are independent
    and symmetric, so the exact expectation is 1/2, independent of lattice size
    and dimension.  For the ferromagnet it is exactly 0.
    """
    d = J.shape[0]
    assert d >= 2, "plaquettes need at least two dimensions"
    products = []
    for k in range(d):
        for m in range(k + 1, d):
            # The four bonds of the plaquette at x in the (k, m) plane:
            #   x -> x+e_k,  x+e_k -> x+e_k+e_m,  x+e_m -> x+e_m+e_k,  x -> x+e_m
            products.append(J[k]
                            * np.roll(J[m], -1, axis=k)
                            * np.roll(J[k], -1, axis=m)
                            * J[m])
    return float(np.mean(np.concatenate([p.ravel() for p in products]) < 0))
