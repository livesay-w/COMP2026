"""The contour "conjecture": stress-test the residue theorem numerically.

    LHS = oint_C f(z) dz                       (walk the path in finite steps: Euler / midpoint)
    RHS = 2 pi i sum_k n(C, p_k) Res_{p_k} f   (algebra: residues + winding numbers)

The two sides deliberately share no code. The LHS integrates f along the path. The
RHS never integrates anything: it locates poles by root-finding and extracts residues
by *differentiation*, which is why a residue taken from a small circle around each
pole is not allowed here -- that would be the LHS machinery again, and the comparison
would degenerate into contour deformation rather than a test of the theorem.

f is typed as text and handled two ways depending on what you type:

  rational   (1 + 2*z**2) / ((z - 0.5 - 0.5j)*(z + 0.3)**2)
             read symbolically with z a polynomial, so poles and residues are exact
             algebra: root-finding on the denominator factors, then a power-series
             division for the Laurent coefficient.

  general    1/sin(z),  tan(z),  1/(z**2*sin(z)),  1/(exp(z) - 1)
             no symbolic route, so poles are found by Newton on 1/f from a grid of
             starts, their orders come from the log-log slope of |1/f|, and the
             residues come from a finite-difference stencil. Only meromorphic
             building blocks are in the vocabulary, so a branch cut cannot be typed.

A contour is a list of pieces z(t), t in [0, 1]; consecutive pieces join end to end
and the last one returns to the start. No derivative z'(t) is needed: the LHS uses
the step dz = z(t + dt) - z(t) directly.

Run it: consistency checks run first, then a GUI opens. Type f(z) in the box, drag
the mouse to draw a closed contour, press Stop to quit.
"""
from functools import reduce
from math import factorial


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox
from numpy.polynomial import Polynomial
from numpy.polynomial import polynomial as P


# ---------------------------------------------------------------- problem input
def get_problem():
    """Function and contour the GUI opens with."""
    return dict(
        expr="(1 + 2*z**2) / ((z - 0.5 - 0.5j)*(z + 0.3)**2*(z - 2 + 1j))",
        contour=circle_contour(0.0, 1.0),           # last pole lies outside
    )


# ---------------------------------------------------------------- reading typed f(z)
# The vocabulary the text may use. Every entry is meromorphic on the whole plane, so
# no combination of them has a branch cut -- sqrt and log are deliberately absent,
# because the residue theorem does not apply across a cut and the tool would quietly
# be answering a different question.
NAMES = {"__builtins__": {}, "pi": np.pi,
         "sin": np.sin, "cos": np.cos, "tan": np.tan,
         "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp}
SYMBOLS = {"z", "i", "pi"}                          # names the symbolic path can handle


def compile_expr(text):
    """Allow ^ for powers and i for the imaginary unit (Python also accepts 2j etc.)."""
    return compile(text.replace("^", "**"), "<f(z)>", "eval")


def numeric_function(code):
    """LHS side: f as a vectorized callable, straight from the typed text."""
    return lambda z: eval(code, NAMES, {"z": z, "i": 1j})


def is_rational_expr(code):
    """True if the text builds f out of z and constants with + - * / ** alone.

    Those are exactly the expressions the symbolic path below can evaluate, so this
    is the switch between exact algebra and the numerical route.
    """
    return not set(code.co_names) - SYMBOLS


def prod(polys):
    return reduce(lambda a, b: a * b, polys, Polynomial([1]))


class Rational:
    """prod_j N_j(z) / prod_k D_k(z), numerator and denominator kept as factor lists.

    Keeping factors means poles come from root-finding on small factors (exact for
    linear ones) instead of on one big expanded denominator: 1/(z-0.5)**3 stays three
    linear factors rather than a cubic whose triple root np.roots would smear out.
    """

    def __init__(self, nums, dens=()):
        self.nums, self.dens = list(nums), list(dens)

    @staticmethod
    def of(x):
        return x if isinstance(x, Rational) else Rational([Polynomial([x])])

    def __add__(self, other):
        o = Rational.of(other)
        num = prod(self.nums) * prod(o.dens) + prod(o.nums) * prod(self.dens)
        return Rational([num], self.dens + o.dens)

    def __mul__(self, other):
        o = Rational.of(other)
        return Rational(self.nums + o.nums, self.dens + o.dens)

    def inverse(self):
        assert all(np.any(n.coef != 0) for n in self.nums), "division by zero"
        return Rational(self.dens, self.nums)

    def __pow__(self, n):
        assert n == int(n), "only integer powers of rational functions"
        n = int(n)
        if n < 0:
            return self.inverse() ** -n
        return Rational(self.nums * n, self.dens * n)

    __radd__, __rmul__ = __add__, __mul__
    __neg__ = lambda self: self * -1
    __sub__ = lambda self, o: self + (-Rational.of(o))
    __rsub__ = lambda self, o: Rational.of(o) + (-self)
    __truediv__ = lambda self, o: self * Rational.of(o).inverse()
    __rtruediv__ = lambda self, o: Rational.of(o) * self.inverse()

    def parts(self, tol=1e-4):
        """(numerator coefficients, {pole: order}) with f = N / prod (z - p)^m.

        Roots closer than tol are merged into one higher-order pole: np.roots splits
        an m-fold root of an expanded factor like z^2 + 2z + 1 by ~eps^(1/m), and
        averaging the cluster recovers it. So poles closer than 1e-4 can't be told
        apart. Overstating an order is harmless: the Laurent coefficient residue()
        extracts is unchanged.
        """
        num, clusters = prod(self.nums), []         # clusters: [sum of roots, count]
        for d in self.dens:
            d = d.trim()
            num = num / d.coef[-1]                  # make each factor monic
            for r in d.roots():
                for c in clusters:
                    if abs(c[0] / c[1] - r) < tol * (1 + abs(r)):
                        c[0], c[1] = c[0] + r, c[1] + 1
                        break
                else:
                    clusters.append([r, 1])
        return num.coef, {complex(s / n): n for s, n in clusters}


def rational_parts(code):
    """RHS side: evaluate the typed text with z a polynomial symbol."""
    z = Rational([Polynomial([0, 1])])
    return Rational.of(eval(code, NAMES, {"z": z, "i": 1j})).parts()


# ---------------------------------------------------------------- contours
def circle_contour(center, radius, turns=1):
    """Circle traversed `turns` times (negative = clockwise)."""
    return [lambda t: center + radius * np.exp(2j * np.pi * turns * t)]


def polyline_contour(vertices):
    """Closed polygon through `vertices`, closing back to vertices[0]."""
    v = np.asarray(vertices, dtype=complex)
    return [lambda t, a=a, b=b: a + (b - a) * t for a, b in zip(v, np.roll(v, -1))]


def sample_contour(contour, n_per_piece=200):
    t = np.linspace(0.0, 1.0, n_per_piece)
    return np.concatenate([z(t) for z in contour])


def steps_per_piece(contour, n_total):
    """Spread n_total steps over the pieces, at least 16 per piece."""
    return max(16, n_total // len(contour))


# ---------------------------------------------------------------- LHS
def contour_integral(f, contour, n_steps=100_000, method="midpoint"):
    """oint_C f(z) dz by walking the path in finite steps, like an Euler solve.

    Treat the integral as an ODE along the path, dI = f(z) dz, and walk it:
        euler:     I <- I + f(z_k) * dz_k                    slope at start of step, O(h)
        midpoint:  I <- I + f((z_k + z_{k+1})/2) * dz_k      symmetric step,         O(h^2)
    with dz_k = z_{k+1} - z_k (a forward difference of the path, so no z'(t) needed).
    Midpoint is to Euler what the central difference is to the forward one: the
    symmetry cancels the leading error term. n_steps is spread over the pieces.
    """
    assert method in ("euler", "midpoint"), method
    t = np.linspace(0.0, 1.0, steps_per_piece(contour, n_steps) + 1)
    I = 0j
    for piece in contour:
        z = piece(t)
        dz = np.diff(z)
        zk = z[:-1] if method == "euler" else 0.5 * (z[:-1] + z[1:])
        I += np.sum(f(zk) * dz)
    return I


# ---------------------------------------------------------------- RHS
def taylor_coeffs(poly, p, n):
    """First n Taylor coefficients of polynomial `poly` about z = p: poly^(k)(p) / k!."""
    return np.array([P.polyval(p, P.polyder(poly, k)) / factorial(k) for k in range(n)])


def residue(numerator, poles, p):
    """Res_p f for a pole of order m at p, by exact algebra.

    g(z) = (z - p)^m f(z) = N(z) / D(z) with D = prod_{q != p} (z - q)^{m_q} is
    analytic at p, and Res_p f = g^(m-1)(p) / (m-1)! = [Taylor coeff m-1 of g at p].
    We get it by power-series division of N's and D's Taylor coefficients at p.
    """
    m = poles[p]
    D = P.polyfromroots([q for q, mq in poles.items() if q != p for _ in range(mq)])
    n, d = taylor_coeffs(numerator, p, m), taylor_coeffs(D, p, m)
    g = np.zeros(m, dtype=complex)
    for k in range(m):                              # N = D * g, solved order by order
        g[k] = (n[k] - np.dot(d[1:k + 1], g[k - 1::-1] if k else [])) / d[0]
    return g[m - 1]


# ------------------------------------- RHS by differentiation, for any meromorphic f
def reciprocal(f, z):
    """1/f(z), taking the value at a pole of f to be its limit, 0.

    Needed because numpy returns nan, not inf, for complex division by zero. An
    iterate that lands exactly on a pole of f therefore makes f nan, and 1/nan is
    nan, so the root would be thrown away as a failed start -- losing precisely the
    points being hunted, and systematically so, since a simple root is where Newton
    is most likely to land exactly. 1/f really does tend to 0 at a pole, so this is
    the limit rather than a patch. A nan from any other cause is caught later, when
    pole_order() refuses to confirm the candidate.
    """
    v = np.asarray(f(z), dtype=complex)
    at_pole = ~np.isfinite(v)
    return np.where(at_pole, 0.0, 1.0 / np.where(at_pole, 1.0, v))


def newton_poles(f, lim, n_grid=15, n_iter=150, h=1e-5, merge=1e-3):
    """Candidate poles of f inside the box of half-width `lim`, as bare locations.

    A pole of f is a zero of g = 1/f, so Newton on g finds them, with g' from a
    central difference -- the same two-sided formula as the manual notebook, used
    here on a complex argument. Starting points are a grid over the box.

    Starts that wander off or land on a zero of f (where g has a pole) generate
    inf/nan; np.errstate keeps that expected arithmetic from printing warnings, and
    those iterates are dropped. Nothing is hidden by it: every survivor is checked
    against |f| in pole_order() before it is believed.

    Newton converges only linearly onto a zero of multiplicity m, and worse, g is
    computed to a relative accuracy eps, so p itself cannot be pinned down closer
    than about eps**(1/m). That floor -- the same cancellation story as the notebook
    -- is what limits high-order poles here, not the iteration count.
    """
    g = lambda z: reciprocal(f, z)
    s = np.linspace(-lim, lim, n_grid)
    z = (s[:, None] + 1j * s[None, :]).ravel()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(n_iter):
            z = z - g(z) * 2 * h / (g(z + h) - g(z - h))
            z = np.where(np.isfinite(z) & (np.abs(z) < 3 * lim), z, np.nan)
        z = z[np.isfinite(z) & (np.abs(z.real) <= lim) & (np.abs(z.imag) <= lim)]
    # merge the starts that converged to the same root, averaging each cluster
    roots = []
    for w in z:
        for c in roots:
            if abs(c[0] / c[1] - w) < merge * lim:
                c[0], c[1] = c[0] + w, c[1] + 1
                break
        else:
            roots.append([w, 1])
    return [complex(s / n) for s, n in roots]


def pole_order(f, p, h_max, n_h=10):
    """Order of the pole of f at p, from the log-log slope of |1/f| against |w|.

    g = 1/f has a zero of order m at p, so |g(p + w)| ~ C|w|^m and the slope of
    log|g| against log|w| is m. Fitting a slope on a log-log plot is the same
    measurement the manual notebook uses to read off the order of a difference
    scheme. Returns (m, slope); a slope far from an integer means p is not a pole
    of finite order and the caller should discard it.
    """
    hs = h_max * 0.5 ** np.arange(n_h)
    w = hs * np.exp(0.7j)               # off-axis, so a neighbouring real pole is missed
    g = np.abs(1 / f(p + w))
    slope = np.polyfit(np.log(hs), np.log(g), 1)[0]
    return int(round(slope)), slope


def stencil_residue(f, p, m, h_max, n_h=14):
    """Res_p f by numerical differentiation, with no integration anywhere.

    phi(w) = w^m f(p + w) is analytic at w = 0, and the residue is its Taylor
    coefficient of order m-1. Sample phi at the 2J collinear offsets
    w = +-h, +-2h, ..., +-Jh (w = 0 is excluded, since phi is 0/0 there in floating
    point) and solve the Vandermonde system for the Taylor coefficients -- this is
    the construction that generates finite-difference stencils, done once for a
    whole set of coefficients instead of one.

    The coefficient is recovered as B[m-1]/h**(m-1), so the truncation error falls
    with h while the roundoff rises like eps/h**(m-1): the U-shaped curve of the
    notebook's last problem. Rather than guess where the bottom is, sweep h and take
    the estimate where two consecutive h agree best. Returns (residue, spread), and
    the spread is an honest error bar on the result.
    """
    j = max(3, m)
    u = np.concatenate([np.arange(-j, 0), np.arange(1, j + 1)])
    vander = np.vander(u, 2 * j, increasing=True).astype(float)
    hs = h_max * 0.6 ** np.arange(n_h)
    est = np.array([np.linalg.solve(vander, (h * u) ** m * f(p + h * u))[m - 1]
                    / h ** (m - 1) for h in hs])
    k = int(np.argmin(np.abs(np.diff(est))))        # the plateau between the two errors
    return est[k], abs(est[k + 1] - est[k])


def numerical_parts(f, lim):
    """({pole: order}, {pole: residue}) for f, found and differentiated numerically.

    Candidates that turn out not to be poles of integer order -- a point where |f|
    stays bounded, or an essential singularity, where log|1/f| has no slope to fit --
    are dropped here, which is the only place newton_poles' output is trusted.
    """
    cands = newton_poles(f, lim)
    poles, residues = {}, {}
    for p in cands:
        # phi(w) = w^m f(p+w) is analytic only out to the nearest *other* pole, so
        # every offset used below has to stay well inside that disc
        near = min([2.0] + [abs(p - q) for q in cands if q is not p])
        m, slope = pole_order(f, p, 0.1 * near)
        if m < 1 or abs(slope - m) > 0.2:           # not a pole of finite order
            continue
        poles[p] = m
        residues[p] = stencil_residue(f, p, m, 0.3 * near / max(3, m))[0]
    return poles, residues


# ------------------------------------- RHS by Laurent series, for essential singularities
# Numerical differentiation cannot reach an essential singularity, and no amount of
# tuning changes that: the stencil assumes w^m f(p+w) is analytic for some finite m,
# and at exp(1/z) the principal part never terminates. Worse, |exp(1/w)| on a circle
# of radius 0.05 runs from 5e8 to 2e-9 depending only on direction, so there is no
# step size at which sampling f near the point is conditioned at all.
#
# The way out is to stop sampling f and expand it instead. Substituting z -> p + w as
# a truncated two-sided series and running the arithmetic symbolically gives the
# Laurent coefficients directly, exp(1/w) = sum_n w^-n/n! included. It never
# integrates, so the two sides of the theorem still share nothing.
FUNCS = ("sin", "cos", "tan", "sinh", "cosh", "tanh", "exp")


def method_names():
    """Vocabulary where each function defers to the object's own method.

    Lets one piece of typed text be evaluated three ways -- z a number, a Laurent
    series, or a divisor tracker -- without parsing it three times.
    """
    def make(name):
        fallback = NAMES[name]
        return lambda a: getattr(a, name)() if hasattr(a, name) else fallback(a)
    return {"__builtins__": {}, "pi": np.pi, **{n: make(n) for n in FUNCS}}


SYMBOLIC = method_names()


class Laurent:
    """Truncated two-sided series sum_k c_k w^k over a fixed window k in [lo, hi].

    Two-sided, and a window rather than a lowest term, because an essential
    singularity has no lowest term. Coefficients outside the window are dropped, so
    every answer is only as trustworthy as the window is wide -- which is why
    laurent_parts() recomputes with the window doubled and compares.
    """

    def __init__(self, c, lo):
        self.c, self.lo = np.asarray(c, dtype=complex), lo

    @property
    def hi(self):
        return self.lo + len(self.c) - 1

    @staticmethod
    def seed(p, lo, n):
        """z itself, expanded about p: z = p + w."""
        c = np.zeros(n, dtype=complex)
        c[-lo], c[1 - lo] = p, 1.0
        return Laurent(c, lo)

    def like(self, x):
        """Promote a plain number into this window; pass a series through unchanged."""
        if isinstance(x, Laurent):
            assert (x.lo, len(x.c)) == (self.lo, len(self.c)), "mismatched windows"
            return x
        c = np.zeros(len(self.c), dtype=complex)
        c[-self.lo] = x
        return Laurent(c, self.lo)

    def coeff(self, k):
        return self.c[k - self.lo] if self.lo <= k <= self.hi else 0.0

    def rewindow(self, lo, n):
        """Move to the standard window. Zeros below a series' own lowest term are
        genuine, not an approximation; only the top is truncated."""
        c = np.zeros(n, dtype=complex)
        k0, k1 = max(lo, self.lo), min(lo + n - 1, self.hi)
        if k0 <= k1:
            c[k0 - lo:k1 - lo + 1] = self.c[k0 - self.lo:k1 - self.lo + 1]
        return Laurent(c, lo)

    def __add__(self, o):
        return Laurent(self.c + self.like(o).c, self.lo)

    def __mul__(self, o):
        full = np.convolve(self.c, self.like(o).c)   # exponents 2*lo .. 2*hi
        return Laurent(full[-self.lo:len(self.c) - self.lo], self.lo)

    def inverse(self, tol=1e-12):
        """1/series: shift out the lowest power, then the power-series recurrence."""
        scale = np.abs(self.c).max()
        assert scale > 0, "division by an identically zero series"
        nz = np.flatnonzero(np.abs(self.c) > tol * scale)
        d, n = self.lo + nz[0], len(self.c)
        b, tail = np.zeros(n, dtype=complex), self.c[nz[0]:]
        b[:min(n, len(tail))] = tail[:n]
        r = np.zeros(n, dtype=complex)
        r[0] = 1 / b[0]
        for k in range(1, n):                        # b * r = 1, order by order
            r[k] = -np.dot(b[1:k + 1], r[k - 1::-1]) / b[0]
        return Laurent(r, -d).rewindow(self.lo, n)

    def __pow__(self, k):
        assert k == int(k), "only integer powers of a series"
        k = int(k)
        if k < 0:
            return self.inverse() ** -k
        out = self.like(1.0)
        for _ in range(k):
            out = out * self
        return out

    __radd__, __rmul__ = __add__, __mul__
    __neg__ = lambda self: self * -1.0
    __sub__ = lambda self, o: self + (self.like(o) * -1.0)
    __rsub__ = lambda self, o: self.like(o) + (self * -1.0)
    __truediv__ = lambda self, o: self * self.like(o).inverse()
    __rtruediv__ = lambda self, o: self.like(o) * self.inverse()

    def exp(self):
        """exp of a series, which is where an infinite principal part comes from.

        Split off the principal part P and the regular part R. Both sums terminate
        inside the window on their own: every exponent of P^n is <= -n and every
        exponent of R^n is >= n, so nothing beyond n = -lo and n = hi can land in
        [lo, hi]. exp(P) is what turns 1/w into sum_n w^-n/n!.
        """
        k = np.arange(self.lo, self.hi + 1)
        out = self.like(np.exp(self.coeff(0)))
        for part, terms in ((Laurent(np.where(k < 0, self.c, 0), self.lo), -self.lo),
                            (Laurent(np.where(k > 0, self.c, 0), self.lo), self.hi)):
            if not np.any(part.c):
                continue
            total = term = self.like(1.0)
            for n in range(1, terms + 1):
                term = term * part * (1.0 / n)
                total = total + term
            out = out * total
        return out

    # everything else follows from exp, which keeps one hard case instead of six
    def sin(self):
        return ((self * 1j).exp() - (self * -1j).exp()) * (0.5 / 1j)

    def cos(self):
        return ((self * 1j).exp() + (self * -1j).exp()) * 0.5

    def sinh(self):
        return (self.exp() - (self * -1.0).exp()) * 0.5

    def cosh(self):
        return (self.exp() + (self * -1.0).exp()) * 0.5

    def tan(self):
        return self.sin() / self.cos()

    def tanh(self):
        return self.sinh() / self.cosh()


class Tracked:
    """Evaluates f numerically while recording everything it divides by.

    Every singularity of an expression built from + - * / and exp, sin, cos, sinh,
    cosh (tan and tanh expand into quotients) sits at a zero of one of those
    divisors. Collecting them turns "find the singularities of f" into "find the
    zeros of these few functions", which is far better conditioned than hunting
    zeros of 1/f -- that approach returned 0.0065 instead of 0 for exp(1/z), and ten
    spurious points for sin(1/z).

    Divisors are logged factor by factor: 1/((z-0.5)**3 * sin(z)) logs z-0.5 and
    sin(z) separately rather than their product, so each root stays simple and
    Newton reaches full precision instead of the eps**(1/3) a triple root allows.

    Each divisor is carried twice, as a numeric callable and as a series builder,
    because finding its zeros needs the first and sharpening them needs the second.
    """

    def __init__(self, fn, ser, log, factors=None):
        self.fn, self.ser, self.log = fn, ser, log
        self.factors = [(fn, ser)] if factors is None else factors

    def of(self, x):
        if isinstance(x, Tracked):
            return x
        return Tracked(lambda z: x + 0 * z, lambda s: s.like(x), self.log)

    def __add__(self, o):
        o = self.of(o)
        return Tracked(lambda z: self.fn(z) + o.fn(z),
                       lambda s: self.ser(s) + o.ser(s), self.log)

    def __mul__(self, o):
        o = self.of(o)
        return Tracked(lambda z: self.fn(z) * o.fn(z),
                       lambda s: self.ser(s) * o.ser(s), self.log,
                       self.factors + o.factors)

    def divide_by(self, o):
        self.log.extend(o.factors)
        return Tracked(lambda z: self.fn(z) / o.fn(z),
                       lambda s: self.ser(s) / o.ser(s), self.log)

    def __pow__(self, k):
        k = int(k)
        fn, ser = lambda z: self.fn(z) ** k, lambda s: self.ser(s) ** k
        if k < 0:
            self.log.extend(self.factors)
            return Tracked(fn, ser, self.log)
        return Tracked(fn, ser, self.log, self.factors * k)

    __radd__, __rmul__ = __add__, __mul__
    __neg__ = lambda self: self * -1.0
    __sub__ = lambda self, o: self + (self.of(o) * -1.0)
    __rsub__ = lambda self, o: self.of(o) + (self * -1.0)
    __truediv__ = lambda self, o: self.divide_by(self.of(o))
    __rtruediv__ = lambda self, o: self.of(o).divide_by(self)

    def apply(self, name):
        np_fn = NAMES[name]
        return Tracked(lambda z: np_fn(self.fn(z)),
                       lambda s: getattr(self.ser(s), name)(), self.log)

    exp = lambda self: self.apply("exp")
    sin = lambda self: self.apply("sin")
    cos = lambda self: self.apply("cos")
    sinh = lambda self: self.apply("sinh")
    cosh = lambda self: self.apply("cosh")
    tan = lambda self: self.sin() / self.cos()
    tanh = lambda self: self.sinh() / self.cosh()


def newton_zeros(g, lim, n_grid=15, n_iter=120, h=1e-6, merge=1e-7):
    """Zeros of g inside the box, by Newton from a grid with a central-difference g'.

    Unlike newton_poles this hunts zeros of a function that is well behaved there,
    so the iteration is ordinary Newton and the survivors are verified against the
    typical size of |g| rather than trusted.
    """
    s = np.linspace(-lim, lim, n_grid)
    z = (s[:, None] + 1j * s[None, :]).ravel()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.median(np.abs(g(z))[np.isfinite(g(z))])
        for _ in range(n_iter):
            z = z - g(z) * 2 * h / (g(z + h) - g(z - h))
            z = np.where(np.isfinite(z) & (np.abs(z) < 3 * lim), z, np.nan)
        keep = np.isfinite(z) & (np.abs(z.real) <= lim) & (np.abs(z.imag) <= lim)
        z = z[keep]
        z = z[np.abs(g(z)) < 1e-8 * max(scale, 1e-30)]
    roots = []
    for w in z:
        if not any(abs(r - w) < merge * max(lim, 1.0) for r in roots):
            roots.append(w)
    return roots


def refine_root(ser, p, k=8, iters=10, tol=1e-15):
    """Sharpen a root of a divisor using its own Taylor coefficients about p.

    Newton on g stalls at a multiple root, and evaluating g there destroys it:
    sin(z) - z has a triple zero at 0, and near 0 the subtraction cancels away all
    but a digit or two, so the grid search only reaches about 1e-7. Expanding about
    p instead avoids that entirely -- the coefficient g_2 = -sin(p)/2 is computed
    with no cancellation whatever -- and for a root of multiplicity m sitting a
    distance d away, g_{m-1}/g_m = m*d exactly. So read m off the coefficients (the
    first one that is not suppressed by a power of d) and step by that.
    """
    for _ in range(iters):
        c = ser(Laurent.seed(p, -k, 2 * k + 1))
        scale = np.abs(c.c).max()
        if scale == 0 or np.any(np.abs(c.c[:k]) > 1e-10 * scale):
            return p                            # a pole, not a plain zero: leave it
        g = c.c[k:]                             # Taylor coefficients g_0, g_1, ...
        m = next((j for j in range(1, len(g)) if abs(g[j]) >= 0.1 * scale), 1)
        step = g[m - 1] / (m * g[m])
        p = p - step
        if abs(step) <= tol * max(1.0, abs(p)):
            break
    return p


def find_singularities(code, lim):
    """Every point in the box where the typed f can blow up: zeros of its divisors."""
    log = []
    eval(code, SYMBOLIC, {"z": Tracked(lambda w: w, lambda s: s, log), "i": 1j})
    found = []
    for fn, ser in log:
        for p in newton_zeros(fn, lim):
            p = refine_root(ser, p)
            if not any(abs(q - p) < 1e-7 * max(lim, 1.0) for q in found):
                found.append(p)
    return found


def laurent_series(code, p, k):
    """The typed f re-evaluated with z a series about p, over exponents [-k, k]."""
    seed = Laurent.seed(p, -k, 2 * k + 1)
    out = eval(code, SYMBOLIC, {"z": seed, "i": 1j})
    return out if isinstance(out, Laurent) else seed.like(out)


def laurent_parts(code, p, radius, k=20, tol=1e-9):
    """(order, residue, note) at p from the Laurent expansion about it.

    order is None at an essential singularity, where the principal part runs off the
    bottom of the window, and 0 where the singularity turns out to be removable.
    Two self-checks, both of which would fail loudly if the expansion were wrong:
    doubling the window must not move the residue, and the truncated series must
    reproduce f itself out at |w| = radius. The second is the one that catches an
    off-centre expansion -- a series about a point a hair away from the true pole is
    a *Taylor* series with radius |error|, so it disagrees violently further out.
    """
    a, b = laurent_series(code, p, k), laurent_series(code, p, 2 * k)
    res = b.coeff(-1)
    if abs(a.coeff(-1) - res) > tol * max(1.0, abs(res)):
        return None, res, "window too narrow, residue still moving"
    # Order = where the principal part stops. A pole of order m has coefficients
    # exactly zero below -m, so the answer does not depend on where the threshold
    # is put; an essential singularity has a decaying tail instead, and reading it
    # at two tolerances gives two different answers. That disagreement is the test.
    scale = np.abs(b.c).max()
    if scale == 0:
        return 0, 0.0, None
    lows = [b.lo + int(np.flatnonzero(np.abs(b.c) > t * scale)[0]) for t in (1e-9, 1e-14)]
    order = None if lows[0] != lows[1] else max(0, -lows[1])

    w = radius * np.exp(1j * np.array([0.4, 2.5, 4.6]))
    want = numeric_function(code)(p + w)
    got = sum(b.coeff(j) * w ** float(j) for j in range(b.lo, b.hi + 1))
    if not np.allclose(got, want, rtol=1e-6, atol=1e-9 * np.abs(want).max()):
        return order, res, "series disagrees with f; the centre is probably off"
    return order, res, None


def unexplained_blowups(f, poles, lim, n=60, ratio=10.0):
    """Points where |f| is bigger than an analytic function is allowed to be.

    The guard against the original bug: an RHS assembled from an incomplete list of
    singularities looks exactly like the theorem failing, so the tool has to notice
    when |f| does something it has not accounted for.

    Divide out the known poles and cut a small disc around each singularity. What
    remains should be analytic, and the maximum modulus principle then fixes where
    its largest value can sit: on the boundary of that region, meaning the edge of
    the box together with those little circles. Any interior point that beats the
    boundary is a singularity nobody accounted for. A plain size threshold does not
    work here -- a simple pole the grid happens to miss only lifts |f| by a factor
    of twenty or so, while a pole of order 8 lifts its whole neighbourhood.

    Poles are divided out rather than masked because an order-8 pole dominates |f|
    far beyond any sensible radius; an essential singularity cannot be tamed by any
    such factor, so only the disc is cut and its own circle carries the bound.
    """
    s = np.linspace(-lim, lim, n)
    z = (s[:, None] + 1j * s[None, :]).ravel()
    radius = 3 * (2 * lim / (n - 1))

    def tamed(w):
        out = np.asarray(f(w), dtype=complex)
        for q, m in poles.items():
            out = out * (w - q) ** (m or 0)
        return np.abs(out)

    inside, rings = np.ones(len(z), dtype=bool), []
    for p in poles:
        inside &= np.abs(z - p) > radius
        rings.append(p + radius * np.exp(2j * np.pi * np.linspace(0, 1, 65)))
    edge = ((np.abs(np.abs(z.real) - lim) < 1e-9)
            | (np.abs(np.abs(z.imag) - lim) < 1e-9)) & inside
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        walls = [tamed(z[edge])] + [tamed(c) for c in rings]
        bound = max((v[np.isfinite(v)].max() for v in walls if np.any(np.isfinite(v))),
                    default=0.0)
        v = tamed(z[inside])
    return list(z[inside][~np.isfinite(v) | (v > ratio * max(bound, 1e-300))])


def laurent_all(code, lim):
    """({p: order}, {p: residue}, notes) for any meromorphic f, essential included."""
    sings = find_singularities(code, lim)
    poles, residues, notes = {}, {}, []
    for p in sings:
        near = min([2.0] + [abs(p - q) for q in sings if q is not p])
        order, res, note = laurent_parts(code, p, 0.3 * near)
        if note:
            notes.append(f"{p:+.4g}: {note}")
        if order == 0:                               # removable, contributes nothing
            continue
        poles[p], residues[p] = order, res
    notes += [f"|f| blows up near {w:+.4g} but no singularity was found there"
              for w in unexplained_blowups(numeric_function(code), poles, lim)]
    return poles, residues, notes


def analyze(code, lim):
    """({pole: order}, {pole: residue}, how, notes) for the typed f.

    `lim` bounds the search box and only matters for the series route: a rational f
    has finitely many poles and the symbolic route returns all of them, while for a
    general f only what lies inside the box is found. `notes` carries anything the
    caller must not paper over -- an unexplained blow-up there means the residue sum
    is incomplete, and showing it as a finished RHS would libel the theorem.
    """
    if is_rational_expr(code):
        numerator, poles = rational_parts(code)
        return poles, {p: residue(numerator, poles, p) for p in poles}, "exact algebra", []
    poles, residues, notes = laurent_all(code, lim)
    return poles, residues, "Laurent series", notes


def dense_samples(contour, n_samples=4000):
    """Points along C, n_samples spread over the pieces."""
    return sample_contour(contour, steps_per_piece(contour, n_samples))


def on_contour(contour, p):
    """True if p is within one sample spacing of C (winding number / integral undefined)."""
    z = dense_samples(contour)
    return np.min(np.abs(z - p)) < np.max(np.abs(np.diff(z)))


def winding_number(contour, p):
    """n(C, p): total change in arg(z - p) along C, divided by 2 pi."""
    z = dense_samples(contour)
    phase = np.unwrap(np.angle(z - p))
    # unwrap is only trustworthy if no step is near pi; fails if p is (almost) on C
    assert np.max(np.abs(np.diff(phase))) < np.pi / 2, f"pole {p} too close to contour"
    return int(round((phase[-1] - phase[0]) / (2 * np.pi)))


def residue_sum(residues, contour):
    """RHS: 2 pi i sum_k n(C, p_k) Res_{p_k} f, for {pole: residue} from either route."""
    return 2j * np.pi * sum(winding_number(contour, p) * r for p, r in residues.items())


# ---------------------------------------------------------------- consistency checks
def check_lhs():
    """Known integrals, Cauchy's theorem, and the step-halving test for the order."""
    circle = circle_contour(0.0, 1.0)
    square = polyline_contour([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])   # CCW around 0
    # oint z^n dz = 2 pi i if n = -1, else 0;  oint exp(z) dz = 0 (entire)
    for C in (circle, square):
        for n in range(-3, 3):
            exact = 2j * np.pi if n == -1 else 0.0
            assert abs(contour_integral(lambda z: z ** n, C) - exact) < 1e-8, n
        assert abs(contour_integral(np.exp, C)) < 1e-8
    # halving test: halve the step, the error must drop by 2^order
    f, exact = lambda z: 1 / (z - 0.3), 2j * np.pi
    for C, name in ((circle, "circle"), (square, "square")):
        for method, order in (("euler", 1), ("midpoint", 2)):
            e1, e2 = (abs(contour_integral(f, C, n, method) - exact) for n in (400, 800))
            print(f"  {name} {method:8s} halving ratio = {e1 / e2:.3f}  (expect {2 ** order})")
            assert abs(np.log2(e1 / e2) - order) < 0.1, (name, method)


def check_residues():
    """Hand-derived residues, simple-pole formula N(p)/Q'(p), and sum of residues = 0."""
    # f = 1 / (z^2 (z - 1)):  Res_0 = d/dz[1/(z-1)] at 0 = -1,  Res_1 = 1/1^2 = 1
    poles = {0.0: 2, 1.0: 1}
    assert abs(residue([1.0], poles, 0.0) + 1) < 1e-14
    assert abs(residue([1.0], poles, 1.0) - 1) < 1e-14
    # simple poles: Res_p = N(p) / Q'(p), a formula that shares nothing with residue()
    N, poles = [2.0, -1.0, 3.0], {1j: 1, -1j: 1, 0.5: 1, -2.0 + 0.3j: 1}
    Qp = P.polyder(P.polyfromroots(list(poles)))
    for p in poles:
        assert abs(residue(N, poles, p) - P.polyval(p, N) / P.polyval(p, Qp)) < 1e-12
    # deg N <= deg Q - 2  =>  no residue at infinity  =>  finite residues sum to 0
    N, poles = [1.0, 0.0, 2.0], {0.5 + 0.5j: 1, -0.3: 2, 2.0 - 1.0j: 3}
    assert abs(sum(residue(N, poles, p) for p in poles)) < 1e-12


def check_winding():
    """Winding numbers of circles (both orientations, multiple turns) and a square."""
    assert winding_number(circle_contour(0.0, 1.0), 0.2j) == 1
    assert winding_number(circle_contour(0.0, 1.0, turns=-1), 0.2j) == -1
    assert winding_number(circle_contour(0.0, 1.0, turns=2), 0.2j) == 2
    assert winding_number(circle_contour(0.0, 1.0), 3.0) == 0
    assert winding_number(polyline_contour([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]), 0.9) == 1


def check_parser():
    """Typed text -> the right poles/residues, and both readings describe the same f."""
    # 1/(2z - 1): leading coefficient matters, Res_{1/2} = 1/2
    N, poles = rational_parts(compile_expr("1/(2*z - 1)"))
    assert list(poles) == [0.5] and abs(residue(N, poles, 0.5) - 0.5) < 1e-14
    # 1/(z^2 + 2z + 1) = 1/(z+1)^2: np.roots splits the double root; must merge back
    N, poles = rational_parts(compile_expr("1/(z^2 + 2*z + 1)"))
    (p, m), = poles.items()
    assert m == 2 and abs(p + 1) < 1e-12 and abs(residue(N, poles, p)) < 1e-6
    # triple pole, factored and expanded: 1/(z - 1/2)^3 has residue 0 (only a z^-3 term)
    for text in ("1/(z-0.5)**3", "1/(z^3 - 1.5*z^2 + 0.75*z - 0.125)"):
        N, poles = rational_parts(compile_expr(text))
        (p, m), = poles.items()
        assert m == 3 and abs(p - 0.5) < 1e-12 and abs(residue(N, poles, p)) < 1e-6, text
    # symbolic reading, evaluated as N / prod (z-p)^m, matches the numeric reading
    code = compile_expr(get_problem()["expr"] + " + 3/z - i*z^-2")
    N, poles = rational_parts(code)
    z = np.random.default_rng(0).normal(size=20) + 1j * np.random.default_rng(1).normal(size=20)
    sym = P.polyval(z, N) / np.prod([(z - p) ** m for p, m in poles.items()], axis=0)
    assert np.allclose(sym, numeric_function(code)(z), rtol=1e-10)


def check_numerical_rhs():
    """The numerical route reproduces the exact algebra, and hand-derived residues.

    The first block is the strong one: for a rational f both routes are available
    and they share nothing, so agreement pins down the pole finder, the order fit
    and the stencil all at once. The rest are non-rational f whose residues can be
    worked out by hand from a series, which is what the numerical route is for.
    """
    code = compile_expr(get_problem()["expr"])
    numerator, exact = rational_parts(code)
    found, residues = numerical_parts(numeric_function(code), 3.0)
    assert len(found) == len(exact), (len(found), len(exact))
    for p, m in exact.items():
        q = min(found, key=lambda w: abs(w - p))
        assert abs(q - p) < 1e-6 and found[q] == m, (p, q, m, found[q])
        assert abs(residues[q] - residue(numerator, exact, p)) < 1e-7, p

    # non-rational f, residues by hand:
    #   1/sin z   simple poles at n pi,      Res = 1/cos(n pi) = (-1)^n
    #   tan z     simple poles at pi/2 + n pi, Res = sin/(-sin) = -1
    #   1/(e^z-1) simple poles at 2 pi i n,   Res = 1/e^p = 1
    #   cot z     simple pole at 0,           Res = 1
    #   1/sin^2 z double poles at n pi,       even about the pole, so Res = 0
    #   1/(z^2 sin z) = z^-3 (1 - z^2/6 + ...)^-1 = z^-3 + z^-1/6 + ...,  Res = 1/6
    cases = [("1/sin(z)", 4.0, {0: 1.0, np.pi: -1.0, -np.pi: -1.0}, 1),
             ("tan(z)", 4.0, {np.pi / 2: -1.0, -np.pi / 2: -1.0}, 1),
             ("1/(exp(z)-1)", 3.0, {0: 1.0}, 1),
             ("cos(z)/sin(z)", 2.0, {0: 1.0}, 1),
             ("1/sin(z)**2", 2.0, {0: 0.0}, 2),
             ("1/(z**2*sin(z))", 2.0, {0: 1 / 6}, 3)]
    for text, lim, want, order in cases:
        code = compile_expr(text)
        assert not is_rational_expr(code), text            # must take the numeric route
        poles, residues = numerical_parts(numeric_function(code), lim)
        assert len(poles) == len(want), (text, sorted(poles), sorted(want))
        for p, r in want.items():
            q = min(poles, key=lambda w: abs(w - p))
            assert abs(q - p) < 1e-6, (text, p, q)
            assert poles[q] == order, (text, p, poles[q], order)
            assert abs(residues[q] - r) < 1e-6, (text, p, residues[q], r)
    print("  numerical route agrees with exact algebra and with hand-derived residues")


def check_laurent():
    """The series route: residues that no finite-order method can reach.

    Hand-derived from the Laurent expansions:
      exp(1/z)      = sum_n z^-n / n!                      Res = 1     (essential)
      z^2 exp(1/z)  = sum_n z^(2-n) / n!                   Res = 1/6   (essential)
      sin(1/z)      = sum (-1)^k z^-(2k+1) / (2k+1)!       Res = 1     (essential)
      exp(z)/z^n    = sum_k z^(k-n) / k!                   Res = 1/(n-1)!
      1/(z^2 sin z) = z^-3 (1 - z^2/6 + ...)^-1            Res = 1/6
      1/(sin z - z), with sin z - z = -(z^3/6)(1 - z^2/20 + ...)   Res = -3/10
    The last one is the sharp test of the root refinement: sin(z) - z cancels away
    almost every digit near 0, so Newton on it alone stops around 1e-7, and an
    expansion centred 1e-7 off a triple pole returns residue 0 rather than -0.3.
    """
    cases = [("exp(1/z)", None, 1.0), ("z**2*exp(1/z)", None, 1 / 6),
             ("sin(1/z)", None, 1.0), ("exp(z)/z**3", 3, 0.5),
             ("exp(z)/z**8", 8, 1 / factorial(7)),
             ("exp(z)/z**10", 10, 1 / factorial(9)),
             ("1/(z**2*sin(z))", 3, 1 / 6), ("1/(sin(z)-z)", 3, -0.3),
             ("1/sin(z)", 1, 1.0), ("1/(exp(z)-1)", 1, 1.0)]
    for text, order, want in cases:
        code = compile_expr(text)
        poles, residues, notes = laurent_all(code, 2.0)
        assert not notes, (text, notes)
        p = min(poles, key=abs)
        assert abs(p) < 1e-12, (text, p)                  # the singularity is at 0
        assert poles[p] == order, (text, poles[p], order)
        assert abs(residues[p] - want) < 1e-12, (text, residues[p], want)
    # a removable singularity contributes nothing and must not be listed as a pole
    assert not laurent_all(compile_expr("sin(z)/z"), 2.0)[0]
    # and the series route must reproduce the exact algebra on a rational f
    code = compile_expr("(1 + 2*z**2)/((z - 0.5 - 0.5j)*(z + 0.3)**2)")
    numerator, exact = rational_parts(code)
    poles, residues, notes = laurent_all(code, 3.0)
    assert not notes and len(poles) == len(exact)
    for p, m in exact.items():
        q = min(poles, key=lambda w: abs(w - p))
        assert poles[q] == m and abs(residues[q] - residue(numerator, exact, p)) < 1e-10
    print("  Laurent route: essential singularities and hand-derived residues agree")


def check_both_sides():
    """The whole point: LHS and RHS, by two routes that share no machinery.

    1/sin z on |z| = 4 encloses the poles at 0 and +-pi: 2 pi i (1 - 1 - 1) = -2 pi i.
    exp(1/z) on |z| = 1 is the case that used to fail silently, and is the one the
    stencil cannot do at all -- the residue theorem holds at an essential
    singularity, it is only numerical differentiation that cannot see it.
    """
    for text, lim, want in [("1/sin(z)", 4.0, -2j * np.pi),
                            ("exp(1/z)", 1.0, 2j * np.pi),
                            ("sin(1/z)", 1.0, 2j * np.pi)]:
        code = compile_expr(text)
        circle = circle_contour(0.0, lim)
        poles, residues, how, notes = analyze(code, 1.2 * lim)
        lhs, rhs = contour_integral(numeric_function(code), circle), residue_sum(residues, circle)
        print(f"  {text:10s} on |z| = {lim:g} via {how}: {len(poles)} singularities,"
              f" |LHS - RHS| = {abs(lhs - rhs):.2e}")
        assert not notes, (text, notes)
        assert abs(rhs - want) < 1e-9 and abs(lhs - rhs) < 1e-6, (text, lhs, rhs, want)


def check_guard():
    """An incomplete residue sum must never be presented as a finished RHS.

    The original bug: every candidate at exp(1/z) was discarded, the RHS came out
    0, and nothing said so. Starve the finder by handing the sweep an empty list of
    singularities and it must object every time; hand it the right list and it must
    stay silent, including for the high-order and essential cases whose sheer size
    would fool a plain threshold.
    """
    for text in ("exp(1/z)", "1/sin(z)", "1/(z-0.4-0.3j)", "exp(z)/z**8"):
        f, lim = numeric_function(compile_expr(text)), 2.0
        assert unexplained_blowups(f, {}, lim), f"missed singularity not reported: {text}"
    for text in ("exp(1/z)", "1/sin(z)", "exp(z)/z**8", "1/(z**2*sin(z))",
                 "1/(z-0.4-0.3j)", "sin(z)/z", "exp(z)"):
        code = compile_expr(text)
        poles, _, notes = laurent_all(code, 2.0)
        assert not notes, (text, notes)
    # a pole present but left out of the list must still be caught
    code = compile_expr("1/((z-0.4-0.3j)*(z+1.1))")
    poles, _, _ = laurent_all(code, 2.0)
    half = {p: m for p, m in list(poles.items())[:1]}
    assert unexplained_blowups(numeric_function(code), half, 2.0), "half a list passed"
    print("  guard: unaccounted singularities are reported, accounted ones are not")


# ---------------------------------------------------------------- GUI
def fmt(z):
    return f"{z.real:+.12f} {z.imag:+.12f}i"


class ContourGUI:
    """Type f(z), drag the mouse to draw a contour, see LHS vs RHS when it closes."""

    def __init__(self, expr, contour):
        self.fig = plt.figure(figsize=(12, 7.5))
        self.fig.canvas.manager.set_window_title("Residue theorem stress test")
        self.ax = self.fig.add_axes([0.05, 0.18, 0.52, 0.77])
        info = self.fig.add_axes([0.60, 0.18, 0.38, 0.77])
        info.axis("off")
        self.info = info.text(0, 1, "", va="top", family="monospace", fontsize=9)
        self.box = TextBox(self.fig.add_axes([0.10, 0.07, 0.62, 0.05]), "f(z) = ", initial=expr)
        self.stop = Button(self.fig.add_axes([0.85, 0.07, 0.10, 0.05]), "Stop", color="salmon")
        self.status = self.fig.text(0.05, 0.02, "", fontsize=9)

        ax, self.lim = self.ax, 3.0
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.set_title("drag to draw a contour; end where you started to close it")
        self.pole_art, = ax.plot([], [], "rx", ms=10, mew=2, label="pole of f")
        self.path_art, = ax.plot([], [], "b-", lw=1.5, label="contour C")
        self.start_art, = ax.plot([], [], "go", ms=12, mfc="none", mew=2,
                                  label="start (close here)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.labels = []

        self.drawing, self.points, self.contour = False, [], contour
        self.box.on_text_change(self.set_function)   # re-parse on every keystroke
        self.stop.on_clicked(lambda _: plt.close(self.fig))
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        z = sample_contour(contour)
        self.path_art.set_data(z.real, z.imag)
        self.set_function(expr)

    # ---- f(z) from the text box
    def set_function(self, text):
        # Half-typed input ("1/(z-") is expected while typing, so errors here are
        # reported in the status line and the last valid f is kept.
        try:
            code = compile_expr(text)
            f = numeric_function(code)
            f(np.array([0.123 + 0.456j]))            # surface numeric-side errors now
        except Exception as e:
            self.say(f"f(z) not updated -- {type(e).__name__}: {e}", "red")
            return
        self.code, self.f = code, f
        self.find_poles()
        if self.notes:
            self.say(f"f(z) ok, but {len(self.notes)} warning(s) -- see the panel", "red")
            self.update_results()
            return
        self.say(f"f(z) ok, poles by {self.how} -- drag the mouse to draw a closed contour",
                 "black")
        self.update_results()

    def scan_box(self):
        """Half-width of the box the numerical pole search has to cover.

        The view, widened to contain the drawn contour. A pole inside the contour
        but outside the search box is left out of the RHS, and the two sides then
        disagree by its whole contribution -- which reads as the theorem failing
        when in fact the tool just did not look far enough.
        """
        lim = self.lim
        if self.contour is not None:
            z = dense_samples(self.contour, 400)
            lim = max(lim, 1.05 * max(np.abs(z.real).max(), np.abs(z.imag).max()))
        return lim

    def find_poles(self):
        self.scan_lim = self.scan_box()
        self.poles, self.residues, self.how, self.notes = analyze(self.code, self.scan_lim)
        p = np.array(list(self.poles), dtype=complex)
        self.pole_art.set_data(p.real, p.imag)
        if self.how == "exact algebra":
            # the symbolic route returns every pole, so the view can be fitted to them
            self.lim = max([3.0] + [1.2 * max(abs(q.real), abs(q.imag)) for q in p])
            self.ax.set_xlim(-self.lim, self.lim)
            self.ax.set_ylim(-self.lim, self.lim)

    # ---- drawing
    def on_press(self, event):
        toolbar = self.fig.canvas.toolbar
        if event.inaxes is not self.ax or (toolbar is not None and toolbar.mode):
            return                                    # ignore clicks while zoom/pan is on
        self.drawing, self.contour = True, None
        self.points = [complex(event.xdata, event.ydata)]
        self.start_art.set_data([event.xdata], [event.ydata])
        self.update_results()

    def on_move(self, event):
        if not self.drawing or event.inaxes is not self.ax:
            return
        z = complex(event.xdata, event.ydata)
        if abs(z - self.points[-1]) > 0.005 * 2 * self.lim:   # skip jitter-sized steps
            self.points.append(z)
            v = np.array(self.points)
            self.path_art.set_data(v.real, v.imag)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        close_tol = 0.05 * 2 * self.lim
        if len(self.points) < 3 or abs(self.points[-1] - self.points[0]) > close_tol:
            self.say(f"contour NOT closed -- finish within {close_tol:.2f} of the green start circle",
                     "red")
            return
        self.contour = polyline_contour(self.points)
        v = np.array(self.points + self.points[:1])  # show the closing segment
        self.path_art.set_data(v.real, v.imag)
        self.find_poles()                            # the search box must cover the loop
        self.say("contour closed", "black")
        self.update_results()

    # ---- LHS vs RHS
    def update_results(self):
        for t in self.labels:
            t.remove()
        self.labels = []
        lines = [f"singularities by {self.how}:",
                 "  p                              order"]
        lines += [f"  {fmt(p)}   {'essential' if m is None else m}"
                  for p, m in self.poles.items()]
        if self.how != "exact algebra":
            lines += [f"  (searched |Re z|,|Im z| <= {self.scan_lim:.2f}, which covers the",
                      "   contour; anything outside is left out of the RHS)"]
        if self.notes:
            lines += ["", "!! the RHS may be incomplete:"] + [f"   {n}" for n in self.notes]
        on_path = [] if self.contour is None else [
            p for p in self.poles if on_contour(self.contour, p)]
        if self.contour is None:
            lines += ["", "draw a closed contour to compare LHS and RHS"]
        elif on_path:
            lines += ["", "a pole lies ON the contour -- oint f dz is undefined:"]
            lines += [f"  {fmt(p)}" for p in on_path]
        else:
            lhs = contour_integral(self.f, self.contour, method="midpoint")
            lhs_euler = contour_integral(self.f, self.contour, method="euler")
            rhs = residue_sum(self.residues, self.contour)
            lines += ["", "winding n(C,p) and residues:"]
            for p in self.poles:
                n = winding_number(self.contour, p)
                lines.append(f"  n = {n:+d}   Res = {fmt(self.residues[p])}")
                self.labels.append(self.ax.annotate(f"n={n}", (p.real, p.imag), xytext=(6, 6),
                                                    textcoords="offset points", color="darkred"))
            lines += ["",
                      "LHS  oint f dz, midpoint steps  O(h^2):",
                      f"  {fmt(lhs)}",
                      "RHS  2 pi i sum n Res:",
                      f"  {fmt(rhs)}",
                      "",
                      f"|LHS - RHS| = {abs(lhs - rhs):.3e}",
                      "",
                      "for comparison, Euler steps  O(h):",
                      f"  {fmt(lhs_euler)}",
                      f"  |LHS_euler - RHS| = {abs(lhs_euler - rhs):.3e}",
                      f"  ({steps_per_piece(self.contour, 100_000)} steps per piece,"
                      f" {len(self.contour)} pieces)"]
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def say(self, msg, color):
        self.status.set_text(msg)
        self.status.set_color(color)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    check_lhs()
    check_residues()
    check_winding()
    check_parser()
    check_numerical_rhs()
    check_laurent()
    check_guard()
    check_both_sides()
    print("consistency checks passed")

    prob = get_problem()
    gui = ContourGUI(prob["expr"], prob["contour"])
    plt.show()                                       # blocks until Stop closes the window
    print("GUI closed")
