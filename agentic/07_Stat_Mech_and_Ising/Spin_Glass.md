# Spin Glass: Edwards–Anderson in 3D, then Sherrington–Kirkpatrick

**Session 07 — Statistical Mechanics and the Ising Model (agentic).**
Pairs with `notebooks/07_Stat_Mech_and_Ising.ipynb`.

Units: $k_B = 1$, and the coupling distribution is normalized to unit variance
($\langle J^2 \rangle = 1$), so temperature is measured in units of the coupling
scale. All RNGs seeded (`np.random.default_rng(seed)`) so every number below is
reproducible for grading.

---

## The problem

An Ising model with **random** couplings:

$$H = -\sum_{\langle ij \rangle} J_{ij} s_i s_j , \qquad s_i = \pm 1$$

where the $J_{ij}$ are drawn once from a fixed distribution and then held frozen
(*quenched* disorder). Mixed signs mean some bonds want their spins aligned and
some want them anti-aligned; on any loop with an odd number of antiferromagnetic
bonds, no spin assignment satisfies them all. That is **frustration**, and it
produces an energy landscape with an enormous number of local minima separated by
barriers that grow with system size.

Two models, chosen at setup time — not stages of one simulation:

- **Edwards–Anderson (EA).** An $L^d$ lattice, nearest-neighbor bonds only, each
  with a random $J_{ij}$. Every spin couples to its $2d$ neighbors.
- **Sherrington–Kirkpatrick (SK).** Every spin couples to every other spin, with
  $J_{ij} \sim \mathcal{N}(0, 1/N)$. The $1/N$ keeps the energy per spin finite.
  No lattice, no geometry.

SK is the mean-field, effectively infinite-dimensional limit of EA; EA approaches
mean-field behavior as $d$ increases and is essentially mean-field above $d = 6$.

**Why both.** SK is the one spin glass with an *exact* answer — Parisi solved it
with replica symmetry breaking (2021 Nobel), giving $T_c = 1$ and ground-state
energy per spin $\approx -0.7633$. So SK validates the code against known numbers.
Whether 3D EA actually behaves like Parisi's solution, or like the competing
"droplet" picture in which there is essentially one ground state plus its flip, is
a long-running open question. Simulations of exactly this kind are one of the main
ways people probe it.

**Order parameter.** The magnetization is useless here — it averages to zero even
in the frozen phase, because the frozen pattern is random. The right quantity is
the **overlap** between two independent replicas sharing the same couplings:

$$q = \frac{1}{N} \sum_i s_i^{(1)} s_i^{(2)}$$

Histogramming $q$ over many measurements and many disorder realizations gives
$P(q)$, and its shape is the whole argument:

| System | $P(q)$ below $T_c$ |
|---|---|
| Ferromagnet | two sharp peaks at $\pm q$ — one state and its flip |
| SK | those two peaks **plus a continuous plateau between them** — many distinct states |
| 3D EA | something in between; how to read it is the open debate |

---

## Why 3D, and why `d` is still a parameter

The 2D EA Ising spin glass has $T_c = 0$ — no spin glass phase at any finite
temperature, for Gaussian or $\pm J$ couplings alike. The lower critical dimension
is near $d \approx 2.5$, so 2D sits below it. The finite-temperature transition and
the Parisi-vs-droplet question both live in **3D**, where $T_c \approx 0.95$ for
Gaussian bonds and $T_c \approx 1.12$ for $\pm J$.

So 3D is the physics target. But the lattice is built as an $L^d$ array with a
neighbor-offset list and `d` as an argument, because that buys three things for
about twenty lines of indexing:

1. **2D is the development testbed.** $L=8$ is 64 spins in 2D, 512 in 3D. Debug
   parallel tempering and tune the temperature ladder at 2D speed, then set `d=3`.
2. **2D is a negative control.** The 3D analysis will claim that $\xi_L/L$ curves
   for different $L$ cross at $T_c$. The identical pipeline run in 2D must show
   **no crossing** — the curves splay apart with no intersection down to the lowest
   temperature reached. A method that finds a transition in 2D is broken.
3. **1D is exactly solvable**, a free closed-form target for the machinery.

---

## Architecture

One parallel tempering driver; the model sits behind an interface. Everything the
sampler needs is:

```
energy(s)         -> scalar total energy
local_fields(s)   -> array h_i, so a flip costs dE = 2 * s_i * h_i
couplings         -> built once from a seeded RNG
```

- **EA** supplies a sparse neighbor list ($2d$ neighbors per site). The cubic
  lattice is bipartite, so an entire sublattice can be flip-proposed at once in
  numpy — essential, since a pure-Python site loop over $512$ sites $\times\ 20$
  temperatures $\times\ 10^5$ sweeps $\times\ 100$ disorder samples will not finish.
- **SK** supplies a dense $N \times N$ matrix; the local field is one matvec. Not
  bipartite, so no checkerboard — instead lean on BLAS and accept smaller $N$
  (a few thousand spins max, since each flip costs $O(N)$).

The driver — ladder, swaps, measurement, binning — never knows which model it is
driving. That is what makes "later evolve to SK" an addition rather than a rewrite.

No `try`/`except`, no silent defaults. Let it crash.

---

## Steps, each with its consistency check

Every step below has a check that is self-contained, understandable, and would
**actually fail** if the step were wrong. Nothing proceeds until its check passes.

### Step 0 — Ferromagnet first

Set every $J = +1$ and run the EA code as a plain Ising model.

> **Check.** 2D must reproduce Onsager: $T_c = 2/\ln(1+\sqrt{2}) \approx 2.2692$.
> 3D must give $T_c \approx 4.5115$. This validates energy bookkeeping, Metropolis
> acceptance, and measurement *before* disorder enters. Debugging a wrong $T_c$ in
> a clean ferromagnet is tractable; debugging it in a glass is not.

### Step 1 — Disorder, single-spin Metropolis

Quenched Gaussian couplings, standard Metropolis.

> **Check A — frustration is real.** Count elementary plaquettes whose four bonds
> have a negative product; no assignment of those four spins can satisfy all four
> bonds. For any symmetric coupling distribution the four signs are independent
> and symmetric, so the fraction is **exactly 1/2**, independent of $L$ and $d$ —
> and **exactly 0** for the Step 0 ferromagnet. Static, no sampling, and it
> confirms the defining feature of the model is actually present.
>
> **Check B — incremental vs. full energy.** Accumulate $E$ from $\Delta E$ as
> flips are accepted, and every $N$ sweeps recompute from scratch. They must agree
> to machine precision. This catches sign errors and double-counted bonds, the two
> most common bugs in this code. It does more here: the accumulated $\Delta E$
> equals the true change *only* if no two simultaneously-flipped spins are
> neighbours, so a broken checkerboard (odd $L$, or both sublattices at once)
> appears as drift growing with sweep count.
>
> **Check C — high-temperature limit.** Expanding $\tanh$,
> $$e\,T = -d\,\langle J^2 \rangle + \frac{d\,\langle J^4 \rangle}{3\,T^2} + O(T^{-4})$$
> so $e\,T$ against $1/T^2$ is a straight line whose intercept is the prediction.
> (Lattice loop corrections enter at the same order but vanish on disorder
> average, since $\langle J_{ij}J_{jk}J_{kl}J_{li}\rangle = 0$.) No free parameters.
>
> Two design points this check needs to be meaningful, both learned the hard way:
> compare against each sample's **empirical** $\langle J^2 \rangle$ rather than 1
> (a finite lattice has only $dN$ couplings, so its sample variance is several
> percent off — larger than the statistical error); and keep the temperature
> window **moderate** ($T \sim 6\text{–}16$), because $\sigma(e)$ is
> $T$-independent at high $T$, so $\sigma(e\,T)$ grows *linearly* with $T$ and
> extreme temperatures add noise rather than information.
>
> Assert on the **mean** discrepancy against its standard error, not on the mean
> of $|$discrepancies$|$ — the latter measures per-sample scatter and does not
> shrink with more samples however correct the code is. Also assert the error bar
> is small, so the test cannot pass vacuously.
>
> **Both** fit parameters are predicted, so test both: intercept $-d\langle J^2\rangle$
> and slope $d\langle J^4\rangle/3$. Plot the per-sample *residual*
> $e\,T + d\langle J^2\rangle_k$ against $1/T^2$ — a line through the origin with
> the predicted slope — rather than raw $e\,T$. Subtracting each sample's own
> $\langle J^2\rangle$ removes the realisation-to-realisation variation in coupling
> variance, which is real physics rather than uncertainty in the law, and which
> otherwise inflates the error bars by roughly an order of magnitude and makes the
> check look far weaker than it is.

### Step 2 — Exact enumeration

The strongest check available, and cheap. Use **open** boundaries on a small
system: 2D $L=4$ is 16 spins (65 536 states); 3D $L=2$ open is 8 spins and 12
bonds (256 states). Enumerate every state, build $Z$ exactly, and compute
$\langle E \rangle$, $\langle q^2 \rangle$, and $C$ at a dozen temperatures.

> **Check.** Monte Carlo must match exact enumeration within error bars across the
> whole temperature range — for $e$, for the heat capacity $c$, and for
> $\langle q^2 \rangle$.
>
> Include $\langle q^2 \rangle$, not just the energy. Two replicas share couplings
> but are otherwise independent, so it factorises exactly:
> $$\langle q^2 \rangle = \frac{1}{N^2}\sum_{ij} \langle s_i s_j \rangle^2$$
> which enumeration gives directly, while the Monte Carlo side has to get it from
> two genuinely independent chains. That makes this a real test of the replica
> machinery Step 4 depends on, not only of the energy.
>
> **Trap.** With $L=2$ and periodic boundaries, site $i$ is its own neighbor twice
> along each axis and bonds get double-counted. Use open boundaries here.
>
> **Implementation.** Get open boundaries by setting the wrap-around couplings to
> **zero**, not by branching inside `energy` / `local_fields`. A zero coupling
> contributes nothing, so both keep their single validated code path — and the
> $L=2$ double-bond problem disappears automatically, since the wrap bond is
> exactly the one that gets zeroed, leaving one bond per line.
>
> **What this check actually found.** Plain Metropolis cannot equilibrate even a
> **16-spin** system at $T = 0.4$: acceptance there is $\sim 10^{-3}$, so 20 000
> sweeps buy only a few hundred accepted flips. The failure has a diagnostic
> *direction* — the Monte Carlo energy sits **above** the exact value while
> $\langle q^2 \rangle$ sits **below** it, both saying the chains never reached the
> low-lying states. That is the Bhatt–Young signature of Step 6, appearing in a
> system small enough to have an exact answer: the argument for parallel tempering
> made quantitative instead of asserted. Assert over $T \ge 0.6$ and print the
> $T = 0.4$ row as a documented failure rather than hiding it.

### Step 3 — Gauge invariance ($\pm J$ only)

For any fixed pattern $\sigma_i = \pm 1$, the transformation
$s_i \to \sigma_i s_i$, $J_{ij} \to \sigma_i \sigma_j J_{ij}$ leaves the partition
function **exactly** invariant.

> **Check.** Generate a random $\sigma$ pattern, gauge-transform the couplings,
> rerun. The energy histogram must be statistically identical. A few lines, and it
> tests the coupling construction in a way nothing else does.

### Step 4 — Two replicas and $P(q)$

Two independent copies with the *same* couplings, different initial conditions and
independent RNG streams. Measure $q$, histogram it.

> **Check A — symmetry.** $P(q)$ must be symmetric under $q \to -q$ (global Ising
> symmetry). Visible asymmetry means the run is not thermalized or the replicas
> are correlated — both fatal.
>
> **Check B — high-$T$ shape.** Well above $T_c$, $P(q)$ must be a Gaussian
> centered at zero of width $\sim 1/\sqrt{N}$. Verify the width quantitatively
> against $N$, not by eye.
>
> **Binning trap.** $q$ is **quantised** in steps of $2/N$, since
> $q = (N - 2m)/N$ for integer $m$. Histogram bins narrower than that spacing
> produce a comb — full bins alternating with empty ones — which looks like
> structure and is not. At $L = 8$ ($N = 512$) a 101-bin histogram holds ~5
> allowed values per bin and is fine; at $L = 4$ ($N = 64$) it is not. Merge
> adjacent bins until each holds at least two allowed values.

### Step 5 — Parallel tempering

Single-spin Metropolis freezes solid below $T_c$; this step is what makes the
project possible. Roughly 20–24 temperatures spanning $T \in [0.4, 2.0]$ for 3D
Gaussian, geometric spacing to start, then tuned for uniform acceptance.

> **Check A — swap acceptance.** Roughly uniform at 20–40% across *all* adjacent
> pairs. A pair stuck near zero means the ladder has a gap, typically at the
> specific-heat peak.
>
> **Check B — round trips.** Every replica must travel from the lowest temperature
> to the highest and back several times during the run. The round-trip count is the
> honest thermalization diagnostic; "the energy looked flat" is not.
>
> **Check C — rerun Step 2 with PT (the decisive one).** Point parallel tempering
> at the *same* 8-spin lattice, the same couplings and the same temperatures as the
> exact enumeration, and confirm the $T = 0.4$ row that failed under plain
> Metropolis now passes. Same geometry, same exact answer, only the sampler
> changed — a demonstration rather than an assertion, and the single best evidence
> that PT is doing what it claims.
>
> **Acceptance can also be too *high*.** 0.75–0.88 at $L = 4$ means the ladder is
> denser than necessary and compute is being wasted on redundant rungs. Acceptance
> falls as $\sqrt{N}$, so the largest $L$ sets the rung count; print the minimum
> acceptance every production run.

### Step 6 — Thermalization, done properly

Log-binning: measure over the last half, last quarter, and last eighth of the run
and confirm the results have stopped moving.

> **Check.** The spin-glass-specific test (Bhatt–Young): $\langle q^2 \rangle$
> measured from the replica overlap approaches equilibrium **from below**, while
> the link overlap approaches **from above**. When the two meet, the run is
> thermalized. Worth implementing — "it looked converged" is how most wrong spin
> glass results happen.

### Step 7 — Locate $T_c$ in 3D

Finite-size scaling of the dimensionless ratio $\xi_L / L$ for $L = 4, 6, 8$
(and 10 if affordable). Curves for different $L$ cross at $T_c$.

> **Check A.** The crossing must land near $T_c \approx 0.95$ for Gaussian bonds.
>
> **Check B — the 2D negative control.** Same pipeline, `d=2`: no crossing at any
> temperature reached.
>
> **Check C — universality.** Rerun 3D with $\pm J$ bonds. $T_c$ must shift
> (to $\approx 1.12$) but the critical exponents must not.

### Step 8 — SK, against Parisi's exact numbers

Swap in the dense model; the driver is unchanged.

> **Check A.** $T_c = 1$.
>
> **Check B.** Ground-state energy per spin $\to -0.7633$ as $N \to \infty$.
> Finite-$N$ values approach it **from above** with an $N^{-2/3}$ correction —
> extrapolate and compare.
>
> **Check C.** $P(q)$ must show the Parisi plateau filling the region between the
> two peaks, qualitatively unlike the ferromagnet's two bare spikes.

### Step 9 — The comparison

Put the 3D EA $P(q)$ beside the SK $P(q)$ at matched reduced temperature $T/T_c$
and discuss what the 3D shape does and does not settle about Parisi vs. droplet.
This is the writeup, and it should be honest that the question is open.

---

## The cost that will dominate everything

**$P(q)$ is not self-averaging.** Unlike the energy, $P(q)$ does *not* converge to
a single shape as $N$ grows — individual disorder realizations give wildly
different, spiky distributions, and only the average over samples is smooth. This
needs $O(100\text{–}1000)$ coupling realizations, each with its own full parallel
tempering run, and it dominates the compute budget entirely.

Structure the code to run **one disorder sample end-to-end and write results to
disk**, then loop. The problem is embarrassingly parallel across samples.

Non-self-averaging is also a physics result in its own right, not just a nuisance —
it is one of the signatures separating the two pictures. Say so in the writeup.

---

## Decisions to make before writing code

**Coupling distribution.** Gaussian gives a unique ground state (up to global flip),
no accidental degeneracy, and a cleaner $P(q)$ — better for the headline
measurement. $\pm J$ gives integer energies, enables the gauge check (Step 3), and
opens the door to multispin coding. They are one line apart. **Default: Gaussian
for the physics, $\pm J$ kept available** for Steps 3 and 7C.

**System sizes.** $L = 4, 6, 8$ in 3D is realistic in numpy with checkerboard
updates. $L = 12$ or 16 needs the push below.

---

## Now push

- **Multispin coding.** With $\pm J$ couplings, pack 64 disorder realizations into
  the bits of a `uint64` and update all of them with bitwise operations in one
  pass. This is the technique the Janus collaboration built custom hardware
  around; a ~64× speedup reaches $L = 12\text{–}16$ instead of 8.
  *Check: multispin and plain results must agree sample for sample — bit for bit
  at $T=0$, and within error bars at finite $T$.*
- **Ultrametricity in SK.** Draw three replicas, measure all three pairwise
  overlaps, and test whether the two smallest are nearly equal — the signature of
  the hierarchical, family-tree organization of states that Parisi's solution
  predicts. A direct look at the structure behind the plateau.
- **Domain-wall stiffness.** Compute ground-state energies with periodic vs.
  antiperiodic boundaries; the defect energy scales as $L^\theta$. In 2D,
  $\theta \approx -0.28$ (negative — walls cost less as $L$ grows, hence
  $T_c = 0$); in 3D, $\theta \approx +0.2$. Measuring the *sign flip* between 2D
  and 3D explains the lower critical dimension directly.
- **Temperature chaos.** Measure the overlap between equilibrium configurations at
  two nearby temperatures and watch it collapse as $L$ grows.

---

## Environment note

Only `numpy` and `matplotlib` are installed — no `scipy`, no `torch`. Nothing in
this plan needs them, but it does mean hand-rolling the histogramming, the
jackknife/bootstrap error bars, and the crossing-point fit in Step 7. For a course
like this that is a feature.
