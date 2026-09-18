# A Neural Network Ground State and Beyond

## The Problem

A neural network is just a flexible function, and the variational principle says that
for any trial wavefunction,

$$E[\psi] = \frac{\langle \psi | H | \psi\rangle}{\langle \psi | \psi \rangle} \ge E_0,$$

with equality only for the true ground state. So put the two together. Let the network
be the wavefunction, make the energy the loss, and run the training loop from lecture.
Gradient descent on $\langle H \rangle$ then becomes a search for the ground state.

Start with the 1D harmonic oscillator, $\hbar = m = \omega = 1$:

$$H = -\tfrac{1}{2}\frac{d^2}{dx^2} + \tfrac{1}{2}x^2, \qquad E_0 = \tfrac{1}{2}.$$

## Your Checks

The variational principle gives you a one-sided check, which is unusual and very
useful. Your energy is allowed to come out too high, since that only means the ansatz
or the optimization is imperfect. It is not allowed to come out below $1/2$. If it
does, you have a bug somewhere, and it is usually either a missing normalization or a
mishandled kinetic term.

A few more worth having in place before you start:

- $E \to 1/2$ tells you that you found the right state.
- The overlap $|\langle \psi | \psi_\text{exact}\rangle| \to 1$ tells you that you
  found the right function and not only the right number.

Decide on your checks before you write code. See [../README.md](../README.md).

## A Couple of Natural Steps

Represent $\psi$ on a grid and get $\psi'$ by autodiff. Compute the kinetic energy as
$\tfrac12\int |\psi'|^2$ rather than $-\tfrac12\int \psi\,\psi''$, which needs fewer
derivatives and behaves better numerically, though you should be able to say why you
are allowed to drop the boundary term. Normalize at every step rather than at the end.

## Now Push

Go further! Do something interesting! Suggestions:
1. Anharmonic oscillator, $V(x) = \tfrac12 x^2 + \lambda x^4$, using perturbation theory to check the numerical energy level.
2. Excited states with NNs, using the orthogonality of the wavefunctions to check your work.
3. Double well, $V(x) = \lambda(x^2 - a^2)^2$.
