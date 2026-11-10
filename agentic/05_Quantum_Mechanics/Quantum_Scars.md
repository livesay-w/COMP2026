# Quantum Billiards and Scars

## The Problem

A free particle is confined to a 2D region with hard walls, i.e. a billiards table or some shape. The problem is defined by the Shrodinger equations and the boundary.

$$-\nabla^2 \psi = E\psi, \qquad \psi = 0 \text{ on the wall.}$$

The results depend crucially on the shape. A rectangle is fairly straightforward, but a stadium, i.e. two half discs joined by straight sides, is chaotic, and its trajectories
separate exponentially the same way the Lorenz system's do.

A **periodic orbit** is a trajectory that
closes on itself and repeats forever, e.g. a ball bouncing straight up and down between
the two flat walls. In a chaotic table almost all of these are **unstable**, meaning a
particle started a hair off one leaves within a few bounces. And an eigenstate is
called **ergodic** if $|\psi|^2$ is spread evenly over the table with no structure
beyond speckle.

Quantum mechanically the question of chaos is more complicated. The Schrödinger equation is linear
and the eigenstates just sit there, as they are stationary states. So the question is what the classical chaos does to
the quantum problem, if anything.

However, some eigenstates of the stadium pile up along
a classical periodic orbit instead of spreading out, i.e. they look like that orbit
with quantum uncertainty smeared over it. These are called quantum **scars**. Heller found them in 1984:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1515

## Goal

Solve for a few hundred eigenstates of a stadium, find the scarred ones. Compared to the rectangle case.

## Natural Steps

1. A finite difference eigensolver can be used to solve the Schrodinger equation with boundary conditions built in by masking.
2. Test it where you already know the answer: a rectangle has
   $E_{nm} = \pi^2(n^2/L_x^2 + m^2/L_y^2)$, degeneracies and all.
3. Swap the rectangle mask for a stadium mask and look at a few dozen states.
4. Something that measures how localized a state is, so you can rank them instead of
   flipping through pictures. The inverse participation ratio $I = \sum_i |\psi_i|^4
   \Delta x$ is helpful.
5. Sort by inverse participation ratio and plot some of the more localized wavefunctions, maybe 20.

## Push Harder

The scars are the famous part, but the spectrum carries a cleaner signature of the
same thing. Unfold your eigenvalues so the mean spacing is one, then look at the gaps
between neighbors. An integrable shape gives Poisson statistics, with levels perfectly
willing to sit on top of each other. A chaotic one gives level repulsion and the Wigner
distribution, which is what you would get from a random matrix. That is the
Bohigas-Giannoni-Schmit conjecture, also from 1984, still unproven, and changing a mask
is enough to watch it happen.

Two warnings: the full stadium has two reflection
symmetries, so its spectrum is really four independent spectra laid on top of one
another and the statistics come out looking wrong; solve a quarter stadium with hard
walls on the symmetry axes instead. And a square has so many accidental degeneracies
coming from $n^2 + m^2$ that it makes a bad simple example, so give your rectangle
an irrational aspect ratio to keep the numbers cleaner.

After that: a disc staircases on a square grid where a rectangle doesn't, so find out
what that costs you. Try a Sinai billiard, or an L-shape. And if you can find a real
periodic orbit by root-finding the bounce map and lay it on top of a scarred state,
that is the figure worth putting in your writeup.
