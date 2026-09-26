# The Contour "Conjecture"
## The Problem

Take a meromorphic function and integrate it around a
closed loop $C$ in the complex plane. The residue theorem says

$$\oint_C f(z)\,dz = 2\pi i \sum_k n(C, p_k)\,\mathrm{Res}_{p_k} f,$$

where $n(C, p_k)$ counts how many times the loop winds around the pole $p_k$. The left
side is a line integral you evaluate by parametrizing the path and calling a
quadrature routine. The right side is algebra: find the poles, take residues, count
windings. The two computations share no machinery at all, and they have to agree.


The first time I saw this, it seemed like a bit of a miracle --- hard to believe, too good to be true, etc. But you know it's true, you've taken courses on it.

But for the moment, **pretend** like you don't know it's true. You're shocked that anyone would propose such a thing, and you want to stress test it.

## Goal

Agentic models are rapid prototypers. To stress test this conjecture in the pre-agentic age might have take a good coder half a day, or, someone like me, longer.

Your task is to build evidence for the conjecture by writing numerical routines for both sides of the theorem and testing them against each other.

## Natural Steps

1. A numerical integrator for the LHS.
2. An evaluator for the RHS.
3. Non-trivial test-cases that show they're equal and compute the error.

## Push Harder

You really should do this, I did and it's incredible.

What would you really want? A GUI that lets you specify a meromorphic function, plots its poles, lets you draw the integration contour, and then evaluated the LHS and RHS to show that the residue theorem works.

## Push Even Harder

If you get to here, consider other creative ways to i) push the science or ii) make a really useful tool.