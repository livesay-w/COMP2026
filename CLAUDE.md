# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Coursework for PHYS 7321 / COMP2026, *Computational Physics* (Fall 2026, Jim Halverson,
Northeastern). This is a personal fork of the instructor's repo, so it holds both
upstream course material and the student's own work.

The course is deliberately split into two modes, and the distinction governs how Claude
should behave in a given directory:

- **`notebooks/` — manual (Tuesday) sessions.** These are *agent-free by course rule*:
  no LLMs, no autocomplete. They are graded on the student's own understanding.
- **`agentic/` — agentic (Friday) sessions.** These are where agents are the point. Each
  session pushes a more advanced idea within the week's topic as far as possible, and
  three of them become portfolio projects.

See `README.md` for the syllabus and calendar, and `agentic/README.md` for the
instructor's guidance on working with agents.

## Do not solve the manual notebooks

Cells in `notebooks/*.ipynb` marked `# FILL IN` are the student's graded work. Do not
write those solutions. If asked about a manual notebook, explain concepts, review
already-written code, or debug an error the student hit — but leave the `# FILL IN`
bodies to them.

Cells marked `# CHECK (GIVEN)` / `# do not edit` are the instructor's assertions. Never
edit a check to make code pass; a failing check means the solution is wrong.

`.gitignore` excludes `notebooks/*_solutions.ipynb` — answer keys are never published.

## Git remotes

`origin` is the instructor's upstream repo and **push is disabled** on it. `personal`
(`github.com/livesay-w/COMP2026`) is the student's fork and the only push target.

```
git pull origin main    # get new course material
git push personal main  # submit work
```

Per `agentic/README.md`, commit before every agent turn and start from a clean
`git status` — it is the only reliable undo.

## Environment and commands

Python comes from the miniconda `base` env (`C:\Users\wrl62\miniconda3\python.exe`,
currently 3.13). There is no `requirements.txt`, no build step, no test suite, and no
linter. Only `numpy` and `matplotlib` are installed — `torch`, `jax`, and `scipy` are
**not**, so autodiff-based work (the variational ground state project needs it) requires
installing a framework first, or hand-written gradients.

```powershell
jupyter lab                              # or: jupyter notebook
python agentic/<NN_Topic>/<script>.py    # agentic work runs as plain scripts too
```

## Layout conventions

Agentic session directories are named to match their notebook: `notebooks/02_Neural_Networks.ipynb`
pairs with `agentic/02_Neural_Networks/`. Each agentic directory holds the instructor's
prompt as a markdown file (e.g. `Variational_Ground_State.md`) stating the problem, the
required consistency checks, and "now push" extensions.

## Working style for agentic sessions

The instructor's guidance in `agentic/README.md` is explicit, and it is the standard to
hold code in `agentic/` to:

- **The student is the architect; Claude is the contractor.** Propose a plan and break
  the problem into steps before writing code. Plan mode (`Shift+Tab`) is preferred for
  this.
- **Every step needs a self-contained, non-trivial consistency check** that the student
  can understand and that would actually fail if the step were wrong. Physics gives good
  ones: a variational energy may come out above $E_0$ but never below it; an overlap
  $|\langle\psi|\psi_\text{exact}\rangle|$ should approach 1; a hand-derived gradient
  should match central finite differences.
- **Short, readable code.** No `try`/`except` fallbacks and no silent defaults — in
  physics code they hide the bug being hunted. Let it crash.
- **Natural units.** Problems are posed with $\hbar = m = \omega = 1$ unless stated
  otherwise; keep that convention and say so in comments.
- Seed RNGs (`np.random.default_rng(0)`, `np.random.seed(0)`) so results are
  reproducible for grading.
