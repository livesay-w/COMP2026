# Agentic Sessions

These are the sessions where we use the full power of AI agents to push as far as we can. I want you to figure out what works and what doesn't *by experimentation,* subject to the guidelines below. I'm being intentionally minimal at first pass; we'll fill this in as we learn what the best practices for this course actually are.

# Tools

You are free to use whatever tools you'd like, provided that no subscription you purchase costs more than $20/month. I encourage you to use the tools that are free to you — they are excellent. They're described below.

### Claude Code

Northeastern provides all students with a Claude account, with Claude Code access. The upside of Claude Code (CC) is that it is the most powerful tool we will use; the downside is that it can do a lot without you understanding what's happening.

**Success Modes:** For this course, when using CC, you want to be the architect and CC is the contractor.

**Failure Modes:** If it feels like CC is the architect, making all of the decisions, you will not learn as much and CC is more liable to make a mis-step. To be clear, *these models know many things* and can teach you new techniques, but you should *work with them, not for them.*

**Getting Started:** `cd` into your repo and run `claude`, then `/init` once — it writes a `CLAUDE.md` that CC reads every session, where you keep conventions (units, `hbar=1`, how to run things). `Shift+Tab` toggles plan mode, where CC proposes without editing.

**Rules of Thumb:**
- Commit before every agent turn, and start with `git status` clean. It's the only undo that always works.
- Be in `auto-mode`, it will keep you from having to OK permissions every single step.
- Ask for short code that is easy to read and understand. No `try/except` fallbacks or silent defaults — in physics code they hide the bug you're hunting.
- Ask to chart a course before beginning, breaking the problem into steps.
- Each step should have a self-contained consistency check that 1) you understand and 2) is a non-trivial check that the step works.
- Verification occurs by a combination of these non-trivial checks and your own understanding of the code. The more the better.
- Model selection. `/model` in CC will pull up the model choices. Weaker models are fine for simple tasks. For hard tasks, use stronger models, such as Opus 5.
- Token budget. Keep an eye on your token usage and stay within the limits of your plan. Figure out when to use weaker models and when to use stronger models.

### GitHub Copilot

This is a free tool for students, and it is excellent for code completion. It is not as powerful as Claude Code, in my opinion, but it is better for building your own understanding.

**Success Modes:** Where the scientific details matter you will often need to touch the code yourself, and Copilot is excellent for in-line edits a few lines at a time. If CC mode makes you the architect, Copilot mode makes you both builders.

**Failure Modes:** Less likely than with Claude Code, but Copilot can over-do it and keep you from learning yourself. It can also make mis-steps.

**Rules of Thumb:**
- Turn Copilot **off** for Manual sessions — that is what "no autocomplete" means.
- Know how to turn Copilot on and off with the icon in the bottom corner. For instance: when you need to write a line based on your thinking, its good to have it off, but when you need help with a CS detail such as an indexing convention or string formatting, it is good to have it on.
- I find in-line Copilot most useful, but there is also a sidebar where you can have chats with Copilot.
- Copilot can debug: highlight the code, right click, and select "Ask Copilot" for a chat window where you can ask what is working and what isn't. A great way to learn about code you don't understand.
