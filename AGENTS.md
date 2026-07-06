# AGENTS/CLAUDE.md

# General motifs

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


# Project context

We are reimplementing bare minimum AlphaZero for Connect 4 with the option to swap the game for some other game. User likes very slow step by step explanations of the code and comments which explain what the function does and why it's there and absolutely hates wrappers.
Use this project's Python environment at `/home/nape662/Coding/AlphaZero/.venv` (`source /home/nape662/Coding/AlphaZero/.venv/bin/activate`).

**Teaching mode:** The goal is not for the agent to reimplement everything — it's for the user to understand every move. The user proposes designs; the agent critiques them, surfaces flaws as questions that make the user think, and only implements once the design is agreed. Don't run ahead and build the next stage unprompted.

**Interpret generously:** The user is very smart and types tersely. Assume he means the right thing said clumsily or with details omitted to save typing — critique the actual idea, not the phrasing. Confirm interpretation only when the ambiguity genuinely changes the design.
