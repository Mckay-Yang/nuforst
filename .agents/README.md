# Project Agent Files

This directory stores project-specific context for AI agents working on this
repository.

The main human-facing instructions remain in root `AGENTS.md`. Files under
`.agents/skills/` are narrower task guides that agents can read before doing
specialized work.

Current skills:

- `nufrost-project`: project structure, method boundaries, naming, and data
  conventions.
- `nufrost-rust-workflow`: build, test, cache, and CLI workflow instructions.
- `nufrost-experiments`: evaluation and experiment workflow notes.

These files are intentionally tracked because they describe repository behavior,
not generated data.
