# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
cd python
uv sync --group dev     # UV_PYTHON_DOWNLOADS=automatic if no system 3.14
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

**flox is a convenience, not a requirement.** `flox activate --` works locally, but
every command above runs without it: `uv` sources a standalone CPython 3.14 from
python-build-standalone when no system 3.14 is present. This is verified in CI and is
the path cloud sessions take. Never make flox load-bearing — it does not exist in a
cloud sandbox.

## Working from cloud sessions (web / mobile)

Cloud sessions clone **the GitHub remote at the current branch, not the local
checkout**, so unpushed work is invisible to them. Push before starting one.

Environment: Ubuntu 24.04, 4 vCPU / 16 GB RAM / **30 GB disk**, a fresh VM per session,
nothing persisting between sessions. `uv`, `ruff`, `pytest`, `rustc` and `cargo` are
pre-installed; `flox` and `bd` are not. `scripts/cloud-setup.sh` installs `bd` and
pre-warms Python 3.14 — paste its contents into the Setup script field of the cloud
environment at claude.ai/code.

Network is allowlisted. PyPI, crates.io, `github.com` and `raw.githubusercontent.com`
are all permitted, which covers dependency installs and the entire
`shufinskiy/nba_data` archive. Do not add a data source on another domain without
checking it against the allowlist first.

**Cloud sessions use the `ephemeral` storage profile** — one scratch directory, no NAS,
no hot/cold split, data re-fetched per session. 30 GB is ample for a season end to end.
Issues labelled `local-only` (NAS preflight, toolchain relocation) cannot be worked
from a cloud session, and are deliberately kept off the critical path: no
cloud-reachable work is ever gated on them.

CI is the feedback loop when there is no terminal. Every push runs lint, format and
tests on Ubuntu against a standalone 3.14.

## Architecture Overview

Parquet is the source of truth; DuckDB holds views plus materialized gold marts and is
always rebuildable and safe to delete. Zones: `bronze/` (typed, 1:1 with source),
`silver/` (canonical_event, lineup_stint, possession, possession_chance), `gold/`
(versioned analysis marts — the future API contract), and `oracle/` (pbpstats
reference, **quarantined**: the test suite may read it, the core model never).

All zone paths resolve through a storage profile; nothing downstream knows which
profile is active. The DuckDB file stays on local disk under every profile — DuckDB
does not support database files on network filesystems.

## Conventions & Patterns

_Add your project-specific conventions here_
