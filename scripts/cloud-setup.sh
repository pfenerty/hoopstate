#!/bin/bash
# Setup script for Claude Code cloud sessions (web / mobile).
#
# Paste the contents of this file into the Setup script field of the cloud
# environment at claude.ai/code. It is committed here so the canonical version
# is reviewable and versioned, but nothing reads it from the repo automatically.
#
# Constraints imposed by the platform, all of which this script respects:
#   - runs as root on Ubuntu 24.04, before Claude Code launches
#   - MUST exit zero, or the session fails to start (hence `|| true` throughout)
#   - MUST finish within five minutes
#   - only provisions the VM; repo-level setup belongs in the SessionStart hook
#
# Every domain touched here is on the default network allowlist.

set -u

# --- beads -----------------------------------------------------------------
# All task tracking for this project lives in beads. It is not pre-installed,
# and without it a cloud session cannot read or update the backlog.
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash || true
for candidate in "$HOME/.local/bin/bd" /root/.local/bin/bd; do
    [ -x "$candidate" ] && install -m 0755 "$candidate" /usr/local/bin/bd 2>/dev/null && break
done
command -v bd >/dev/null 2>&1 && bd version || echo "WARN: bd unavailable; backlog will be read-only via .beads/issues.jsonl"

# --- CPython 3.14 ----------------------------------------------------------
# uv is pre-installed but Ubuntu 24.04 ships an older interpreter. uv fetches a
# standalone 3.14 from python-build-standalone on GitHub releases. Pre-warming
# it here keeps it off the session's clock. Verified working with no system
# Python 3.14 and no flox present.
export UV_PYTHON_DOWNLOADS=automatic
uv python install 3.14 || true

echo "hoopstate cloud setup complete"
