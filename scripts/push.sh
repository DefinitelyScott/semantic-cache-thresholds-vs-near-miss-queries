#!/usr/bin/env bash
#
# push.sh: verify, then commit and push the current state of the repo.
#
# Usage:
#   ./scripts/push.sh "feat: add an entity-mismatch check to the guard"
#   ./scripts/push.sh                # uses a default commit message
#
# It refuses to push unless the test suite passes AND the offline experiment
# runs cleanly, so a broken increment (or a README number that no longer
# matches the code) can never reach the public repo. Run it from anywhere
# inside the repo.
set -euo pipefail

# Move to the repo root (the directory above this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Prefer the project virtualenv if present so pytest resolves without manual
# activation. Override with PYTHON=... if desired.
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi
MSG="${1:-chore: update semantic-cache-thresholds-vs-near-miss-queries}"

echo "==> Running test suite"
"$PYTHON" -m pytest -q

echo "==> Running the experiment (offline)"
"$PYTHON" scripts/run_experiment.py > /dev/null

# Nothing to commit if the working tree is clean.
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "==> No changes to commit. Pushing any unpushed commits."
else
    echo "==> Staging and committing changes"
    git add -A
    git commit -m "$MSG"
fi

# Ensure a remote exists before attempting to push.
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: no 'origin' remote configured." >&2
    echo "Set it once with:" >&2
    echo "  git remote add origin git@github.com:DefinitelyScott/semantic-cache-thresholds-vs-near-miss-queries.git" >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> Pushing $BRANCH to origin"
git push -u origin "$BRANCH"
echo "==> Done."
