#!/bin/bash
# sync.sh — Stage, commit, and push website changes to GitHub Pages
# Usage:
#   ./sync.sh                      # auto-generates commit message from changed files
#   ./sync.sh "My commit message"  # use a custom commit message

set -e

SITE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SITE_DIR"

# ── Check for changes ────────────────────────────────────────
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "Nothing to sync — working tree is clean."
  exit 0
fi

# ── Stage all changes ─────────────────────────────────────────
git add -A

# ── Build commit message ──────────────────────────────────────
if [ -n "$1" ]; then
  MSG="$1"
else
  CHANGED=$(git diff --cached --name-only | head -6 | tr '\n' ', ' | sed 's/,$//')
  COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')
  MSG="Update website: $COUNT file(s) changed — $CHANGED"
fi

# ── Commit ────────────────────────────────────────────────────
git commit -m "$MSG"

# ── Push main → master (GitHub Pages branch) ─────────────────
echo ""
echo "Pushing to GitHub Pages (master)..."
git push origin main:master

echo ""
echo "Done! Site will be live at https://shibaji7.github.io in ~60 seconds."
echo "Changes: $(git log --oneline -1)"
