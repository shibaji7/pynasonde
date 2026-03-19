# =============================================================================
#  pynasonde — Project Makefile
#
#  Everyday targets:
#    make clean                    Remove all build/cache/test artefacts
#    make format                   Run isort + autoflake + black in-place
#    make install-dev              pip install -e .[dev]
#    make test                     Run pytest with coverage
#    make build-dist               Format → clean → build sdist + wheel
#    make sync [MSG="..."]         Commit + push (replaces sync.sh)
#
#  Release targets:
#    make check   VERSION=X.Y      Consistency check only (no changes)
#    make release VERSION=X.Y      Full interactive release pipeline
#
#  Each release step describes what it will do and asks Y/N before running.
# =============================================================================

SHELL     := /bin/bash
.ONESHELL:
.PHONY: help clean format install-dev test build-dist sync check release

VERSION   ?= __UNSET__
FULL_VER  := $(VERSION)
TAG       := v$(FULL_VER)
MSG       ?=

# --------------------------------------------------------------------------- #
help:
	@echo ""
	@echo "  pynasonde — available targets"
	@echo "  ═══════════════════════════════════════════════════════════════"
	@echo "  Everyday"
	@echo "  ───────────────────────────────────────────────────────────────"
	@echo "  make clean                  Remove build/cache/test artefacts"
	@echo "  make format                 isort + autoflake + black (in-place)"
	@echo "  make install-dev            pip install -e .[dev]"
	@echo "  make test                   pytest with coverage"
	@echo "  make build-dist             format → clean → sdist + wheel"
	@echo "  make sync                   Commit all changes and push to GitHub"
	@echo "  make sync MSG=\"my message\"  Commit with a custom message"
	@echo ""
	@echo "  Release"
	@echo "  ───────────────────────────────────────────────────────────────"
	@echo "  make check   VERSION=X.Y.Z   Version consistency check (read-only)"
	@echo "  make release VERSION=X.Y.Z   Full release pipeline (interactive)"
	@echo "  ═══════════════════════════════════════════════════════════════"
	@echo ""

# --------------------------------------------------------------------------- #
#  clean — remove all build, cache, and test artefacts
# --------------------------------------------------------------------------- #
clean:
	@set -euo pipefail
	echo ""
	echo "  Cleaning build artefacts, caches, and test outputs..."
	echo ""
	rm -rf dist/ build/ site_build/ site/
	rm -rf pynasonde.egg-info/ *.egg-info/
	rm -rf .eggs/
	rm -rf junit.xml .coverage coverage.xml .pytest_cache/
	find . -type d -name '__pycache__'       -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -o -type f -name '*.pyo' | xargs rm -f 2>/dev/null || true
	echo "  ✔  Clean complete."
	echo ""

# --------------------------------------------------------------------------- #
#  format — isort → autoflake → black
# --------------------------------------------------------------------------- #
format:
	@set -euo pipefail
	echo ""
	echo "  Running formatters (isort → autoflake → black)..."
	echo ""
	isort -rc -sl .
	autoflake --in-place --imports=SDCarto,scienceplots -r .
	isort -rc -m 3 .
	black .
	echo "  ✔  Format complete."
	echo ""

# --------------------------------------------------------------------------- #
#  install-dev — editable install with dev extras
# --------------------------------------------------------------------------- #
install-dev:
	@set -euo pipefail
	echo ""
	echo "  Installing pynasonde in editable mode with dev dependencies..."
	pip install -e .[dev]
	echo "  ✔  Done."
	echo ""

# --------------------------------------------------------------------------- #
#  test — run the full test suite with coverage
# --------------------------------------------------------------------------- #
test:
	@set -euo pipefail
	echo ""
	echo "  Running pytest..."
	echo ""
	python -m pytest
	echo ""

# --------------------------------------------------------------------------- #
#  build-dist — format + clean artefacts + build sdist + wheel
# --------------------------------------------------------------------------- #
build-dist:
	@set -euo pipefail
	echo ""
	echo "  Building source distribution and wheel..."
	echo "  (runs format → clean artefacts → python -m build)"
	echo ""
	isort -rc -sl .
	autoflake --in-place --imports=SDCarto,scienceplots -r .
	isort -rc -m 3 .
	black .
	rm -rf dist/ build/ pynasonde.egg-info/ *.egg-info/
	if ! python -m build --version &>/dev/null; then
	  echo "  ⚠  python -m build not found — installing..."
	  pip install build
	fi
	python -m build --sdist --wheel
	echo ""
	echo "  ✔  Build complete. Artifacts:"
	ls -lh dist/ | sed 's/^/    /'
	echo ""

# --------------------------------------------------------------------------- #
#  sync — stage all, commit (auto or custom message), push  [replaces sync.sh]
#   Usage:  make sync
#           make sync MSG="feat: add new parser"
# --------------------------------------------------------------------------- #
sync:
	@set -euo pipefail

	confirm() {
	  local prompt="$$1"
	  local reply
	  printf "\n  $$prompt  [y/N] → "
	  read -r reply
	  [[ "$${reply,,}" == "y" ]]
	}

	# Check for any changes at all
	if git diff --quiet && git diff --cached --quiet && \
	   [ -z "$$(git ls-files --others --exclude-standard)" ]; then
	  echo ""
	  echo "  ✔  Nothing to sync — working tree is clean."
	  echo ""
	  exit 0
	fi

	# Build commit message
	if [[ -n "$(MSG)" ]]; then
	  COMMIT_MSG="$(MSG)"
	else
	  CHANGED=$$(git diff --name-only; git diff --cached --name-only; git ls-files --others --exclude-standard | head -6 | tr '\n' ',' | sed 's/,$$//')
	  COUNT=$$(  git diff --name-only; git diff --cached --name-only; git ls-files --others --exclude-standard | wc -l | tr -d ' ')
	  COMMIT_MSG="Update codebase: $${COUNT} file(s) changed — $${CHANGED}"
	fi

	CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD)

	echo ""
	echo "  Modified / untracked files:"
	git status --short | sed 's/^/    /'
	echo ""
	echo "  Will run:"
	echo "    git add -A"
	echo "    git commit -m \"$${COMMIT_MSG}\""
	echo "    git push origin $${CURRENT_BRANCH}"
	echo ""

	if confirm "Commit and push to origin/$${CURRENT_BRANCH}?"; then
	  git add -A
	  git commit -m "$${COMMIT_MSG}"
	  git push origin "$${CURRENT_BRANCH}"
	  echo ""
	  echo "  ✔  Done!  $$(git log --oneline -1)"
	  echo "     https://github.com/shibaji7/pynasonde"
	else
	  echo "  ↳ Aborted."
	fi
	echo ""

# --------------------------------------------------------------------------- #
#  check — read-only version consistency check
# --------------------------------------------------------------------------- #
check:
	@set -euo pipefail
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo "  ERROR: No version provided.  Usage:  make check VERSION=1.1.0"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then
	  echo "ERROR: VERSION must be MAJOR.MINOR.PATCH (e.g. 1.1.0), got '$(VERSION)'"
	  exit 1
	fi
	echo ""
	echo "  Checking version consistency for $(FULL_VER) ..."
	echo ""
	SETUP_VER=$$(grep -oP '(?<=version=")[^"]+' setup.py || true)
	INIT_VER=$$(grep  -oP '(?<=__version__ = ")[^"]+' pynasonde/__init__.py 2>/dev/null || echo "__missing__")
	TOML_DYN=$$(grep  'dynamic.*version' pyproject.toml || echo "(dynamic — ok)")
	echo "  setup.py       version          = \"$${SETUP_VER:-<not found>}\""
	echo "  __init__.py    __version__      = \"$${INIT_VER}\""
	echo "  pyproject.toml                    $${TOML_DYN}"
	echo ""
	if [[ "$${SETUP_VER}" == "$(FULL_VER)" && \
	      ( "$${INIT_VER}" == "$(FULL_VER)" || "$${INIT_VER}" == "__missing__" ) ]]; then
	  echo "  ✔  All version references are consistent with $(FULL_VER)."
	else
	  echo "  ⚠  Inconsistency detected — run  make release VERSION=$(VERSION)  to fix."
	fi
	echo ""

# --------------------------------------------------------------------------- #
#  release — full interactive release pipeline
# --------------------------------------------------------------------------- #
release:
	@set -euo pipefail

	# ── 0. Validate input ─────────────────────────────────────────────────
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo "  ERROR: No version provided."
	  echo "  Usage:  make release VERSION=X.Y.Z   (e.g.  make release VERSION=1.1.0)"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then
	  echo "ERROR: VERSION must be MAJOR.MINOR.PATCH (e.g. 1.1.0), got '$(VERSION)'"
	  exit 1
	fi

	echo ""
	echo "╔══════════════════════════════════════════════════════════════╗"
	echo "║           pynasonde  —  Release $(FULL_VER)                "
	echo "║  Git tag  : $(TAG)                                          "
	echo "║  Remote   : git@github.com:shibaji7/pynasonde.git           "
	echo "╚══════════════════════════════════════════════════════════════╝"
	echo ""

	confirm() {
	  local prompt="$$1"
	  local reply
	  printf "\n  $$prompt  [y/N] → "
	  read -r reply
	  [[ "$${reply,,}" == "y" ]]
	}
	skip_step() { echo "  ↳ Skipped."; }

	# ── 1. Version consistency check & update ─────────────────────────────
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 1 — Check & update version references"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	SETUP_VER=$$(grep -oP '(?<=version=")[^"]+' setup.py || echo "__missing__")
	INIT_VER=$$(grep  -oP '(?<=__version__ = ")[^"]+' pynasonde/__init__.py 2>/dev/null || echo "__missing__")

	echo "  Current version references:"
	echo "    setup.py       version       = \"$${SETUP_VER}\""
	echo "    __init__.py    __version__   = \"$${INIT_VER}\""
	echo "    pyproject.toml dynamic = [\"version\"]  (driven by git tag — ok)"
	echo ""
	echo "  Target version : $(FULL_VER)"
	echo ""

	NEED_SETUP=false
	NEED_INIT=false
	[[ "$${SETUP_VER}" != "$(FULL_VER)" ]] && { echo "  ⚠  setup.py        → will be updated to \"$(FULL_VER)\""; NEED_SETUP=true; }
	if [[ "$${INIT_VER}" != "$(FULL_VER)" ]]; then
	  [[ "$${INIT_VER}" == "__missing__" ]] \
	    && echo "  ⚠  __init__.py     → __version__ will be added as \"$(FULL_VER)\"" \
	    || echo "  ⚠  __init__.py     → will be updated to \"$(FULL_VER)\""
	  NEED_INIT=true
	fi

	if [[ "$$NEED_SETUP" == "false" && "$$NEED_INIT" == "false" ]]; then
	  echo "  ✔  All version references already match $(FULL_VER)."
	else
	  if confirm "Apply version updates to setup.py and __init__.py?"; then
	    if [[ "$$NEED_SETUP" == "true" ]]; then
	      sed -i "s/version=\"[^\"]*\"/version=\"$(FULL_VER)\"/" setup.py
	      echo "  ✔  setup.py updated."
	    fi
	    if [[ "$$NEED_INIT" == "true" ]]; then
	      if [[ "$${INIT_VER}" == "__missing__" ]]; then
	        echo "__version__ = \"$(FULL_VER)\"" >> pynasonde/__init__.py
	        echo "  ✔  __version__ added to pynasonde/__init__.py."
	      else
	        sed -i "s/__version__ = \"[^\"]*\"/__version__ = \"$(FULL_VER)\"/" pynasonde/__init__.py
	        echo "  ✔  pynasonde/__init__.py updated."
	      fi
	    fi
	  else
	    skip_step
	  fi
	fi

	# ── 2. Commit working tree ─────────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 2 — Commit working tree changes"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	GIT_STATUS=$$(git status --short)
	if [[ -z "$$GIT_STATUS" ]]; then
	  echo "  ✔  Working tree is clean — nothing to commit."
	else
	  echo "  Modified / untracked files:"
	  git status --short | sed 's/^/    /'
	  echo ""
	  echo "  Will run:"
	  echo "    git add -A"
	  echo "    git commit -m \"chore: bump version to $(FULL_VER)\""
	  echo ""
	  if confirm "Commit all working-tree changes?"; then
	    git add -A
	    git commit -m "chore: bump version to $(FULL_VER)"
	    echo "  ✔  Committed."
	  else
	    skip_step
	  fi
	fi

	# ── 3. Create annotated git tag ────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 3 — Create annotated git tag"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	TAG_SKIPPED=false
	if git tag | grep -q "^$(TAG)$$"; then
	  echo "  ⚠  Tag $(TAG) already exists locally."
	  echo "  Will run:  git tag -d $(TAG)  then re-create it."
	  echo ""
	  if confirm "Delete and re-create tag $(TAG)?"; then
	    git tag -d $(TAG)
	    echo "  ✔  Old local tag deleted."
	  else
	    echo "  ↳ Keeping existing tag."
	    TAG_SKIPPED=true
	  fi
	fi

	if [[ "$$TAG_SKIPPED" != "true" ]]; then
	  DEFAULT_MSG="Release $(FULL_VER)"$$'\n\n'"Precision ionospheric radio sounding tools — pynasonde $(FULL_VER)"$$'\n\nChanges in this release:'$$'\n  - Bump version to $(FULL_VER)'
	  echo "  Will run:  git tag -a $(TAG) -m \"Release $(FULL_VER)\""
	  echo "  Tag message:"
	  echo "$$DEFAULT_MSG" | sed 's/^/    /'
	  echo ""
	  if confirm "Create annotated tag $(TAG)?"; then
	    git tag -a $(TAG) -m "$$DEFAULT_MSG"
	    echo "  ✔  Tag $(TAG) created."
	  else
	    skip_step
	  fi
	fi

	# ── 4. Push commits to GitHub ──────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 4 — Push commits to GitHub"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD)
	AHEAD=$$(git rev-list origin/$${CURRENT_BRANCH}..HEAD --count 2>/dev/null || echo "?")
	echo "  Branch  : $${CURRENT_BRANCH}"
	echo "  Commits ahead of origin : $${AHEAD}"
	echo "  Will run:  git push origin $${CURRENT_BRANCH}"
	echo ""
	if confirm "Push commits to origin/$${CURRENT_BRANCH}?"; then
	  git push origin "$${CURRENT_BRANCH}"
	  echo "  ✔  Commits pushed."
	else
	  skip_step
	fi

	# ── 5. Push tag to GitHub ──────────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 5 — Push tag to GitHub"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	REMOTE_TAG_EXISTS=false
	if git ls-remote --tags origin "refs/tags/$(TAG)" | grep -q "$(TAG)"; then
	  REMOTE_TAG_EXISTS=true
	fi

	if [[ "$$REMOTE_TAG_EXISTS" == "true" ]]; then
	  echo "  ⚠  Tag $(TAG) already exists on the remote."
	  echo "     Force-push will overwrite it — any GitHub release or CI"
	  echo "     already pointing at it will be re-anchored."
	  echo "  Will run:  git push origin --force $(TAG)"
	  echo ""
	  if confirm "Force-push tag $(TAG) to GitHub (overwrites remote)?"; then
	    git push origin --force $(TAG)
	    echo "  ✔  Tag force-pushed."
	  else
	    skip_step
	  fi
	else
	  echo "  Will run:  git push origin $(TAG)"
	  echo ""
	  if confirm "Push tag $(TAG) to GitHub?"; then
	    git push origin $(TAG)
	    echo "  ✔  Tag pushed."
	  else
	    skip_step
	  fi
	fi

	# ── 6. Create GitHub release ───────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 6 — Create GitHub release"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	if ! command -v gh &>/dev/null; then
	  echo "  ⚠  gh CLI not found — skipping GitHub release creation."
	  echo "     Install from https://cli.github.com/ and re-run."
	else
	  RELEASE_NOTES="## pynasonde $(FULL_VER)"$$'\n\n'"Precision ionospheric radio sounding tools."$$'\n\n'"### Installation"$$'\n\n'"```bash"$$'\n'"pip install pynasonde==$(FULL_VER)"$$'\n'"```"$$'\n\n'"### Changelog"$$'\n\n'"- Version bump to $(FULL_VER)"
	  echo "  Will run:"
	  echo "    gh release create $(TAG) --title \"pynasonde $(FULL_VER)\" --notes \"...\""
	  echo ""
	  if confirm "Create GitHub release $(TAG)?"; then
	    gh release create $(TAG) \
	        --title "pynasonde $(FULL_VER)" \
	        --notes "$$RELEASE_NOTES"
	    echo "  ✔  GitHub release created."
	    echo "     $$(gh release view $(TAG) --json url -q .url 2>/dev/null || true)"
	  else
	    skip_step
	  fi
	fi

	# ── 7. Build distribution packages ────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 7 — Build source distribution and wheel"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	if ! python -m build --version &>/dev/null; then
	  echo "  ⚠  python -m build not found in active environment."
	  echo "     Will install with:  pip install build"
	  echo ""
	  if confirm "Install 'build' now?"; then
	    pip install build
	    echo "  ✔  build installed."
	  else
	    echo "  ↳ Skipping build step."
	    SKIP_BUILD=true
	  fi
	fi

	if [[ "$${SKIP_BUILD:-false}" != "true" ]]; then
	  echo "  Will run:"
	  echo "    rm -rf dist/ build/ *.egg-info"
	  echo "    python -m build --sdist --wheel"
	  echo ""
	  echo "  Produces dist/pynasonde-$(FULL_VER).tar.gz"
	  echo "           dist/pynasonde-$(FULL_VER)-py3-none-any.whl"
	  echo ""
	  if confirm "Build distribution packages?"; then
	    rm -rf dist/ build/ pynasonde.egg-info/ *.egg-info/
	    python -m build --sdist --wheel
	    echo ""
	    echo "  ✔  Build complete. Artifacts:"
	    ls -lh dist/ | sed 's/^/    /'
	  else
	    skip_step
	  fi
	fi

	# ── 8. Upload to PyPI ──────────────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 8 — Upload to PyPI"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	if ! command -v twine &>/dev/null; then
	  echo "  ⚠  twine not found.  Will install with:  pip install --upgrade twine"
	  echo ""
	  if confirm "Install twine now?"; then
	    pip install --upgrade twine
	    echo "  ✔  twine installed."
	  else
	    echo "  ↳ Skipping upload."
	    SKIP_TWINE=true
	  fi
	fi

	if [[ "$${SKIP_TWINE:-false}" != "true" ]]; then
	  echo "  Running pre-upload metadata check:  twine check dist/*"
	  if ! python -m twine check dist/*; then
	    echo ""
	    echo "  ⚠  twine check failed — usually an outdated twine that does not"
	    echo "     understand Metadata-Version 2.4 (License-File field)."
	    echo "     Will run:  pip install --upgrade twine"
	    echo ""
	    if confirm "Upgrade twine and retry?"; then
	      pip install --upgrade twine
	      echo "  ✔  twine upgraded to $$(twine --version)"
	      if ! python -m twine check dist/*; then
	        echo "  ✗  twine check still failing — aborting upload."
	        SKIP_TWINE=true
	      else
	        echo "  ✔  twine check passed after upgrade."
	      fi
	    else
	      echo "  ↳ Skipping upload."
	      SKIP_TWINE=true
	    fi
	  else
	    echo "  ✔  twine check passed."
	  fi
	fi

	if [[ "$${SKIP_TWINE:-false}" != "true" ]]; then
	  echo ""
	  echo "  Will run:  python -m twine upload dist/*"
	  echo "  You will be prompted for your PyPI username / API token."
	  echo "  (Set TWINE_USERNAME and TWINE_PASSWORD env vars to skip the prompt.)"
	  echo ""
	  if confirm "Upload dist/* to PyPI?"; then
	    python -m twine upload dist/*
	    echo "  ✔  Uploaded to PyPI."
	    echo "     https://pypi.org/project/pynasonde/$(FULL_VER)/"
	  else
	    skip_step
	  fi
	fi

	# ── Done ───────────────────────────────────────────────────────────────
	echo ""
	echo "╔══════════════════════════════════════════════════════════════╗"
	echo "║  Release $(FULL_VER) complete.                             "
	echo "║  Tag    : $(TAG)                                            "
	echo "║  PyPI   : https://pypi.org/project/pynasonde/$(FULL_VER)/  "
	echo "║  GitHub : https://github.com/shibaji7/pynasonde/releases    "
	echo "╚══════════════════════════════════════════════════════════════╝"
	echo ""
