# =============================================================================
#  pynasonde — Release Makefile
#  Usage:  make release VERSION=1.2
#
#  Steps (each described and confirmed before execution):
#    1.  Validate VERSION format (MAJOR.MINOR)
#    2.  Check & update version in setup.py and pynasonde/__init__.py
#    3.  Commit any dirty working tree
#    4.  Create annotated git tag  vMAJOR.MINOR.0
#    5.  Push commits to GitHub
#    6.  Push tags to GitHub
#    7.  Create GitHub release (gh CLI)
#    8.  Build source dist + wheel
#    9.  Upload to PyPI (twine)
# =============================================================================

SHELL     := /bin/bash
.ONESHELL:
.PHONY: release check help

VERSION   ?= __UNSET__
FULL_VER  := $(VERSION).0
TAG       := v$(FULL_VER)

# --------------------------------------------------------------------------- #
help:
	@echo ""
	@echo "  pynasonde release targets"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make release VERSION=X.Y   Full release pipeline (interactive)"
	@echo "  make check  VERSION=X.Y   Consistency check only (no changes)"
	@echo "  ─────────────────────────────────────────────────────"
	@echo ""

# --------------------------------------------------------------------------- #
check:
	@set -euo pipefail
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo "  ERROR: No version provided."
	  echo "  Usage:  make check VERSION=1.2"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+$$'; then
	  echo "ERROR: VERSION must be MAJOR.MINOR (e.g. 1.2), got '$(VERSION)'"
	  exit 1
	fi
	echo ""
	echo "  Checking version consistency for $(FULL_VER) ..."
	echo ""
	SETUP_VER=$$(grep -oP '(?<=version=")[^"]+' setup.py || true)
	INIT_VER=$$(grep -oP '(?<=__version__ = ")[^"]+' pynasonde/__init__.py 2>/dev/null || echo "__missing__")
	TOML_DYN=$$(grep 'dynamic.*version' pyproject.toml || echo "(dynamic — ok)")
	echo "  setup.py       version = $${SETUP_VER:-<not found>}"
	echo "  __init__.py    version = $${INIT_VER}"
	echo "  pyproject.toml          $${TOML_DYN}"
	echo ""
	if [[ "$${SETUP_VER}" == "$(FULL_VER)" && \
	      ( "$${INIT_VER}" == "$(FULL_VER)" || "$${INIT_VER}" == "__missing__" ) ]]; then
	  echo "  All version references are consistent with $(FULL_VER)."
	else
	  echo "  Inconsistency detected — run  make release VERSION=$(VERSION)  to fix."
	fi
	echo ""

# --------------------------------------------------------------------------- #
release:
	@set -euo pipefail

	# ── 0. Validate input ─────────────────────────────────────────────────
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo "  ERROR: No version provided."
	  echo "  Usage:  make release VERSION=X.Y   (e.g.  make release VERSION=1.2)"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+$$'; then
	  echo "ERROR: VERSION must be MAJOR.MINOR (e.g. 1.2), got '$(VERSION)'"
	  exit 1
	fi

	echo ""
	echo "╔══════════════════════════════════════════════════════════════╗"
	echo "║           pynasonde  —  Release $(FULL_VER)                "
	echo "║  Git tag  : $(TAG)                                          "
	echo "║  Remote   : git@github.com:shibaji7/pynasonde.git           "
	echo "╚══════════════════════════════════════════════════════════════╝"
	echo ""

	# shared confirm helper
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
	INIT_VER=$$(grep -oP '(?<=__version__ = ")[^"]+' pynasonde/__init__.py 2>/dev/null || echo "__missing__")

	echo "  Current version references:"
	echo "    setup.py       version = \"$${SETUP_VER}\""
	echo "    __init__.py    __version__ = \"$${INIT_VER}\""
	echo "    pyproject.toml dynamic = [\"version\"]  (driven by git tag — ok)"
	echo ""
	echo "  Target version : $(FULL_VER)"
	echo ""

	NEED_SETUP=false
	NEED_INIT=false

	if [[ "$${SETUP_VER}" != "$(FULL_VER)" ]]; then
	  echo "  ⚠  setup.py has version=\"$${SETUP_VER}\", will be updated to \"$(FULL_VER)\""
	  NEED_SETUP=true
	fi
	if [[ "$${INIT_VER}" != "$(FULL_VER)" ]]; then
	  if [[ "$${INIT_VER}" == "__missing__" ]]; then
	    echo "  ⚠  pynasonde/__init__.py has no __version__, will add __version__ = \"$(FULL_VER)\""
	  else
	    echo "  ⚠  pynasonde/__init__.py has __version__=\"$${INIT_VER}\", will be updated to \"$(FULL_VER)\""
	  fi
	  NEED_INIT=true
	fi

	if [[ "$$NEED_SETUP" == "false" && "$$NEED_INIT" == "false" ]]; then
	  echo "  ✔  All version references already match $(FULL_VER). No changes needed."
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
	  if confirm "Commit all working-tree changes now?"; then
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

	if git tag | grep -q "^$(TAG)$$"; then
	  echo "  ⚠  Tag $(TAG) already exists."
	  echo "  Will run:  git tag -d $(TAG)  then re-create it."
	  echo ""
	  if confirm "Delete and re-create tag $(TAG)?"; then
	    git tag -d $(TAG)
	    echo "  ✔  Old tag deleted."
	  else
	    echo "  ↳ Keeping existing tag and skipping tag creation."
	    TAG_SKIPPED=true
	  fi
	fi

	if [[ "$${TAG_SKIPPED:-false}" != "true" ]]; then
	  echo "  Will run:"
	  echo "    git tag -a $(TAG) -m \"Release $(FULL_VER)\""
	  echo ""
	  echo "  Annotated tag message (edit if desired, then save & close editor):"
	  DEFAULT_MSG="Release $(FULL_VER)"$'\n\n'"Precision ionospheric radio sounding tools — pynasonde $(FULL_VER)"$'\n\nChanges in this release:'$'\n  - Bump version to $(FULL_VER)'
	  echo "    \"$$DEFAULT_MSG\""
	  echo ""
	  if confirm "Create annotated tag $(TAG) with the above message?"; then
	    git tag -a $(TAG) -m "$$DEFAULT_MSG"
	    echo "  ✔  Tag $(TAG) created."
	    echo "     $$(git tag -v $(TAG) 2>/dev/null || git show $(TAG) --no-patch --oneline)"
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

	echo "  Will run:  git push origin $(TAG)"
	echo ""
	if confirm "Push tag $(TAG) to GitHub?"; then
	  git push origin $(TAG)
	  echo "  ✔  Tag pushed."
	else
	  skip_step
	fi

	# ── 6. Create GitHub release ───────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 6 — Create GitHub release"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	if ! command -v gh &>/dev/null; then
	  echo "  ⚠  gh CLI not found — skipping GitHub release creation."
	  echo "     Install from https://cli.github.com/ and re-run to create the release."
	else
	  RELEASE_NOTES="## pynasonde $(FULL_VER)"$'\n\n'"Precision ionospheric radio sounding tools."$'\n\n'"### Installation"$'\n\n'"```bash"$'\n'"pip install pynasonde==$(FULL_VER)"$'\n'"```"$'\n\n'"### Changelog"$'\n\n'"- Version bump to $(FULL_VER)"
	  echo "  Will run:"
	  echo "    gh release create $(TAG) \\"
	  echo "        --title \"pynasonde $(FULL_VER)\" \\"
	  echo "        --notes \"<release notes above>\""
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

	echo "  Will run:"
	echo "    rm -rf dist/ build/ *.egg-info"
	echo "    python -m build --sdist --wheel"
	echo ""
	echo "  This produces dist/pynasonde-$(FULL_VER).tar.gz"
	echo "                  dist/pynasonde-$(FULL_VER)-py3-none-any.whl"
	echo ""
	if confirm "Build distribution packages now?"; then
	  rm -rf dist/ build/ pynasonde.egg-info/
	  python -m build --sdist --wheel
	  echo ""
	  echo "  ✔  Build complete. Artifacts:"
	  ls -lh dist/ | sed 's/^/    /'
	else
	  skip_step
	fi

	# ── 8. Upload to PyPI ──────────────────────────────────────────────────
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  STEP 8 — Upload to PyPI"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""

	if ! command -v twine &>/dev/null; then
	  echo "  ⚠  twine not found — install with:  pip install twine"
	  echo "     Then run:  python -m twine upload dist/*"
	else
	  echo "  Will run:"
	  echo "    python -m twine upload dist/*"
	  echo ""
	  echo "  You will be prompted for your PyPI username/token."
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
