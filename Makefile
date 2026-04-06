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
#    make vega-sync                Copy NN-inversion module to VEGA HPC via scp
#    make tutorials                Build both tutorial PDFs (keeps .tex + .pdf only)
#    make tutorials-grad           Build only the graduate student tutorial
#    make tutorials-researcher     Build only the researcher reference
#	 make tutorials-nn        	   Build only the NN Invrsion tutorial / Paper
#    make tutorials-clean          Remove all LaTeX auxiliary files from tutorials/
#
#  Release targets:
#    make check   VERSION=X.Y      Consistency check only (no changes)
#    make release VERSION=X.Y      Full interactive release pipeline
#
#  Each release step describes what it will do and asks Y/N before running.
# =============================================================================

SHELL        := /bin/bash
.DEFAULT_GOAL := help
.ONESHELL:
.PHONY: help clean format install-dev test build-dist sync check release vega-sync

# VEGA HPC settings — override on the command line if needed
VEGA_USER    ?= chakras4
VEGA_HOST    ?= vegaln1.erau.edu
VEGA_ROOT    ?= ~/Research/CodeBase/pynasonde

# --------------------------------------------------------------------------- #
#  vega-sync — copy only the NN-inversion module + pyproject.toml to VEGA
#
#  Usage:  make vega-sync
#          make vega-sync VEGA_USER=myuser VEGA_HOST=vega.example.edu
# --------------------------------------------------------------------------- #
vega-sync:
	@set -euo pipefail
	echo ""
	echo "  Syncing NN-inversion module to $(VEGA_USER)@$(VEGA_HOST):$(VEGA_ROOT)/"
	echo ""
	echo "  Files to transfer:"
	echo "    pynasonde/vipir/analysis/nn_inversion/  (full module)"
	echo "    OMNI/  (full OMNI dataset, if not already present on VEGA)"
	echo ""
	ssh $(VEGA_USER)@$(VEGA_HOST) "mkdir -p $(VEGA_ROOT)/pynasonde/vipir/analysis"
	scp -r pynasonde/vipir/analysis/nn_inversion \
	    $(VEGA_USER)@$(VEGA_HOST):$(VEGA_ROOT)/pynasonde/vipir/analysis/
	scp -r /home/chakras4/OMNI/ $(VEGA_USER)@$(VEGA_HOST):~/
	echo ""
	echo "  ✔  Sync complete."
	echo "     Remote: $(VEGA_USER)@$(VEGA_HOST):$(VEGA_ROOT)"
	echo ""

VERSION      ?= __UNSET__
FULL_VER     := $(VERSION)

LATEX        := pdflatex -interaction=nonstopmode
LATEX_AUXEXT := aux log out toc nav snm vrb bbl blg lof lot fls fdb_latexmk synctex.gz
TUTORIAL_GRAD    := tutorials/grad_student
TUTORIAL_RES     := tutorials/researcher
TUTORIAL_METHODS := tutorials/methods_ppt
TUTORIAL_NN	  := tutorials/nn_inv
TAG       := v$(FULL_VER)
MSG       ?=

# Convenience phony aliases — only these are phony; the PDF targets are real files
.PHONY: tutorials tutorials-grad tutorials-researcher tutorials-methods tutorials-nn tutorials-clean

# --------------------------------------------------------------------------- #
#  tutorials — build PDFs only when the .tex source is newer than the PDF
#
#  make tutorials            → rebuild whichever PDF is out of date
#  make tutorials-grad       → alias for the grad PDF file target
#  make tutorials-researcher → alias for the researcher PDF file target
#  make tutorials-nn		 → alias for the NN PDF file target
#  make tutorials-clean      → remove LaTeX aux files (keep .tex + .pdf)
# --------------------------------------------------------------------------- #

# Real file targets — make skips the recipe if PDF is newer than .tex
WORKSHOP_PDF   := $(TUTORIAL_GRAD)/pynasonde_workshop.pdf
METHODS_PDF    := $(TUTORIAL_RES)/pynasonde_methods.pdf
METHODS_PPT_PDF := $(TUTORIAL_METHODS)/pynasonde_methods_ppt.pdf
WORKSHOP_TEX   := $(TUTORIAL_GRAD)/pynasonde_workshop.tex
METHODS_TEX    := $(TUTORIAL_RES)/pynasonde_methods.tex
METHODS_PPT_TEX := $(TUTORIAL_METHODS)/pynasonde_methods_ppt.tex
METHODS_NN_PDF := $(TUTORIAL_NN)/whitepaper_nn_inversion.pdf
METHODS_NN_TEX := $(TUTORIAL_NN)/whitepaper_nn_inversion.tex

tutorials: $(WORKSHOP_PDF) $(METHODS_PDF) $(METHODS_PPT_PDF)
	@echo "  Tutorials up to date: $(WORKSHOP_PDF)  $(METHODS_PDF)  $(METHODS_PPT_PDF)"

tutorials-grad: $(WORKSHOP_PDF)
	@echo "  Tutorial up to date: $(WORKSHOP_PDF)"

tutorials-researcher: $(METHODS_PDF)
	@echo "  Tutorial up to date: $(METHODS_PDF)"

tutorials-methods: $(METHODS_PPT_PDF)
	@echo "  Tutorial up to date: $(METHODS_PPT_PDF)"

tutorials-nn: $(METHODS_NN_PDF)
	@echo "  Tutorial up to date: $(METHODS_NN_PDF)"

$(WORKSHOP_PDF): $(WORKSHOP_TEX)
	@echo ""
	@echo "  [tutorials] $< is newer than $@ — rebuilding..."
	cd $(TUTORIAL_GRAD) && $(LATEX) pynasonde_workshop.tex && $(LATEX) pynasonde_workshop.tex
	cd $(CURDIR)/$(TUTORIAL_GRAD) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_PDF): $(METHODS_TEX)
	@echo ""
	@echo "  [tutorials] $< is newer than $@ — rebuilding..."
	cd $(TUTORIAL_RES) && $(LATEX) pynasonde_methods.tex && $(LATEX) pynasonde_methods.tex
	cd $(CURDIR)/$(TUTORIAL_RES) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_PPT_PDF): $(METHODS_PPT_TEX)
	@echo ""
	@echo "  [tutorials] $< is newer than $@ — rebuilding..."
	cd $(TUTORIAL_METHODS) && $(LATEX) pynasonde_methods_ppt.tex && $(LATEX) pynasonde_methods_ppt.tex
	cd $(CURDIR)/$(TUTORIAL_METHODS) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_NN_PDF): $(METHODS_NN_TEX)
	@echo ""
	@echo "  [tutorials] $< is newer than $@ — rebuilding..."
	cd $(TUTORIAL_NN) && $(LATEX) whitepaper_nn_inversion.tex && $(LATEX) whitepaper_nn_inversion.tex
	cd $(CURDIR)/$(TUTORIAL_NN) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

tutorials-clean:
	@cd $(CURDIR)/$(TUTORIAL_GRAD)    && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_RES)     && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_METHODS) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_NN)      && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  LaTeX auxiliary files removed (kept .tex and .pdf)."

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
	@echo "  make vega-sync              Copy NN-inversion module + pyproject.toml to VEGA"
	@echo "  make tutorials              Build all three tutorial PDFs"
	@echo "  make tutorials-grad         Build only the graduate student workshop"
	@echo "  make tutorials-researcher   Build only the researcher reference"
	@echo "  make tutorials-methods      Build only the methods overview PPT"
	@echo "  make tutorials-clean        Remove LaTeX aux files (keep .tex + .pdf)"
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
