# =============================================================================
#  pynasonde — Project Makefile
#
#  Run `make` (or `make help`) for the list of targets.
#  Dev tools run inside the conda env named by ENV (default: pynasonde):
#      make env            create / update that env      make test
#      make install-dev    editable install + dev extras  make format
#  Override the env or disable colours:  make test ENV=myenv   NO_COLOR=1 make test
# =============================================================================

SHELL         := /bin/bash
.DEFAULT_GOAL := help
.ONESHELL:
.PHONY: help env install-dev test test-fast format format-check docs docs-serve \
        build-dist clean check release

ENV      ?= pynasonde
# `conda run` keeps the env isolated without needing `conda activate` in a Makefile
RUN      := conda run --no-capture-output -n $(ENV)
VERSION  ?= __UNSET__
FULL_VER := $(VERSION)
TAG      := v$(FULL_VER)

# --- colours (disabled when NO_COLOR is set) -------------------------------- #
ifeq ($(NO_COLOR),)
  ESC    := $(shell printf '\033')
  RESET  := $(ESC)[0m
  BOLD   := $(ESC)[1m
  DIM    := $(ESC)[2m
  RED    := $(ESC)[31m
  GREEN  := $(ESC)[32m
  YELLOW := $(ESC)[33m
  BLUE   := $(ESC)[34m
  MAGENTA:= $(ESC)[35m
  CYAN   := $(ESC)[36m
endif

# --------------------------------------------------------------------------- #
#  help — generated from the `## description` after each target
# --------------------------------------------------------------------------- #
help: ## Show this help
	@printf "\n$(BOLD)$(MAGENTA)  pynasonde$(RESET) $(DIM)— make targets (conda env: $(ENV))$(RESET)\n"
	printf "$(DIM)  ───────────────────────────────────────────────────────────────$(RESET)\n"
	awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	printf "$(DIM)  ───────────────────────────────────────────────────────────────$(RESET)\n"
	printf "  $(YELLOW)Release$(RESET): make check VERSION=X.Y.Z  ·  make release VERSION=X.Y.Z\n\n"

# --------------------------------------------------------------------------- #
#  environment
# --------------------------------------------------------------------------- #
env: ## Create (or update) the conda dev env from env.yml
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ Setting up conda env '$(ENV)' from env.yml ...$(RESET)\n\n"
	if conda env list | awk '{print $$1}' | grep -qx "$(ENV)"; then
	  conda env update -n $(ENV) -f env.yml --prune
	else
	  conda env create -n $(ENV) -f env.yml
	fi
	printf "\n$(GREEN)  ✔  Env '$(ENV)' ready.$(RESET)  Next: make test\n\n"

install-dev: ## Editable install with dev extras into the conda env
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ pip install -e .[dev] in env '$(ENV)' ...$(RESET)\n\n"
	$(RUN) python -m pip install -e ".[dev]"
	printf "\n$(GREEN)  ✔  Done.$(RESET)\n\n"

# --------------------------------------------------------------------------- #
#  quality
# --------------------------------------------------------------------------- #
test: ## Run the full test suite with coverage
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ pytest (env '$(ENV)') ...$(RESET)\n\n"
	mkdir -p /tmp/mplconfig
	NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/mplconfig $(RUN) python -m pytest
	printf "\n$(GREEN)  ✔  Tests finished.$(RESET)\n\n"

test-fast: ## Run tests without coverage, stop at first failure
	@set -euo pipefail
	mkdir -p /tmp/mplconfig
	NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/mplconfig $(RUN) python -m pytest -x -q -o addopts=""

format: ## Format in place: autoflake -> isort -> black
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ Formatting (autoflake → isort → black) ...$(RESET)\n\n"
	$(RUN) autoflake --in-place --remove-all-unused-imports --exclude=__init__.py \
	    --ignore-init-module-imports -r pynasonde tests examples
	$(RUN) isort pynasonde tests examples
	$(RUN) black pynasonde tests examples
	printf "\n$(GREEN)  ✔  Format complete.$(RESET)\n\n"

format-check: ## Check formatting without changing files
	@set -euo pipefail
	$(RUN) isort --check-only pynasonde tests examples
	$(RUN) black --check pynasonde tests examples
	printf "$(GREEN)  ✔  Formatting OK.$(RESET)\n"

# --------------------------------------------------------------------------- #
#  docs
# --------------------------------------------------------------------------- #
docs: ## Build the MkDocs site (strict) into site/
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ mkdocs build ...$(RESET)\n\n"
	$(RUN) mkdocs build --strict
	printf "\n$(GREEN)  ✔  Site built in site/$(RESET)\n\n"

docs-serve: ## Serve the docs locally with live reload
	@$(RUN) mkdocs serve

# --------------------------------------------------------------------------- #
#  build / clean
# --------------------------------------------------------------------------- #
build-dist: clean ## Clean, then build sdist + wheel into dist/
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ python -m build ...$(RESET)\n\n"
	$(RUN) python -m build --sdist --wheel
	printf "\n$(GREEN)  ✔  Build complete. Artifacts:$(RESET)\n"
	ls -lh dist/ | sed 's/^/    /'
	echo ""

clean: ## Remove build, cache and test artefacts
	@set -euo pipefail
	printf "\n$(CYAN)  ▶ Cleaning build artefacts, caches, and test outputs ...$(RESET)\n"
	rm -rf dist/ build/ site_build/ site/ .eggs/ pynasonde.egg-info/ *.egg-info/
	rm -rf junit.xml .coverage coverage.xml .pytest_cache/
	find . -type d -name '__pycache__'        -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -not -path './.git/*' -delete 2>/dev/null || true
	printf "$(GREEN)  ✔  Clean complete.$(RESET)\n\n"

# --------------------------------------------------------------------------- #
#  tutorials (LaTeX → PDF) — unchanged apart from colours
# --------------------------------------------------------------------------- #
LATEX        := pdflatex -interaction=nonstopmode
LATEX_AUXEXT := aux log out toc nav snm vrb bbl blg lof lot fls fdb_latexmk synctex.gz
TUTORIAL_GRAD    := tutorials/grad_student
TUTORIAL_RES     := tutorials/researcher
TUTORIAL_METHODS := tutorials/methods_ppt
TUTORIAL_NN	  := tutorials/nn_inv

# Convenience phony aliases — only these are phony; the PDF targets are real files
.PHONY: tutorials tutorials-grad tutorials-researcher tutorials-methods tutorials-nn tutorials-nn-ppt tutorials-clean

# --------------------------------------------------------------------------- #
#  tutorials — build PDFs only when the .tex source is newer than the PDF
#
#  make tutorials            → rebuild whichever PDF is out of date
#  make tutorials-grad       → alias for the grad PDF file target
#  make tutorials-researcher → alias for the researcher PDF file target
#  make tutorials-nn          → alias for the NN whitepaper PDF target
#  make tutorials-nn-ppt      → alias for the NN show-and-tell PPT PDF target
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
NN_PPT_PDF     := $(TUTORIAL_NN)/nn_inversion_ppt.pdf
NN_PPT_TEX     := $(TUTORIAL_NN)/nn_inversion_ppt.tex

tutorials: $(WORKSHOP_PDF) $(METHODS_PDF) $(METHODS_PPT_PDF) ## Build the tutorial PDFs (only if the .tex changed)
	@echo -e "$(GREEN)  Tutorials up to date: $(WORKSHOP_PDF)  $(METHODS_PDF)  $(METHODS_PPT_PDF)$(RESET)"

tutorials-grad: $(WORKSHOP_PDF) ## Build only the graduate student workshop
	@echo -e "$(GREEN)  Tutorial up to date: $(WORKSHOP_PDF)$(RESET)"

tutorials-researcher: $(METHODS_PDF) ## Build only the researcher reference
	@echo -e "$(GREEN)  Tutorial up to date: $(METHODS_PDF)$(RESET)"

tutorials-methods: $(METHODS_PPT_PDF) ## Build only the methods overview slides
	@echo -e "$(GREEN)  Tutorial up to date: $(METHODS_PPT_PDF)$(RESET)"

tutorials-nn: $(METHODS_NN_PDF) ## Build only the NN inversion whitepaper
	@echo -e "$(GREEN)  Tutorial up to date: $(METHODS_NN_PDF)$(RESET)"

tutorials-nn-ppt: $(NN_PPT_PDF) ## Build only the NN inversion slides
	@echo -e "$(GREEN)  Tutorial up to date: $(NN_PPT_PDF)$(RESET)"

$(WORKSHOP_PDF): $(WORKSHOP_TEX)
	@echo ""
	@echo -e "$(CYAN)  [tutorials] $< is newer than $@ — rebuilding...$(RESET)"
	cd $(TUTORIAL_GRAD) && $(LATEX) pynasonde_workshop.tex && $(LATEX) pynasonde_workshop.tex
	cd $(CURDIR)/$(TUTORIAL_GRAD) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_PDF): $(METHODS_TEX)
	@echo ""
	@echo -e "$(CYAN)  [tutorials] $< is newer than $@ — rebuilding...$(RESET)"
	cd $(TUTORIAL_RES) && $(LATEX) pynasonde_methods.tex && $(LATEX) pynasonde_methods.tex
	cd $(CURDIR)/$(TUTORIAL_RES) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_PPT_PDF): $(METHODS_PPT_TEX)
	@echo ""
	@echo -e "$(CYAN)  [tutorials] $< is newer than $@ — rebuilding...$(RESET)"
	cd $(TUTORIAL_METHODS) && $(LATEX) pynasonde_methods_ppt.tex && $(LATEX) pynasonde_methods_ppt.tex
	cd $(CURDIR)/$(TUTORIAL_METHODS) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(METHODS_NN_PDF): $(METHODS_NN_TEX)
	@echo ""
	@echo -e "$(CYAN)  [tutorials] $< is newer than $@ — rebuilding...$(RESET)"
	cd $(TUTORIAL_NN) && $(LATEX) whitepaper_nn_inversion.tex && $(LATEX) whitepaper_nn_inversion.tex
	cd $(CURDIR)/$(TUTORIAL_NN) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

$(NN_PPT_PDF): $(NN_PPT_TEX)
	@echo ""
	@echo -e "$(CYAN)  [tutorials] $< is newer than $@ — rebuilding...$(RESET)"
	cd $(TUTORIAL_NN) && $(LATEX) nn_inversion_ppt.tex && $(LATEX) nn_inversion_ppt.tex
	cd $(CURDIR)/$(TUTORIAL_NN) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo "  Output : $@"
	@echo ""

tutorials-clean: ## Remove LaTeX aux files (keep .tex + .pdf)
	@cd $(CURDIR)/$(TUTORIAL_GRAD)    && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_RES)     && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_METHODS) && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@cd $(CURDIR)/$(TUTORIAL_NN)      && ls | grep -vE "\.(tex|pdf)$$" | xargs -r rm -f
	@echo -e "$(GREEN)  ✔  LaTeX auxiliary files removed (kept .tex and .pdf).$(RESET)"

# --------------------------------------------------------------------------- #
#  check — read-only version consistency check
# --------------------------------------------------------------------------- #
check: ## Version/tag consistency check (VERSION=X.Y.Z, read-only)
	@set -euo pipefail
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo -e "$(RED)  ERROR: No version provided.  Usage:  make check VERSION=1.1.0$(RESET)"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then
	  echo -e "$(RED)ERROR: VERSION must be MAJOR.MINOR.PATCH (e.g. 1.1.0), got '$(VERSION)'$(RESET)"
	  exit 1
	fi
	echo ""
	echo "  Checking version consistency for $(FULL_VER) ..."
	echo ""
	PKG_VER=$$(python -c "import pynasonde;print(pynasonde.__version__)" 2>/dev/null || echo "__missing__")
	TAG_VER=$$(git describe --tags --abbrev=0 2>/dev/null || echo "<no tag>")
	echo "  git tag (latest)      = \"$${TAG_VER}\""
	echo "  pynasonde.__version__ = \"$${PKG_VER}\"  (generated by setuptools-scm)"
	echo "  pyproject.toml          dynamic = [\"version\"] — single source is the git tag $(TAG)"
	echo ""
	if git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null; then
	  echo -e "$(GREEN)  ✔  Tag $(TAG) exists.$(RESET)"
	else
	  echo -e "$(YELLOW)  ℹ  Tag $(TAG) does not exist yet — make release VERSION=$(VERSION) will create it.$(RESET)"
	fi
	echo ""

# --------------------------------------------------------------------------- #
#  release — full interactive release pipeline
# --------------------------------------------------------------------------- #
release: ## Interactive release pipeline (VERSION=X.Y.Z)
	@set -euo pipefail

	# ── 0. Validate input ─────────────────────────────────────────────────
	if [[ "$(VERSION)" == "__UNSET__" ]]; then
	  echo ""
	  echo -e "$(RED)  ERROR: No version provided.$(RESET)"
	  echo "  Usage:  make release VERSION=X.Y.Z   (e.g.  make release VERSION=1.1.0)"
	  echo ""
	  exit 1
	fi
	if ! echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then
	  echo -e "$(RED)ERROR: VERSION must be MAJOR.MINOR.PATCH (e.g. 1.1.0), got '$(VERSION)'$(RESET)"
	  exit 1
	fi

	echo ""
	echo -e "$(BOLD)$(BLUE)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	echo -e "$(BOLD)$(BLUE)║           pynasonde  —  Release $(FULL_VER)                $(RESET)"
	echo -e "$(BOLD)$(BLUE)║  Git tag  : $(TAG)                                          $(RESET)"
	echo -e "$(BOLD)$(BLUE)║  Remote   : git@github.com:shibaji7/pynasonde.git           $(RESET)"
	echo -e "$(BOLD)$(BLUE)╚══════════════════════════════════════════════════════════════╝$(RESET)"
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
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 1 — Version (single-sourced from git tag)$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	echo "  Version is single-sourced from the git tag by setuptools-scm."
	echo "  No file edits needed: tagging $(TAG) in a later step sets $(FULL_VER)."
	echo ""

	# ── 2. Commit working tree ─────────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 2 — Commit working tree changes$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	GIT_STATUS=$$(git status --short)
	if [[ -z "$$GIT_STATUS" ]]; then
	  echo -e "$(GREEN)  ✔  Working tree is clean — nothing to commit.$(RESET)"
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
	    echo -e "$(GREEN)  ✔  Committed.$(RESET)"
	  else
	    skip_step
	  fi
	fi

	# ── 3. Create annotated git tag ────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 3 — Create annotated git tag$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	TAG_SKIPPED=false
	if git tag | grep -q "^$(TAG)$$"; then
	  echo -e "$(YELLOW)  ⚠  Tag $(TAG) already exists locally.$(RESET)"
	  echo "  Will run:  git tag -d $(TAG)  then re-create it."
	  echo ""
	  if confirm "Delete and re-create tag $(TAG)?"; then
	    git tag -d $(TAG)
	    echo -e "$(GREEN)  ✔  Old local tag deleted.$(RESET)"
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
	    echo -e "$(GREEN)  ✔  Tag $(TAG) created.$(RESET)"
	  else
	    skip_step
	  fi
	fi

	# ── 4. Push commits to GitHub ──────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 4 — Push commits to GitHub$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD)
	AHEAD=$$(git rev-list origin/$${CURRENT_BRANCH}..HEAD --count 2>/dev/null || echo "?")
	echo "  Branch  : $${CURRENT_BRANCH}"
	echo "  Commits ahead of origin : $${AHEAD}"
	echo "  Will run:  git push origin $${CURRENT_BRANCH}"
	echo ""
	if confirm "Push commits to origin/$${CURRENT_BRANCH}?"; then
	  git push origin "$${CURRENT_BRANCH}"
	  echo -e "$(GREEN)  ✔  Commits pushed.$(RESET)"
	else
	  skip_step
	fi

	# ── 5. Push tag to GitHub ──────────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 5 — Push tag to GitHub$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	REMOTE_TAG_EXISTS=false
	if git ls-remote --tags origin "refs/tags/$(TAG)" | grep -q "$(TAG)"; then
	  REMOTE_TAG_EXISTS=true
	fi

	if [[ "$$REMOTE_TAG_EXISTS" == "true" ]]; then
	  echo -e "$(YELLOW)  ⚠  Tag $(TAG) already exists on the remote.$(RESET)"
	  echo "     Force-push will overwrite it — any GitHub release or CI"
	  echo "     already pointing at it will be re-anchored."
	  echo "  Will run:  git push origin --force $(TAG)"
	  echo ""
	  if confirm "Force-push tag $(TAG) to GitHub (overwrites remote)?"; then
	    git push origin --force $(TAG)
	    echo -e "$(GREEN)  ✔  Tag force-pushed.$(RESET)"
	  else
	    skip_step
	  fi
	else
	  echo "  Will run:  git push origin $(TAG)"
	  echo ""
	  if confirm "Push tag $(TAG) to GitHub?"; then
	    git push origin $(TAG)
	    echo -e "$(GREEN)  ✔  Tag pushed.$(RESET)"
	  else
	    skip_step
	  fi
	fi

	# ── 6. Create GitHub release ───────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 6 — Create GitHub release$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	if ! command -v gh &>/dev/null; then
	  echo -e "$(YELLOW)  ⚠  gh CLI not found — skipping GitHub release creation.$(RESET)"
	  echo "     Install from https://cli.github.com/ and re-run."
	else
	  RELEASE_NOTES_FILE="issue/3/release_notes_v$(FULL_VER).md"
	  echo "  Will run:"
	  echo "    gh release create $(TAG) --title \"pynasonde $(FULL_VER)\" --notes-file $$RELEASE_NOTES_FILE"
	  echo ""
	  if [[ ! -f "$$RELEASE_NOTES_FILE" ]]; then
	    echo -e "$(YELLOW)  ⚠  Release notes file not found: $$RELEASE_NOTES_FILE$(RESET)"
	    echo "     Create this file and re-run release."
	    skip_step
	  elif confirm "Create GitHub release $(TAG)?"; then
	    gh release create $(TAG) \
	        --title "pynasonde $(FULL_VER)" \
	        --notes-file "$$RELEASE_NOTES_FILE"
	    echo -e "$(GREEN)  ✔  GitHub release created.$(RESET)"
	    echo "     $$(gh release view $(TAG) --json url -q .url 2>/dev/null || true)"
	  else
	    skip_step
	  fi
	fi

	# ── 7. Build distribution packages ────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 7 — Build source distribution and wheel$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	if ! python -m build --version &>/dev/null; then
	  echo -e "$(YELLOW)  ⚠  python -m build not found in active environment.$(RESET)"
	  echo "     Will install with:  pip install build"
	  echo ""
	  if confirm "Install 'build' now?"; then
	    pip install build
	    echo -e "$(GREEN)  ✔  build installed.$(RESET)"
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
	    echo -e "$(GREEN)  ✔  Build complete. Artifacts:$(RESET)"
	    ls -lh dist/ | sed 's/^/    /'
	  else
	    skip_step
	  fi
	fi

	# ── 8. Upload to PyPI ──────────────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo -e "$(BOLD)$(CYAN)  STEP 8 — Upload to PyPI$(RESET)"
	echo -e "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	echo ""

	if ! command -v twine &>/dev/null; then
	  echo -e "$(YELLOW)  ⚠  twine not found.  Will install with:  pip install --upgrade twine$(RESET)"
	  echo ""
	  if confirm "Install twine now?"; then
	    pip install --upgrade twine
	    echo -e "$(GREEN)  ✔  twine installed.$(RESET)"
	  else
	    echo "  ↳ Skipping upload."
	    SKIP_TWINE=true
	  fi
	fi

	if [[ "$${SKIP_TWINE:-false}" != "true" ]]; then
	  echo "  Running pre-upload metadata check:  twine check dist/*"
	  if ! python -m twine check dist/*; then
	    echo ""
	    echo -e "$(YELLOW)  ⚠  twine check failed — usually an outdated twine that does not$(RESET)"
	    echo "     understand Metadata-Version 2.4 (License-File field)."
	    echo "     Will run:  pip install --upgrade twine"
	    echo ""
	    if confirm "Upgrade twine and retry?"; then
	      pip install --upgrade twine
	      echo -e "$(GREEN)  ✔  twine upgraded to $$(twine --version)$(RESET)"
	      if ! python -m twine check dist/*; then
	        echo "  ✗  twine check still failing — aborting upload."
	        SKIP_TWINE=true
	      else
	        echo -e "$(GREEN)  ✔  twine check passed after upgrade.$(RESET)"
	      fi
	    else
	      echo "  ↳ Skipping upload."
	      SKIP_TWINE=true
	    fi
	  else
	    echo -e "$(GREEN)  ✔  twine check passed.$(RESET)"
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
	    echo -e "$(GREEN)  ✔  Uploaded to PyPI.$(RESET)"
	    echo "     https://pypi.org/project/pynasonde/$(FULL_VER)/"
	  else
	    skip_step
	  fi
	fi

	# ── Done ───────────────────────────────────────────────────────────────
	echo ""
	echo -e "$(BOLD)$(BLUE)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	echo -e "$(BOLD)$(BLUE)║  Release $(FULL_VER) complete.                             $(RESET)"
	echo -e "$(BOLD)$(BLUE)║  Tag    : $(TAG)                                            $(RESET)"
	echo -e "$(BOLD)$(BLUE)║  PyPI   : https://pypi.org/project/pynasonde/$(FULL_VER)/  $(RESET)"
	echo -e "$(BOLD)$(BLUE)║  GitHub : https://github.com/shibaji7/pynasonde/releases    $(RESET)"
	echo -e "$(BOLD)$(BLUE)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	echo ""
