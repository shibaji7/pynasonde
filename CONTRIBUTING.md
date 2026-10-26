# Contributing to pynasonde

Thank you for considering a contribution. pynasonde is scientific research
software used to process ionosonde data, so **correctness and backward
compatibility come before speed of feature delivery**.

- What is done, in progress and open for contribution: [ROADMAP.md](ROADMAP.md)
- What changed in each release: [CHANGELOG.md](CHANGELOG.md)
- Documentation (one build per release): <https://pynasonde.readthedocs.io/>

## Compatibility policy

pynasonde 2.0 is a *selective* update of 1.x. Until a major release says
otherwise:

- Every 1.x import path keeps working (for example
  `from pynasonde import SaoExtractor`, or `from pynasonde.vipir.analysis import
  AbsorptionAnalyzer`). Do not remove or rename public names; add the new name
  next to the old one and emit a `DeprecationWarning` from the old one.
- Public package `__init__` files resolve names lazily (PEP 562, see
  `pynasonde/__init__.py`). A new public class or function must be added to the
  `_LAZY` map **and** to `__all__` of its package, otherwise it is not importable
  from the package root.
- Heavy or optional dependencies are imported only inside the modules that need
  them, never from a package `__init__`, so that `import pynasonde` stays light.

## Branch strategy

GitHub flow, one long-lived branch:

```
main        <- always releasable; releases are tags (v1.3.0, v2.0.0, ...)
feature/*   <- short-lived, merged into main through a pull request
```

Branch names: `feature/<area>/<short-desc>` (also `fix/...`, `docs/...`), for
example `feature/digisonde/rsf-scaler-presets`. Every merge into `main` goes
through a pull request with CI green. Releases are cut by tagging `main`.

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/)-style
prefixes. They are recommended, not enforced by a hook:

```
feat(digisonde): add Viterbi tracer presets for the scaler
fix(vipir): use the file response status when downloading RIQ files
docs(install): document the per-module extras
test(scaler): cover the solar-regime helpers
refactor(riq): move parity helpers to utils
chore(ci): add Python 3.13 to the test matrix
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`. Keep the title to about 80 characters and put
the detail in the body. Mention the verification you ran (tests, docs build).

## Local environment setup

One-time. Assumes [conda](https://docs.conda.io/) is installed. The `Makefile`
runs every tool inside a conda environment named `pynasonde` (override with
`make test ENV=myenv`).

```bash
git clone https://github.com/shibaji7/pynasonde.git
cd pynasonde
make env            # creates/updates the conda env from env.yml
make install-dev    # editable install with every extra + test/format tools
make test           # run the suite once to confirm a green start
```

`make install-dev` is `pip install -e ".[dev]"`. Rerun it after you change
`pyproject.toml`. Python 3.11 and 3.12 are supported.

### Everyday commands

Run `make` (or `make help`) for the full, colourful list.

| Command | What it does |
|---------|--------------|
| `make test` | Full test suite with coverage |
| `make test-fast` | Tests without coverage, stop at the first failure |
| `make test-live` | Opt-in live tests that need credentials from `.env` (network, about a minute) |
| `make format` | `autoflake`, then `isort`, then `black`, in place |
| `make format-check` | Verify formatting without changing files (what CI runs) |
| `make typecheck` | `mypy` (informational, see [Type hints](#type-hints)) |
| `make check-docs` | Check that doc code blocks parse, API directives resolve and imports in docs and examples exist |
| `make docs` | Strict MkDocs build into `site/` |
| `make docs-serve` | Serve the docs locally with live reload |
| `make build-dist` | Clean, then build sdist and wheel |
| `make clean` | Remove build, cache and test artefacts |
| `make check VERSION=X.Y.Z` | Read-only version/tag consistency check |
| `make release VERSION=X.Y.Z` | Interactive release pipeline (maintainers) |

## Install extras

`pip install pynasonde` installs only a lean core (numpy, pandas, matplotlib,
loguru, toml, tqdm). Each module has an extra: `digisonde`, `digisonde-scaler`,
`digisonde-db`, `ocr`, `vipir`, `vipir-analysis`, `time`, `web`, `plots`, plus
`all`, `docs` and `dev`. The table of what each one adds is in the
[install guide](docs/user/install.md). If your change needs a new dependency,
add it to the extra of the module that uses it in `pyproject.toml` (never to the
core unless every module needs it) and update that table.

## Tests

Tests live in `tests/` (about 980 tests, pytest). Conventions:

- **Synthetic data first.** Most tests build small DataFrames or byte buffers in
  code. Small sample files that ship with the repo are in `examples/data/`.
  Skip (never fail) when an optional sample file is missing.
- **Optional dependencies.** A test that needs a package from an extra must guard
  it, so the test is skipped on a lean install:

  ```python
  pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
  ```

- **The CI "extras" job installs each extra on its own** and runs the tests that
  belong to it (see `.github/workflows/tests.yml`). The Digisonde and VIPIR
  families are separated with `--ignore` lists of test files. If you add a test
  file whose imports belong to one family, add it to the other family's
  `--ignore` list in the workflow, or the other job will fail loudly.
  `tests/extras_smoke.py <extra>` is the import check each job runs; add your new
  module there.
- **Examples are tested.** `tests/test_examples_smoke.py` runs examples that ship
  with data, inside a temporary directory, so they cannot overwrite tracked
  figures. New examples with bundled data should get a smoke test.
- **Live tests** (anything touching a real server, for example DIDBase) are marked
  `@pytest.mark.live` and are opt-in (`make test-live`). They must skip, not fail,
  when the driver, client library or network is missing.
- Matplotlib uses the `Agg` backend in tests (set in `tests/conftest.py`).

## Credentials and secrets

Never commit credentials, tokens or private data paths. Never commit `.claude/` (local
Claude Code worktrees): it is git-ignored, and CI fails if the tree contains a
submodule or worktree pointer.

- Local credentials go in a **git-ignored** `.env` file at the project root.
  Copy the template: `cp .env.example .env` and fill it in. `tests/conftest.py`
  and the DIDBase example load it; real environment variables take precedence.
- Examples must not contain personal absolute paths. Take data locations from
  environment variables with a safe default (for example the bundled files in
  `examples/data/`), as `examples/digisonde/rsf_direction_ionogram.py` does.

## Code style

- Format with `make format`. CI runs `isort --check-only` and `black --check`
  (black 24.x, isort 5.x with the black profile; configured in `pyproject.toml`).
- **Docstrings: Google style** (`Args:`, `Returns:`, `Raises:`, `Attributes:`),
  because mkdocstrings parses them with the `google` handler. Pitfall: names such
  as `Classes:`, `Methods:`, `Attributes:` and `Returns:` at the start of a line
  are treated as sections. Use them only with the Google layout (indented
  `name: description` entries); a free-text list under a `Classes:` header breaks
  the docs build.
- One entry per parameter. Do not group names (`min_range, max_range:`); griffe
  warns that the name is not in the signature.
- Public functions and classes need a docstring. Do not paste numpy-style
  (`Parameters` with a dashed underline) sections into new code.

### Type hints

New public functions should be annotated. Typing is partial in older modules and
`mypy` currently reports a baseline of several hundred issues, so `make typecheck`
is informational and no `py.typed` marker is shipped yet (see the roadmap). Do not
add a new `mypy` error in code you touch.

## Documentation

Docs are MkDocs (Material) with mkdocstrings and live in `docs/`; the navigation
is in `mkdocs.yml`. For a new module:

1. Add an API page under `docs/dev/<package>/` (copy a neighbouring page) and list
   it in `mkdocs.yml` and `docs/dev/api_index.md`.
2. For a new example, add a page under `docs/examples/` that explains the call
   flow, shows the figure, and says which extra to install and how to run it.
3. `make docs` must pass (it builds with `--strict`), and so must `make check-docs`, which
   verifies that every `:::` API directive and `members:` entry resolves, every Python code
   block parses, and every `pynasonde` import in the docs and `examples/` exists. CI runs it too.

Documentation is published per release on Read the Docs: each tag builds from its
own copy of `.readthedocs.yaml`, so older versions keep their original docs.

The version dropdown in the header comes from `docs/javascripts/version-selector.js`,
which reads the active versions from Read the Docs. Only versions that are **active
in the Read the Docs dashboard** are listed, and releases built before the script
existed (1.x) rely on Read the Docs' own floating flyout instead. Its logic has a
Node test: `node tests/js/version_selector.test.js` (also run in CI).

## Pull request checklist

Before opening a pull request:

- [ ] Branch is up to date with `main` (rebase).
- [ ] `make format-check` passes.
- [ ] `make test` passes (and `make test-live` if you touched a live integration).
- [ ] `make docs` and `make check-docs` pass if you touched docs, examples or docstrings.
- [ ] New public names are in `_LAZY` and `__all__`; no heavy import in an `__init__`.
- [ ] Tests guard optional dependencies; new test files are in the right CI family.
- [ ] No secrets or personal paths; no large data files (keep sample data small).
- [ ] New public functions and classes have Google-style docstrings (and type hints).
- [ ] `CHANGELOG.md` entry added under the next version.
- [ ] Commit messages follow the conventions above.

## Versioning and releases

[Semantic Versioning](https://semver.org/). The version comes **only from the git
tag** through setuptools-scm (`pynasonde/_version.py` is generated and
git-ignored); there is no version literal to edit.

| Change | Bump |
|--------|------|
| Backward-incompatible API change | MAJOR |
| New feature, new module, new extra | MINOR |
| Bug fix, docs, internal change | PATCH |

Release procedure (maintainers):

1. Move the `CHANGELOG.md` entries to the new version heading and update the date.
2. Update the version in `CITATION.cff` and `codemeta.json` (still manual).
3. `make check VERSION=X.Y.Z`, then `make release VERSION=X.Y.Z`. The pipeline
   asks before each step: version check, commit, annotated tag `vX.Y.Z`, push
   commits, push the tag, create the GitHub release, build sdist and wheel.
4. In the Read the Docs dashboard, make sure the new tag is **active** so it
   appears in the version menu, and that `stable` points to it.
5. Publishing the built files to PyPI is a separate maintainer step.

## Reporting issues

Open an [issue](https://github.com/shibaji7/pynasonde/issues) with the pynasonde
version (`python -c "import pynasonde; print(pynasonde.__version__)"`), the
instrument and file format, a minimal example (a small file or synthetic data),
and the full traceback.

## License

By contributing you agree that your contribution is licensed under the
[MIT License](LICENSE) that covers the project.
