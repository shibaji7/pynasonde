"""Check the documentation against the installed pynasonde package.

Usage: python tools/check_docs.py [SRC_DIR]      (default: the current directory)

Needs every extra installed (``pip install -e ".[dev]"``) so all modules import.
Checks, and exits non-zero if any fails:

* every ``::: dotted.path`` API directive in ``docs/**/*.md`` resolves, and so does
  every name in its ``members:`` list;
* every ```python code block in ``docs/**/*.md`` parses;
* every ``pynasonde`` import in those blocks, and in ``examples/**/*.py``, exists.

The same check can be pointed at an older release (extract it with ``git archive``,
install it, then run this script on the extract) to audit that version's docs.
"""

import ast
import importlib
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
warnings.filterwarnings("ignore")

FENCE = re.compile(
    r"^(?P<ind>[ \t]*)```(?:python|py)\b[^\n]*\n(?P<body>.*?)^(?P=ind)```",
    re.S | re.M,
)
DIRECTIVE = re.compile(r"^\s*::: ([\w.]+)\s*$")

_modules: dict = {}


def resolve(dotted: str):
    """Import the longest module prefix of ``dotted`` and getattr the rest."""
    parts = dotted.split(".")
    last_error = "cannot import"
    for i in range(len(parts), 0, -1):
        name = ".".join(parts[:i])
        try:
            obj = _modules.get(name) or importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report whatever the import raised
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        _modules[name] = obj
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj, None
        except AttributeError as exc:
            return None, str(exc)
    return None, last_error


def check_imports(tree: ast.AST, where: str, problems: list) -> int:
    """Verify every pynasonde import in ``tree``; return how many were checked."""
    checked = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.split(".")[0] == "pynasonde"
        ):
            checked += 1
            mod, err = resolve(node.module)
            if mod is None:
                problems.append(f"{where}: from {node.module} import ...  -> {err}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except Exception:  # noqa: BLE001
                        problems.append(
                            f"{where}: from {node.module} import {alias.name}"
                            "  -> name not found"
                        )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "pynasonde":
                    checked += 1
                    obj, err = resolve(alias.name)
                    if obj is None:
                        problems.append(f"{where}: import {alias.name}  -> {err}")
    return checked


def check_markdown(path: Path, rel: str, problems: list, counts: dict) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")

    for k, match in enumerate(FENCE.finditer(text), 1):
        body, indent = match.group("body"), match.group("ind")
        if indent:
            body = "\n".join(
                line[len(indent) :] if line.startswith(indent) else line
                for line in body.split("\n")
            )
        counts["blocks"] += 1
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            problems.append(
                f"{rel} block {k}: syntax error, line {exc.lineno}: {exc.msg}"
            )
            continue
        counts["imports"] += check_imports(tree, f"{rel} block {k}", problems)

    lines = text.split("\n")
    for i, line in enumerate(lines):
        match = DIRECTIVE.match(line)
        if not match:
            continue
        counts["directives"] += 1
        target = match.group(1)
        obj, err = resolve(target)
        if obj is None:
            problems.append(f"{rel}: ::: {target}  -> {err}")
            continue
        in_members = False
        for nxt in lines[i + 1 :]:
            if nxt.strip() and not nxt.startswith(" "):
                break
            stripped = nxt.strip()
            if stripped == "members:":
                in_members = True
            elif in_members and stripped.startswith("- "):
                member = stripped[2:].strip()
                if not hasattr(obj, member):
                    problems.append(f"{rel}: ::: {target}  member '{member}' not found")
            elif stripped and in_members:
                in_members = False


def main(src: Path) -> int:
    problems: list = []
    counts = {"blocks": 0, "imports": 0, "directives": 0, "examples": 0}

    for path in sorted((src / "docs").rglob("*.md")):
        check_markdown(path, str(path.relative_to(src)), problems, counts)

    for path in sorted((src / "examples").rglob("*.py")):
        rel = str(path.relative_to(src))
        counts["examples"] += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            problems.append(f"{rel}: syntax error, line {exc.lineno}: {exc.msg}")
            continue
        counts["imports"] += check_imports(tree, rel, problems)

    print(
        f"checked {counts['directives']} API directives, {counts['blocks']} code blocks, "
        f"{counts['examples']} example scripts, {counts['imports']} pynasonde imports"
    )
    for problem in problems:
        print(f"  PROBLEM  {problem}")
    print("docs check:", "FAILED" if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
