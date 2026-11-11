import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIPPED_DIRS = {".venv", ".git", ".superpowers", ".claude", ".ipynb_checkpoints", "node_modules"}
MAX_CELL_LINES = 15
FORBIDDEN_TEST_WORDS = ("assert", "unittest", "pytest")
FORBIDDEN_EXECUTION_MODULES = ("nbconvert", "nbclient")
FORBIDDEN_EXECUTION_NAMES = ("ExecutePreprocessor",)


def notebook_paths() -> list[Path]:
    found = []
    for path in ROOT.rglob("*.ipynb"):
        relative_parts = path.relative_to(ROOT).parts
        if any(part in SKIPPED_DIRS for part in relative_parts[:-1]):
            continue
        found.append(path)
    return sorted(found)


def code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            cells.append((index, source))
    return cells


def definition_kinds(source: str) -> list[str]:
    """Kinds of function, class and lambda definitions found by parsing the cell."""
    kinds = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            kinds.append("def")
        elif isinstance(node, ast.ClassDef):
            kinds.append("class")
        elif isinstance(node, ast.Lambda):
            kinds.append("lambda")
    return kinds


def checked_definition_kinds(where: str, source: str) -> list[str]:
    """Definition kinds of a cell. IPython magic and shell lines are skipped.

    A cell that still cannot be parsed fails with a message naming the cell.
    """
    lines = [
        "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
    ]
    try:
        return definition_kinds("\n".join(lines))
    except SyntaxError as error:
        raise AssertionError(f"{where}: cell cannot be parsed ({error.msg}, line {error.lineno})")


def test_cell_with_ipython_magics_is_parsed_without_syntax_error():
    source = "%matplotlib inline\n!ls\nvalue = 1\n"
    assert checked_definition_kinds("probe.ipynb cell 0", source) == []
    assert checked_definition_kinds("probe.ipynb cell 1", "%time\ndef f():\n    pass\n") == ["def"]


def test_unparseable_cell_fails_with_message_naming_notebook_and_cell():
    with pytest.raises(AssertionError) as excinfo:
        checked_definition_kinds("probe.ipynb cell 3", "x = (\n")
    assert "probe.ipynb cell 3" in str(excinfo.value)


def test_no_function_or_class_definitions_in_notebooks():
    for path in notebook_paths():
        for index, source in code_cells(path):
            kinds = checked_definition_kinds(f"{path.relative_to(ROOT)} cell {index}", source)
            assert not kinds, f"{path.relative_to(ROOT)} cell {index}: {kinds}"


def test_no_test_or_assert_code_in_notebooks():
    for path in notebook_paths():
        for index, source in code_cells(path):
            for word in FORBIDDEN_TEST_WORDS:
                assert word not in source, f"{path.relative_to(ROOT)} cell {index}: {word}"


def test_notebook_code_cells_are_short():
    for path in notebook_paths():
        for index, source in code_cells(path):
            lines = len(source.splitlines())
            assert lines <= MAX_CELL_LINES, f"{path.relative_to(ROOT)} cell {index}: {lines} lines"


def test_notebooks_have_no_stored_outputs():
    for path in notebook_paths():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            where = f"{path.relative_to(ROOT)} cell {index}"
            assert cell.get("outputs", []) == [], f"{where}: stored output"
            assert cell.get("execution_count") is None, f"{where}: execution count"


def test_test_suite_never_executes_notebooks():
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] + [alias.name for alias in node.names]
            for module in modules:
                assert module.split(".")[0] not in FORBIDDEN_EXECUTION_MODULES, (
                    f"{path.name}: imports {module}"
                )
                assert module not in FORBIDDEN_EXECUTION_NAMES, f"{path.name}: imports {module}"
