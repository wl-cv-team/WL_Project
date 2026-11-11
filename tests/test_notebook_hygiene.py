import ast
import json
from pathlib import Path

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


def test_no_function_or_class_definitions_in_notebooks():
    for path in notebook_paths():
        for index, source in code_cells(path):
            kinds = definition_kinds(source)
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
