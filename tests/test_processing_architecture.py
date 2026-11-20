"""Static dependency checks for the common processing layers."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "spectrochempy_nmr"


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_common_processing_has_no_vendor_or_quaternion_imports():
    for name in ("complex_channels.py", "quadrature.py"):
        imports = _imported_modules(ROOT / "processing" / name)
        assert not any(
            "adapters" in module or "readers" in module for module in imports
        )
        assert not any("hypercomplex" in module for module in imports)


def test_entire_runtime_has_no_quaternion_dependency():
    for path in ROOT.rglob("*.py"):
        imports = _imported_modules(path)
        assert not any(
            module.split(".")[0] in {"quaternion", "spectrochempy_hypercomplex"}
            for module in imports
        ), path
    assert not (ROOT / "processing" / "quaternion_phasing.py").exists()
