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
        assert not any("adapters" in module or "readers" in module for module in imports)
        assert not any("hypercomplex" in module for module in imports)


def test_agilent_adapter_depends_on_common_layers_not_the_reverse():
    imports = _imported_modules(ROOT / "adapters" / "agilent_noesy.py")
    assert "processing.complex_channels" in imports
    assert "processing.quadrature" in imports
    common_imports = _imported_modules(ROOT / "processing" / "complex_channels.py")
    assert not any("agilent" in module or "adapters" in module for module in common_imports)
