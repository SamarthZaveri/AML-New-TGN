"""The clean project must resolve its own imports and retain verified formulas."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_internal_imports_exist_in_this_repository():
    missing = []
    for path in ROOT.rglob("*.py"):
        if "data" in path.relative_to(ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            if name.split(".")[0] in {"streaming", "preprocessing", "validation"}:
                target = ROOT.joinpath(*name.split("."))
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    missing.append((str(path.relative_to(ROOT)), name))
    assert not missing, missing


def test_native_processors_and_static_formulas_match_verified_source():
    manifest = json.loads((ROOT / "validation/source_checksums.json").read_text())
    for name, expected in manifest["unchanged_files"].items():
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        assert hashlib.sha256(text.encode()).hexdigest() == expected, name


def test_pipeline_imports_resolve_inside_the_clean_project():
    import streaming.ingestion.pipeline as pipeline
    import streaming.ingestion.events as events
    import streaming.ingestion.outputs as outputs
    for module in (pipeline, events, outputs):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)


def test_runtime_coordination_and_preparation_definitions_are_unchanged():
    manifest = json.loads((ROOT / "validation/source_checksums.json").read_text())
    for path, definitions in manifest["unchanged_definitions"].items():
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        for name, expected in definitions.items():
            node = next(item for item in tree.body if getattr(item, "name", None) == name)
            source = (ROOT / path).read_text(encoding="utf-8-sig")
            actual = hashlib.sha256(ast.get_source_segment(source, node).strip().encode()).hexdigest()
            assert actual == expected, (path, name)


def test_module_names_used_by_subprocess_commands_exist():
    missing = []
    for path in ROOT.rglob("*.py"):
        if "data" in path.relative_to(ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.fullmatch(r"(?:streaming|preprocessing|validation)\.[A-Za-z0-9_.]+", node.value):
                    target = ROOT.joinpath(*node.value.split("."))
                    if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                        missing.append((str(path.relative_to(ROOT)), node.value))
    assert not missing, missing
