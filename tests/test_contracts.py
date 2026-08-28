"""Import and contract smoke tests for the method scaffold."""
import importlib

def test_stage_modules_import():
    for name in ("geometry", "building", "semantic", "materials", "ground", "georef", "export"):
        importlib.import_module(f"src.{name}")

def test_key_contracts_exist():
    from src.semantic import SemanticSource, fuse_semantics
    from src.ground import GroundLabel, label_ground
    from src.geometry import reconstruct_geometry
    assert SemanticSource.PHOTO.value == "photo"
    assert GroundLabel.ROAD.value == "road"
    assert callable(fuse_semantics)
    assert callable(label_ground)
    assert callable(reconstruct_geometry)
