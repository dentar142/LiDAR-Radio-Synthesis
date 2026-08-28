"""Import and contract smoke tests for the method scaffold."""
import importlib

def test_stage_modules_import():
    for name in ("geometry", "building", "semantic", "materials", "ground", "georef", "export"):
        importlib.import_module(f"src.{name}")

def test_key_contracts_exist():
    from src.semantic import ReconstructionPolicy, SegmentationLevel, SemanticSource, fuse_semantics
    from src.ground import GroundLabel, label_ground
    from src.geometry import reconstruct_geometry
    assert SemanticSource.PHOTO.value == "photo"
    assert SemanticSource.EXISTING_MESH_SEMANTIC.value == "existing_mesh_semantic"
    assert SegmentationLevel.SURFACE_COMPONENT.value == "surface_component"
    assert ReconstructionPolicy.REGULARIZE_PLANE.value == "regularize_plane"
    assert GroundLabel.ROAD.value == "road"
    assert callable(fuse_semantics)
    assert callable(label_ground)
    assert callable(reconstruct_geometry)
