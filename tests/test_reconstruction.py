"""Deterministic regression tests for the automatic reconstruction baseline."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.reconstruction.footprints import extract_footprints
from src.reconstruction.heights import BuildingPart
from src.reconstruction.material_transfer import transfer_surface_materials
from src.reconstruction.mesh import build_clean_mesh


class ReconstructionTests(unittest.TestCase):
    def test_plan_extraction_and_closed_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            image[20:100, 25:105] = (110, 165, 210)  # OpenCV BGR -> plan-like RGB fill
            plan = Path(directory) / "plan.png"
            self.assertTrue(cv2.imwrite(str(plan), image))
            footprints, audit = extract_footprints(plan, {
                "crop_px": [0, 0, 128, 128],
                "metres_per_pixel": 1.0,
                "rgb_lower": [160, 120, 70],
                "rgb_upper": [245, 220, 190],
                "channel_difference_min": {"r_minus_g": 10, "g_minus_b": 10},
                "minimum_component_pixels": 100,
                "minimum_area_m2": 40,
                "minimum_hole_area_m2": 100,
                "simplify_m": 1.0,
            })
            self.assertEqual(audit["building_count"], 1)
            self.assertEqual(len(footprints), 1)
            part = BuildingPart(
                "B01", "B01", Polygon(footprints[0].outer_local_m),
                0.0, 12.0, 12.0, "synthetic", (),
            )
            mesh, geometry = build_clean_mesh([part])
            self.assertEqual(geometry["status"], "PASS")
            self.assertEqual(geometry["non_manifold_coordinate_edges"], 0)
            self.assertGreater(len(mesh.faces), 0)

    def test_material_transfer_aggregates_by_surface(self) -> None:
        part = BuildingPart("B01", "B01", Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), 0, 4, 4, "synthetic", ())
        mesh, geometry = build_clean_mesh([part])
        self.assertEqual(geometry["status"], "PASS")
        voxels = {}
        for x in range(-1, 6):
            for y in range(-1, 6):
                for z in range(-1, 6):
                    voxels[(x, y, z)] = (8, 255, 10)
        rows, face_materials, audit = transfer_surface_materials(mesh, voxels, {
            "cell_m": 1.0,
            "neighbor_cells": 1,
            "minimum_hit_rate": 0.2,
            "minimum_top_vote_share": 0.5,
            "minimum_confidence": 0.2,
        })
        facade_rows = [row for row in rows if row.surface_type == "facade"]
        self.assertTrue(facade_rows)
        self.assertTrue(all(row.label == "glass_facade" for row in facade_rows))
        self.assertEqual(len(face_materials), len(mesh.faces))
        self.assertEqual(audit["status"], "PASS_AUTOMATIC_CANDIDATES")


if __name__ == "__main__":
    unittest.main()
