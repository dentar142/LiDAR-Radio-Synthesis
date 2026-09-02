"""Render the 155-second HKUSTGZ clean-map research film in Blender 4.5."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector


FPS = 30
FRAME_END = 4650
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CYAN = (0.05, 0.78, 0.95, 1.0)
BLUE = (0.12, 0.36, 0.96, 1.0)
ORANGE = (1.0, 0.45, 0.09, 1.0)
YELLOW = (1.0, 0.76, 0.12, 1.0)
GREEN = (0.22, 0.88, 0.54, 1.0)
WHITE = (0.92, 0.96, 1.0, 1.0)
MUTED = (0.48, 0.57, 0.69, 1.0)
GREY = (0.12, 0.16, 0.22, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-obj", type=Path, required=True)
    parser.add_argument("--raw-obj", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--stills", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def principled(result: bpy.types.Material) -> bpy.types.Node:
    shader = next((node for node in result.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        shader = result.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = next(node for node in result.node_tree.nodes if node.type == "OUTPUT_MATERIAL")
        result.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return shader


def material(name: str, color: tuple[float, float, float, float], *, metallic: float = 0.0,
             roughness: float = 0.48, emission: float = 0.0) -> bpy.types.Material:
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = principled(result)
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = color
        shader.inputs["Emission Strength"].default_value = emission
    return result


def emission_material(name: str, color: tuple[float, float, float, float], strength: float) -> bpy.types.Material:
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return result


def parse_clean_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[str]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_materials: list[str] = []
    current = "UNASSIGNED"
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw in source:
            if raw.startswith("v "):
                values = raw.split()
                vertices.append(tuple(float(value) for value in values[1:4]))
            elif raw.startswith("usemtl "):
                current = raw.split(maxsplit=1)[1].strip()
            elif raw.startswith("f "):
                polygon = [int(value.split("/", 1)[0]) - 1 for value in raw.split()[1:]]
                for offset in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
                    face_materials.append(current)
    return vertices, faces, face_materials


def normalize(vertices: list[tuple[float, float, float]]) -> tuple[list[tuple[float, float, float]], tuple[float, ...]]:
    x_values = [value[0] for value in vertices]
    y_values = [value[1] for value in vertices]
    z_values = [value[2] for value in vertices]
    centre_x = (min(x_values) + max(x_values)) * 0.5
    centre_y = (min(y_values) + max(y_values)) * 0.5
    minimum_z = min(z_values)
    transformed = [
        ((x - centre_x) * 0.010, (y - centre_y) * 0.010, (z - minimum_z) * 0.025)
        for x, y, z in vertices
    ]
    return transformed, (centre_x, centre_y, minimum_z, min(x_values), max(x_values), min(y_values), max(y_values), max(z_values))


def clean_model(path: Path) -> tuple[bpy.types.Object, dict[str, bpy.types.Material], tuple[float, ...]]:
    source_vertices, faces, face_materials = parse_clean_obj(path)
    vertices, bounds = normalize(source_vertices)
    mesh = bpy.data.meshes.new("CleanCampusMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("CleanCampus", mesh)
    bpy.context.collection.objects.link(obj)

    slots: OrderedDict[str, bpy.types.Material] = OrderedDict()
    for name in face_materials:
        if name not in slots:
            slots[name] = material("FILM_" + name, GREY, roughness=0.5)
            mesh.materials.append(slots[name])
    indexes = {name: index for index, name in enumerate(slots)}
    for polygon, name in zip(mesh.polygons, face_materials):
        polygon.material_index = indexes[name]
    return obj, dict(slots), bounds


def sample_raw_points(path: Path, bounds: tuple[float, ...], maximum: int = 95_000) -> list[tuple[float, float, float]]:
    centre_x, centre_y, minimum_z, min_x, max_x, min_y, max_y, _ = bounds
    reservoir: list[tuple[float, float, float]] = []
    accepted = 0
    random.seed(142)
    margin = 70.0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw in source:
            if not raw.startswith("v "):
                continue
            values = raw.split()
            x, y, z = (float(value) for value in values[1:4])
            if not (min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin):
                continue
            point = ((x - centre_x) * 0.010, (y - centre_y) * 0.010, (z - minimum_z) * 0.025)
            accepted += 1
            if len(reservoir) < maximum:
                reservoir.append(point)
            else:
                index = random.randrange(accepted)
                if index < maximum:
                    reservoir[index] = point
    return reservoir


def point_cloud(points: list[tuple[float, float, float]]) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("RawLidarVertices")
    mesh.from_pydata(points, [], [])
    obj = bpy.data.objects.new("RawLidarPointCloud", mesh)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("RenderPoints", "NODES")
    group = bpy.data.node_groups.new("RawLidarPointNodes", "GeometryNodeTree")
    modifier.node_group = group
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    to_points.inputs["Radius"].default_value = 0.009
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = emission_material("LiDAR_Cyan", CYAN, 2.0)
    links.new(input_node.outputs["Geometry"], to_points.inputs["Mesh"])
    links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])
    return obj


def image_plane(name: str, path: Path, z: float) -> bpy.types.Object:
    image = bpy.data.images.load(str(path))
    aspect = image.size[0] / image.size[1]
    mesh = bpy.data.meshes.new(name + "Mesh")
    width, height = 12.4, 12.4 / aspect
    mesh.from_pydata([(-width / 2, -height / 2, z), (width / 2, -height / 2, z),
                      (width / 2, height / 2, z), (-width / 2, height / 2, z)], [], [(0, 1, 2, 3)])
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers[0].data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new(name + "Material")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    emission.inputs["Strength"].default_value = 0.8
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.append(mat)
    return obj


def look_at(camera: bpy.types.Object, point: tuple[float, float, float]) -> None:
    camera.rotation_euler = (Vector(point) - camera.location).to_track_quat("-Z", "Y").to_euler()


def camera_key(camera: bpy.types.Object, frame: int, location: tuple[float, float, float],
               target: tuple[float, float, float]) -> None:
    camera.location = location
    look_at(camera, target)
    camera.keyframe_insert("location", frame=frame)
    camera.keyframe_insert("rotation_euler", frame=frame)


def visible(obj: bpy.types.Object, start: int, end: int) -> None:
    obj.hide_render = True
    obj.keyframe_insert("hide_render", frame=max(1, start - 1))
    obj.hide_render = False
    obj.keyframe_insert("hide_render", frame=start)
    obj.keyframe_insert("hide_render", frame=end)
    if end < FRAME_END:
        obj.hide_render = True
        obj.keyframe_insert("hide_render", frame=end + 1)


def camera_text(camera: bpy.types.Object, body: str, start: int, end: int, *,
                position: tuple[float, float, float] = (-2.5, 1.35, -8.0), size: float = 0.25,
                color: tuple[float, float, float, float] = WHITE, align: str = "LEFT") -> bpy.types.Object:
    curve = bpy.data.curves.new("Text_" + str(start), "FONT")
    curve.body = body
    curve.align_x = align
    curve.align_y = "TOP"
    curve.size = size
    curve.space_line = 1.18
    curve.font = bpy.data.fonts.load(FONT)
    obj = bpy.data.objects.new("Text_" + str(start) + "_" + body[:8], curve)
    bpy.context.collection.objects.link(obj)
    obj.parent = camera
    obj.location = position
    curve.materials.append(emission_material("TextMat_" + str(start) + body[:4], color, 2.0))
    visible(obj, start, end)
    obj.scale = (0.92, 0.92, 0.92)
    obj.keyframe_insert("scale", frame=start)
    obj.scale = (1.0, 1.0, 1.0)
    obj.keyframe_insert("scale", frame=start + 18)
    return obj


def camera_panel(camera: bpy.types.Object, start: int, end: int, *,
                 position: tuple[float, float, float], size: tuple[float, float]) -> bpy.types.Object:
    width, height = size
    mesh = bpy.data.meshes.new(f"PanelMesh_{start}")
    mesh.from_pydata(
        [(-width / 2, -height / 2, 0), (width / 2, -height / 2, 0),
         (width / 2, height / 2, 0), (-width / 2, height / 2, 0)],
        [], [(0, 1, 2, 3)],
    )
    panel = bpy.data.objects.new(f"Panel_{start}", mesh)
    bpy.context.collection.objects.link(panel)
    panel.parent = camera
    panel.location = position
    panel.data.materials.append(emission_material(f"PanelMat_{start}", (0.004, 0.009, 0.020, 1.0), 0.8))
    visible(panel, start, end)
    panel.scale = (0.94, 0.94, 0.94)
    panel.keyframe_insert("scale", frame=start)
    panel.scale = (1.0, 1.0, 1.0)
    panel.keyframe_insert("scale", frame=start + 18)
    return panel


def add_scan_rings(start: int, end: int) -> None:
    for index in range(3):
        bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.018, location=(0, 0, 0.1))
        ring = bpy.context.object
        ring.name = f"SemanticScanRing_{index}"
        ring.data.materials.append(emission_material(f"ScanRingMat_{index}", ORANGE, 6.0))
        offset = index * 115
        ring.scale = (0.15, 0.15, 0.15)
        ring.location.z = 0.08
        ring.keyframe_insert("scale", frame=start + offset)
        ring.keyframe_insert("location", frame=start + offset)
        ring.scale = (7.2, 7.2, 7.2)
        ring.location.z = 1.55
        ring.keyframe_insert("scale", frame=min(end, start + offset + 280))
        ring.keyframe_insert("location", frame=min(end, start + offset + 280))
        visible(ring, start + offset, min(end, start + offset + 300))


def animate_materials(materials: dict[str, bpy.types.Material], start: int) -> None:
    targets = {
        "STRUCTURAL_BASE": (0.08, 0.10, 0.14, 1.0),
        "CONCRETE": (0.12, 0.38, 0.94, 1.0),
        "GLASS": (0.03, 0.78, 0.96, 1.0),
        "ROOF": (1.0, 0.42, 0.08, 1.0),
        "OTHER": (0.42, 0.48, 0.58, 1.0),
    }
    for index, (name, mat) in enumerate(materials.items()):
        shader = principled(mat)
        target = next((value for key, value in targets.items() if key in name.upper()), targets["OTHER"])
        begin = start + index * 28
        shader.inputs["Base Color"].default_value = GREY
        shader.inputs["Base Color"].keyframe_insert("default_value", frame=begin)
        shader.inputs["Base Color"].default_value = target
        shader.inputs["Base Color"].keyframe_insert("default_value", frame=begin + 110)
        if "Emission Color" in shader.inputs:
            shader.inputs["Emission Color"].default_value = target
            shader.inputs["Emission Strength"].default_value = 0.12


def setup_scene(args: argparse.Namespace) -> None:
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 960 if args.preview else 1920
    scene.render.resolution_y = 540 if args.preview else 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(args.output)
    scene.render.film_transparent = False
    scene.world.color = (0.004, 0.007, 0.014)

    camera_data = bpy.data.cameras.new("PresentationCamera")
    camera_data.lens = 50
    camera = bpy.data.objects.new("PresentationCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    clean, clean_materials, bounds = clean_model(args.clean_obj)
    raw = point_cloud(sample_raw_points(args.raw_obj, bounds))
    problem = image_plane("ProblemReference", args.problem, 30.0)
    plan = image_plane("MasterPlan", args.plan, 30.0)
    registration = image_plane("Registration", args.registration, 30.0)

    floor_mat = material("GroundStage", (0.012, 0.018, 0.033, 1.0), roughness=0.72)
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, -0.025))
    floor = bpy.context.object
    floor.data.materials.append(floor_mat)

    area_data = bpy.data.lights.new("KeyArea", "AREA")
    area_data.energy = 1050
    area_data.shape = "DISK"
    area_data.size = 11
    area = bpy.data.objects.new("KeyArea", area_data)
    area.location = (-4, -6, 12)
    bpy.context.collection.objects.link(area)
    look_at(area, (0, 0, 0))
    rim_data = bpy.data.lights.new("RimArea", "AREA")
    rim_data.energy = 850
    rim_data.color = (0.12, 0.42, 1.0)
    rim_data.size = 9
    rim = bpy.data.objects.new("RimArea", rim_data)
    rim.location = (7, 4, 8)
    bpy.context.collection.objects.link(rim)
    look_at(rim, (0, 0, 0.5))

    visible(problem, 241, 480)
    visible(raw, 1, 810)
    visible(plan, 811, 1410)
    visible(registration, 1411, 2010)
    visible(clean, 2011, FRAME_END)
    visible(floor, 481, FRAME_END)

    raw.rotation_euler.z = -0.3
    raw.keyframe_insert("rotation_euler", frame=481)
    raw.rotation_euler.z = 1.05
    raw.keyframe_insert("rotation_euler", frame=810)

    clean.scale = (1.0, 1.0, 0.015)
    clean.keyframe_insert("scale", frame=2011)
    clean.scale = (1.0, 1.0, 1.0)
    clean.keyframe_insert("scale", frame=2780)
    animate_materials(clean_materials, 2911)
    add_scan_rings(2911, 3660)

    camera_key(camera, 1, (0, -15, 8), (0, 0, 0.5))
    camera_key(camera, 240, (3, -14, 7), (0, 0, 0.5))
    camera_key(camera, 241, (0, 0, 39), (0, 0, 30))
    camera_key(camera, 480, (0.5, -0.3, 38), (0, 0, 30))
    camera_key(camera, 481, (0, -14, 8), (0, 0, 0.5))
    camera_key(camera, 810, (11, -9, 7), (0, 0, 0.5))
    camera_key(camera, 811, (0, 0, 39), (0, 0, 30))
    camera_key(camera, 1410, (0.6, -0.2, 38), (0, 0, 30))
    camera_key(camera, 1411, (0, 0, 39), (0, 0, 30))
    camera_key(camera, 2010, (-0.5, 0.4, 38), (0, 0, 30))
    camera_key(camera, 2011, (0, -16, 8), (0, 0, 0.45))
    camera_key(camera, 2910, (10, -11, 7), (0, 0, 0.55))
    camera_key(camera, 2911, (10, -11, 7), (0, 0, 0.55))
    camera_key(camera, 3660, (-10, -11, 6.5), (0, 0, 0.55))
    camera_key(camera, 3661, (-13, -8, 7), (0, 0, 0.5))
    camera_key(camera, 4110, (-11, -10, 6.5), (0, 0, 0.5))
    camera_key(camera, 4111, (0, -16, 8.5), (0, 0, 0.55))
    camera_key(camera, FRAME_END, (12, -9, 7.5), (0, 0, 0.55))

    camera_panel(camera, 1, 240, position=(0, 0.10, -8.08), size=(5.0, 1.55))
    camera_text(camera, "从碎片到可仿真地图", 1, 240, position=(0, 0.45, -8), size=0.48, align="CENTER")
    camera_text(camera, "工程图约束 · LiDAR 高度 · 连通表面材质", 30, 240,
                position=(0, -0.28, -8), size=0.17, color=CYAN, align="CENTER")
    camera_text(camera, "01  原始模型并不等于可信模型", 241, 810)
    camera_text(camera, "碎三角面  /  法线噪声  /  玻璃反射伪影", 300, 810,
                position=(-2.5, 0.92, -8), size=0.15, color=ORANGE)
    camera_text(camera, "02  工程图提供规则平面边界", 811, 1410)
    camera_text(camera, "35 栋建筑轮廓自动提取", 930, 1410,
                position=(-2.5, 0.92, -8), size=0.17, color=CYAN)
    camera_text(camera, "03  跨模态自动配准", 1411, 2010)
    camera_text(camera, "IoU  0.2689  →  0.6685", 1550, 2010,
                position=(-2.5, 0.92, -8), size=0.18, color=GREEN)
    camera_text(camera, "04  LiDAR 决定高度，图纸约束形状", 2011, 2910)
    camera_text(camera, "32 / 35 直接高度  ·  160 个封闭部件", 2260, 2910,
                position=(-2.5, 0.92, -8), size=0.16, color=CYAN)
    camera_text(camera, "05  原模型语义回投影到连通表面", 2911, 3660)
    camera_text(camera, "不按单个三角面独立猜测材质", 3070, 3660,
                position=(-2.5, 0.92, -8), size=0.16, color=ORANGE)
    camera_panel(camera, 3661, 4110, position=(-1.30, 0.12, -8.08), size=(3.0, 2.85))
    camera_text(camera, "运行门禁", 3661, 4110, position=(-2.55, 1.25, -8), size=0.28)
    camera_text(camera, "✓ 轮廓   ✓ 配准   ✓ 高度\n✓ 几何   ✓ 材质", 3720, 4110,
                position=(-2.55, 0.75, -8), size=0.20, color=GREEN)
    camera_text(camera, "1,377  自动候选\n1,654  待复核", 3820, 4110,
                position=(-2.55, -0.12, -8), size=0.17, color=YELLOW)
    camera_text(camera, "operational  PASS\nscientific  REVIEW_REQUIRED", 3950, 4110,
                position=(-2.55, -0.78, -8), size=0.14, color=MUTED)
    camera_panel(camera, 4111, FRAME_END, position=(0, -0.08, -8.08), size=(5.1, 1.85))
    camera_text(camera, "规则 · 可追踪 · 可继续标定", 4111, FRAME_END,
                position=(0, 0.35, -8), size=0.34, align="CENTER")
    camera_text(camera, "35 BUILDINGS   160 PARTS   3,031 SURFACES   54 s", 4250, FRAME_END,
                position=(0, -0.28, -8), size=0.15, color=CYAN, align="CENTER")
    camera_text(camera, "How to Make a Clean Map for Simulation", 4380, FRAME_END,
                position=(0, -0.72, -8), size=0.13, color=MUTED, align="CENTER")

    for fcurve in scene.animation_data.action.fcurves if scene.animation_data and scene.animation_data.action else []:
        for point in fcurve.keyframe_points:
            point.interpolation = "BEZIER"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")))


def main() -> None:
    args = arguments()
    setup_scene(args)
    if args.stills:
        still_directory = args.output.parent / "stills"
        still_directory.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.image_settings.file_format = "PNG"
        for frame in (120, 360, 650, 1050, 1700, 2300, 3200, 3850, 4400):
            bpy.context.scene.frame_set(frame)
            bpy.context.scene.render.filepath = str(still_directory / f"frame_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
