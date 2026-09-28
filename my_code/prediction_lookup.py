"""Compare original ABCParts annotations with HPNet predictions in Open3D.

Run ``python my_code/prediction_lookup.py`` to choose a saved prediction, or pass
an H5 path directly. Both copies share one camera and the same point indices.
"""

import argparse
import random
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

if __package__:
    from .ABCViewer import ABCViewer, categorical_colors
else:
    from ABCViewer import ABCViewer, categorical_colors


DEFAULT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "outputs" / "inspection_skip100_3254"
)


def load_prediction(filepath):
    """Read one saved inspection file and check its per-point fields."""
    keys = ("points", "normals", "labels", "prim", "T_param", "labels_pred", "prim_pred")
    with h5py.File(filepath, "r") as h5_file:
        missing = [key for key in keys if key not in h5_file]
        if missing:
            raise KeyError(f"Missing dataset(s): {', '.join(missing)}")
        print(f"\n{filepath}")
        for key in h5_file:
            dataset = h5_file[key]
            print(f"{key:12s} shape={dataset.shape} dtype={dataset.dtype}")
        data = {key: np.asarray(h5_file[key][:]) for key in keys}

    points = np.asarray(data["points"], dtype=np.float64)
    normals = np.asarray(data["normals"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points must contain finite coordinates")
    if normals.shape != points.shape:
        raise ValueError(f"normals must have shape {points.shape}, got {normals.shape}")
    if not np.isfinite(normals).all():
        raise ValueError("normals must contain finite values")
    if data["T_param"].shape != (len(points), 22):
        raise ValueError(
            f"T_param must have shape ({len(points)}, 22), got {data['T_param'].shape}"
        )

    for key in ("labels", "prim", "labels_pred", "prim_pred"):
        values = data[key]
        if values.size != len(points):
            raise ValueError(f"{key} must contain {len(points)} values, got {values.shape}")
        values = values.reshape(-1)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain finite numeric values")
        data[key] = values

    data["points"] = points
    data["normals"] = normals
    return data


class PredictionViewer:
    """Two annotated copies of a point cloud in a single, shared-camera scene."""

    def __init__(self, data, model_name, window):
        self.data = data
        self.window = window
        self.model_name = model_name
        self.points = data["points"]
        self.count = len(self.points)
        self.selected_index = None
        self.pick_mode = False
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        extent = np.ptp(self.points, axis=0)
        diagonal = float(np.linalg.norm(extent))
        separation = float(extent[0]) + max(0.3 * diagonal, 0.1)
        self.offsets = (-separation / 2, separation / 2)
        self.display_points = []
        self.clouds = []
        self.materials = []
        self.geometry_names = ("original", "prediction")

        labels_both = np.concatenate((data["labels"], data["labels_pred"]))
        prim_both = np.concatenate((data["prim"], data["prim_pred"]))
        self.colors = {
            "Labels": np.split(categorical_colors(labels_both, seed=42), 2),
            "Primitive type": np.split(categorical_colors(prim_both, seed=10), 2),
            "Normals": [np.clip((data["normals"] + 1) / 2, 0, 1)] * 2,
        }
        for side, offset in enumerate(self.offsets):
            shifted = self.points.copy()
            shifted[:, 0] += offset
            self.display_points.append(shifted)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(shifted)
            cloud.normals = o3d.utility.Vector3dVector(data["normals"])
            cloud.colors = o3d.utility.Vector3dVector(self.colors["Labels"][side])
            material = ABCViewer._create_point_material()
            self.scene_widget.scene.add_geometry(self.geometry_names[side], cloud, material)
            self.clouds.append(cloud)
            self.materials.append(material)

        self.all_display_points = np.vstack(self.display_points)
        self.homogeneous_points = np.column_stack(
            (self.all_display_points, np.ones(2 * self.count))
        )
        self.bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(self.all_display_points)
        )
        self.selection_radius = max(0.015 * diagonal, 1e-6)
        self.selection_material = ABCViewer._create_point_material()
        self.camera_initialized = False
        camera = getattr(self.scene_widget.scene, "camera", None)
        self.mouse_picking_supported = (
            hasattr(self.scene_widget, "set_on_mouse")
            and camera is not None
            and hasattr(camera, "get_view_matrix")
            and hasattr(camera, "get_projection_matrix")
        )
        if self.mouse_picking_supported:
            self.scene_widget.set_on_mouse(self.on_mouse)

        self._build_controls()

    def _build_controls(self):
        em = self.window.theme.font_size
        self.original_header = gui.Label("Original")
        self.prediction_header = gui.Label("Prediction")
        self.panel = gui.Vert(
            0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)
        )
        self.panel.add_child(gui.Label(self._statistics()))
        self.panel.add_child(gui.Label("Color both copies by"))
        for title in ("Labels", "Primitive type", "Normals"):
            button = gui.Button(title)
            button.set_on_clicked(lambda name=title: self.set_color_mode(name))
            self.panel.add_child(button)
        self.panel.add_child(gui.Label("Normals uses the original input on both sides."))

        self.panel.add_child(gui.Label("Point size"))
        self.point_size_slider = gui.Slider(gui.Slider.INT)
        self.point_size_slider.set_limits(1, 20)
        self.point_size_slider.int_value = int(self.materials[0].point_size)
        self.point_size_slider.set_on_value_changed(self.set_point_size)
        self.panel.add_child(self.point_size_slider)

        self.pick_button = gui.Button("Pick point")
        self.pick_button.enabled = self.mouse_picking_supported
        self.pick_button.set_on_clicked(self.toggle_pick_mode)
        self.panel.add_child(self.pick_button)
        self.pick_hint = gui.Label(
            "Click 'Pick point', then click either copy."
            if self.mouse_picking_supported
            else f"Point picking needs Open3D 0.13+ (installed: {o3d.__version__})."
        )
        self.panel.add_child(self.pick_hint)
        self.selection_info = gui.Label("No point selected.")
        self.panel.add_child(self.selection_info)

        self.window.add_child(self.original_header)
        self.window.add_child(self.prediction_header)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)
        self.window.set_needs_layout()
        self.window.post_redraw()

    def _statistics(self):
        lines = [f"Model: {self.model_name}", f"Points: {self.count:,}"]
        for title, labels_key, prim_key in (
            ("Original", "labels", "prim"),
            ("Prediction", "labels_pred", "prim_pred"),
        ):
            labels = self.data[labels_key]
            prim = self.data[prim_key]
            types, counts = np.unique(prim, return_counts=True)
            lines.append(f"{title}: {len(np.unique(labels)):,} instances; {len(types)} types")
            for value, count in zip(types, counts):
                name = ABCViewer._PRIMITIVE_NAMES.get(int(value), f"Unknown ({value})")
                lines.append(f"  {int(value)}: {name} ({count:,})")
        matches = np.count_nonzero(self.data["prim"] == self.data["prim_pred"])
        lines.append(f"Primitive type agreement: {matches / self.count:.1%}")
        return "\n".join(lines)

    def set_color_mode(self, name):
        for side in range(2):
            self.clouds[side].colors = o3d.utility.Vector3dVector(self.colors[name][side])
            self._refresh_cloud(side)

    def set_point_size(self, size):
        for side in range(2):
            self.materials[side].point_size = float(size)
            self._refresh_cloud(side)

    def _refresh_cloud(self, side):
        name = self.geometry_names[side]
        self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, self.clouds[side], self.materials[side])
        self.window.post_redraw()

    def toggle_pick_mode(self):
        if not self.mouse_picking_supported:
            return
        self.pick_mode = not self.pick_mode
        self.pick_button.text = "Stop picking" if self.pick_mode else "Pick point"
        self.pick_hint.text = (
            "Click a visible point on either copy."
            if self.pick_mode
            else "Click 'Pick point' to inspect another point."
        )

    def on_mouse(self, event):
        if (
            not self.pick_mode
            or event.type != gui.MouseEvent.Type.BUTTON_DOWN
            or not event.is_button_down(gui.MouseButton.LEFT)
        ):
            return gui.SceneWidget.EventCallbackResult.IGNORED
        frame = self.scene_widget.frame
        x, y = event.x - frame.x, event.y - frame.y
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return gui.SceneWidget.EventCallbackResult.IGNORED
        try:
            self._select_projected_point(x, y, frame.width, frame.height)
        except Exception as error:
            self.pick_hint.text = f"Point picking failed: {error}"
            self.window.post_redraw()
        return gui.SceneWidget.EventCallbackResult.CONSUMED

    def _select_projected_point(self, x, y, width, height):
        camera = self.scene_widget.scene.camera
        view_projection = (
            np.asarray(camera.get_projection_matrix(), dtype=np.float64)
            @ np.asarray(camera.get_view_matrix(), dtype=np.float64)
        )
        clip = self.homogeneous_points @ view_projection.T
        visible = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
        candidates = np.flatnonzero(visible)
        ndc = clip[visible, :3] / clip[visible, 3][:, None]
        in_view = np.all(np.abs(ndc) <= 1, axis=1)
        candidates = candidates[in_view]
        depths = clip[visible, 3][in_view]
        ndc = ndc[in_view]
        if not len(candidates):
            self.pick_hint.text = "No point at that location."
            self.window.post_redraw()
            return
        distance_sq = ((ndc[:, 0] + 1) * width / 2 - x) ** 2 + (
            (1 - ndc[:, 1]) * height / 2 - y
        ) ** 2
        closest = int(np.argmin(distance_sq))
        if distance_sq[closest] > max(8.0, self.materials[0].point_size) ** 2:
            self.pick_hint.text = "No point at that location."
            self.window.post_redraw()
            return
        nearby = np.flatnonzero(distance_sq <= distance_sq[closest] + 1.0)
        picked = nearby[np.argmin(depths[nearby])]
        self._show_point(int(candidates[picked] % self.count))

    def _show_point(self, index):
        for side in range(2):
            marker_name = f"selected_{side}"
            if self.selected_index is not None:
                self.scene_widget.scene.remove_geometry(marker_name)
            marker = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.selection_radius, resolution=12
            )
            marker.paint_uniform_color([1.0, 0.0, 0.0])
            marker.translate(self.display_points[side][index])
            self.scene_widget.scene.add_geometry(marker_name, marker, self.selection_material)
        self.selected_index = index
        point = self.points[index]
        normal = self.data["normals"][index]
        original_type = int(self.data["prim"][index])
        predicted_type = int(self.data["prim_pred"][index])
        names = ABCViewer._PRIMITIVE_NAMES
        parameter_slice = ABCViewer._PARAMETER_SLICES.get(original_type)
        if parameter_slice is None:
            params = np.array2string(
                self.data["T_param"][index], precision=5, separator=", ", max_line_width=35
            )
            parameter_text = f"Original T_param: {params}"
        else:
            start, stop = parameter_slice
            params = np.array2string(
                self.data["T_param"][index, start:stop],
                precision=5,
                separator=", ",
                max_line_width=35,
            )
            parameter_text = f"Original T_param[{start}:{stop}]: {params}"
        self.pick_hint.text = "Click another visible point to inspect it."
        self.selection_info.text = (
            f"Index: {index}\n"
            f"XYZ: {point[0]:.5f}, {point[1]:.5f}, {point[2]:.5f}\n"
            f"Normal: {normal[0]:.5f}, {normal[1]:.5f}, {normal[2]:.5f}\n"
            f"Original instance: {self.data['labels'][index]}\n"
            f"Original type: {names.get(original_type, 'Unknown')} ({original_type})\n"
            f"{parameter_text}\n"
            f"Predicted instance: {self.data['labels_pred'][index]}\n"
            f"Predicted type: {names.get(predicted_type, 'Unknown')} ({predicted_type})"
        )
        self.window.post_redraw()

    def on_layout(self, _):
        rect = self.window.content_rect
        panel_width = min(420, max(300, rect.width // 3))
        header_height = 28
        scene_width = rect.width - panel_width
        self.original_header.frame = gui.Rect(
            rect.x, rect.y, scene_width // 2, header_height
        )
        self.prediction_header.frame = gui.Rect(
            rect.x + scene_width // 2, rect.y, scene_width - scene_width // 2, header_height
        )
        self.scene_widget.frame = gui.Rect(
            rect.x, rect.y + header_height, scene_width, rect.height - header_height
        )
        self.panel.frame = gui.Rect(
            rect.get_right() - panel_width, rect.y, panel_width, rect.height
        )
        if (
            not self.camera_initialized
            and self.scene_widget.frame.width > 0
            and self.scene_widget.frame.height > 0
        ):
            self.camera_initialized = True
            self.scene_widget.setup_camera(60, self.bounds, self.bounds.get_center())
            self.window.post_redraw()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="prediction H5 file to open")
    parser.add_argument(
        "--directory", type=Path, default=DEFAULT_DIRECTORY, help="folder for random/file choice"
    )
    args = parser.parse_args()
    app = gui.Application.instance
    app.initialize()
    chooser = app.create_window("HPNet prediction comparison", 1450, 850)
    layout = gui.Vert(12, gui.Margins(20, 20, 20, 20))
    layout.add_child(gui.Label("Choose a saved HPNet prediction"))
    status = gui.Label("")
    layout.add_child(status)
    active_viewer = None

    def open_model(filepath):
        nonlocal active_viewer
        try:
            data = load_prediction(filepath)
            active_viewer = PredictionViewer(data, filepath.name, chooser)
            layout.visible = False
            chooser.set_needs_layout()
        except Exception as error:
            traceback.print_exc()
            status.text = f"Could not open {filepath.name}: {error}"
            chooser.set_needs_layout()
            if hasattr(chooser, "show_message_box"):
                chooser.show_message_box("Could not open prediction", str(error))

    def choose_random():
        try:
            files = [
                path for path in args.directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
            ]
        except OSError as error:
            status.text = f"Could not read {args.directory}: {error}"
            chooser.post_redraw()
            return
        if files:
            open_model(random.choice(files))
        else:
            status.text = f"No H5 files found in {args.directory}"
            chooser.post_redraw()

    def choose_file():
        file_button.enabled = False
        status.text = "Opening file explorer..."
        chooser.set_needs_layout()

        def run_picker():
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("native_file_picker.py")),
                        str(args.directory),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                selected = result.stdout.decode("utf-8", errors="replace")
                error = result.stderr.decode("utf-8", errors="replace")
                failed = result.returncode != 0
            except Exception as exception:
                selected, error, failed = "", str(exception), True

            def finish():
                file_button.enabled = True
                status.text = ""
                if failed:
                    status.text = f"Could not open file explorer: {error}"
                    chooser.set_needs_layout()
                elif selected:
                    open_model(Path(selected))
                else:
                    chooser.set_needs_layout()

            app.post_to_main_thread(chooser, finish)

        threading.Thread(target=run_picker, daemon=True).start()

    random_button = gui.Button("Choose random")
    file_button = gui.Button("Select a file")
    random_button.set_on_clicked(choose_random)
    file_button.set_on_clicked(choose_file)
    layout.add_child(random_button)
    layout.add_child(file_button)
    chooser.add_child(layout)

    def chooser_layout(_):
        rect = chooser.content_rect
        layout.frame = gui.Rect(
            rect.x + (rect.width - 520) // 2,
            rect.y + (rect.height - 220) // 2,
            520,
            220,
        )

    chooser.set_on_layout(chooser_layout)
    if args.file is not None:
        open_model(args.file)
    app.run()


if __name__ == "__main__":
    main()
