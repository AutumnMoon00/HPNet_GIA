import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def categorical_colors(values, seed=42):
    """Assign a repeatable random RGB color to each distinct label."""
    values = np.asarray(values).reshape(-1)
    unique_values = np.unique(values)
    rng = np.random.default_rng(seed)
    color_map = {value: rng.random(3) for value in unique_values}
    return np.asarray([color_map[value] for value in values])


class ABCViewer:
    """Interactive point-cloud viewer with selectable color schemes."""

    _GEOMETRY_NAME = "pointcloud"
    _SELECTION_GEOMETRY_NAME = "selected_point"
    _PRIMITIVE_NAMES = {
        0: "Closed B-spline",
        1: "Plane",
        2: "Open B-spline",
        3: "Cone",
        4: "Cylinder",
        5: "Sphere",
        6: "Closed B-spline",
        7: "Closed B-spline",
        8: "Open B-spline",
        9: "Closed B-spline",
    }
    _PARAMETER_SLICES = {
        1: (4, 8),
        3: (15, 22),
        4: (8, 15),
        5: (0, 4),
    }

    def __init__(
        self,
        point_cloud,
        label_colors,
        prim_colors,
        normal_colors,
        points,
        normals,
        labels,
        prim,
        t_param,
        model_name="",
        window=None,
    ):
        self.point_cloud = point_cloud
        self.label_colors = self._validate_colors(label_colors, "label_colors")
        self.prim_colors = self._validate_colors(prim_colors, "prim_colors")
        self.normal_colors = self._validate_colors(normal_colors, "normal_colors")
        self.points = self._validate_point_attributes(points, "points", (3,))
        self.normals = self._validate_point_attributes(normals, "normals", (3,))
        self.labels = self._validate_point_attributes(labels, "labels", ()).reshape(-1)
        self.prim = self._validate_point_attributes(prim, "prim", ()).reshape(-1)
        self.t_param = self._validate_point_attributes(t_param, "T_param", (22,))
        self.homogeneous_points = np.column_stack(
            (self.points, np.ones(len(self.points)))
        )
        self.model_name = model_name
        self.pick_mode = False
        self.selected_point_index = None
        self.scene_widget = gui.SceneWidget()

        self.window = window
        self.owns_window = window is None
        try:
            self._build_window()
        except Exception:
            if self.owns_window and self.window is not None:
                self.window.close()
            raise

    def _build_window(self):
        if self.window is None:
            self.window = gui.Application.instance.create_window(
                "ABCParts Viewer", 1200, 800
            )
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        camera = getattr(self.scene_widget.scene, "camera", None)
        self.mouse_picking_supported = (
            hasattr(self.scene_widget, "set_on_mouse")
            and camera is not None
            and hasattr(camera, "get_view_matrix")
            and hasattr(camera, "get_projection_matrix")
        )
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        if self.mouse_picking_supported:
            self.scene_widget.set_on_mouse(self.on_mouse)

        self.material = self._create_point_material()
        self.selection_material = self._create_point_material()
        self.scene_widget.scene.add_geometry(
            self._GEOMETRY_NAME, self.point_cloud, self.material
        )

        self.bounds = self.point_cloud.get_axis_aligned_bounding_box()
        self.selection_radius = max(
            0.015 * float(np.linalg.norm(self.bounds.get_extent())), 1e-6
        )
        self.camera_initialized = False

        em = self.window.theme.font_size
        self.panel = gui.Vert(
            0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)
        )
        self.panel.add_child(gui.Label("Color by"))
        self.panel.add_child(gui.Label(self._format_model_statistics()))

        self.label_button = gui.Button("Labels")
        self.prim_button = gui.Button("Primitive Type")
        self.normal_button = gui.Button("Normals")
        for button in (self.label_button, self.prim_button, self.normal_button):
            self.panel.add_child(button)

        self.panel.add_child(gui.Label("Point size"))
        self.point_size_slider = gui.Slider(gui.Slider.INT)
        self.point_size_slider.set_limits(1, 20)
        self.point_size_slider.int_value = int(self.material.point_size)
        self.panel.add_child(self.point_size_slider)

        self.pick_button = gui.Button("Pick point")
        self.pick_button.enabled = self.mouse_picking_supported
        self.panel.add_child(self.pick_button)
        self.pick_hint = gui.Label(
            "Click 'Pick point', then click a visible point."
            if self.mouse_picking_supported
            else f"Point picking needs Open3D 0.13+\nInstalled: {o3d.__version__}"
        )
        self.panel.add_child(self.pick_hint)
        self.selection_info = gui.Label("No point selected.")
        self.panel.add_child(self.selection_info)

        self.label_button.set_on_clicked(self.show_labels)
        self.prim_button.set_on_clicked(self.show_prim)
        self.normal_button.set_on_clicked(self.show_normals)
        self.point_size_slider.set_on_value_changed(self.set_point_size)
        self.pick_button.set_on_clicked(self.toggle_pick_mode)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)
        self.window.set_needs_layout()
        self.window.post_redraw()

    def _format_model_statistics(self):
        primitive_types, point_counts = np.unique(self.prim, return_counts=True)
        lines = [
            f"Model: {self.model_name}" if self.model_name else "Model statistics",
            f"Points: {len(self.points):,}",
            f"Primitive instances: {len(np.unique(self.labels)):,}",
            f"Primitive types: {len(primitive_types)}",
            "Points by type:",
        ]
        for primitive_type, count in zip(primitive_types, point_counts):
            primitive_type = int(primitive_type)
            name = self._PRIMITIVE_NAMES.get(
                primitive_type, f"Unknown ({primitive_type})"
            )
            lines.append(f"  {primitive_type}: {name} ({count:,} points)")
        return "\n".join(lines)

    def _validate_colors(self, colors, name):
        colors = np.asarray(colors, dtype=np.float64)
        point_count = len(self.point_cloud.points)
        if colors.shape != (point_count, 3):
            raise ValueError(
                f"{name} must have shape ({point_count}, 3), got {colors.shape}."
            )
        return np.clip(colors, 0.0, 1.0)

    def _validate_point_attributes(self, values, name, trailing_shape):
        values = np.asarray(values)
        point_count = len(self.point_cloud.points)
        expected_shape = (point_count,) + trailing_shape
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}.")
        return values

    @staticmethod
    def _create_point_material():
        """Support both old and current Open3D rendering APIs."""
        material_class = getattr(rendering, "MaterialRecord", None)
        if material_class is None:
            material_class = getattr(rendering, "Material", None)
        if material_class is None:
            raise RuntimeError(
                "This Open3D build does not provide a GUI rendering material. "
                "Install a version that includes open3d.visualization.rendering."
            )

        material = material_class()
        material.shader = "defaultUnlit"
        material.point_size = 4.0
        return material

    def update_colors(self, colors):
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self._refresh_geometry()

    def set_point_size(self, size):
        self.material.point_size = float(size)
        self._refresh_geometry()

    def toggle_pick_mode(self):
        if not self.mouse_picking_supported:
            return
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.pick_button.text = "Stop picking"
            self.pick_hint.text = "Click a visible point to inspect it."
        else:
            self.pick_button.text = "Pick point"
            self.pick_hint.text = "Click 'Pick point' to inspect another point."

    def on_mouse(self, event):
        if (
            not self.pick_mode
            or event.type != gui.MouseEvent.Type.BUTTON_DOWN
            or not event.is_button_down(gui.MouseButton.LEFT)
        ):
            return gui.SceneWidget.EventCallbackResult.IGNORED

        frame = self.scene_widget.frame
        click_x = event.x - frame.x
        click_y = event.y - frame.y
        if not (0 <= click_x < frame.width and 0 <= click_y < frame.height):
            return gui.SceneWidget.EventCallbackResult.IGNORED

        try:
            self._select_projected_point(click_x, click_y, frame.width, frame.height)
        except Exception as error:
            self._show_selection_message(f"Point picking failed: {error}")
        return gui.SceneWidget.EventCallbackResult.CONSUMED

    def _select_projected_point(self, click_x, click_y, width, height):
        camera = self.scene_widget.scene.camera
        view_projection = (
            np.asarray(camera.get_projection_matrix(), dtype=np.float64)
            @ np.asarray(camera.get_view_matrix(), dtype=np.float64)
        )
        clip = self.homogeneous_points @ view_projection.T
        in_front = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
        candidates = np.flatnonzero(in_front)
        ndc = clip[in_front, :3] / clip[in_front, 3][:, None]
        in_view = (
            (np.abs(ndc[:, 0]) <= 1)
            & (np.abs(ndc[:, 1]) <= 1)
            & (np.abs(ndc[:, 2]) <= 1)
        )
        candidates = candidates[in_view]
        camera_depth = clip[in_front, 3][in_view]
        ndc = ndc[in_view]
        if not len(candidates):
            self._show_selection_message("No point at that location.")
            return

        screen_x = (ndc[:, 0] + 1) * width / 2
        screen_y = (1 - ndc[:, 1]) * height / 2
        distance_sq = (screen_x - click_x) ** 2 + (screen_y - click_y) ** 2
        closest = int(np.argmin(distance_sq))
        if distance_sq[closest] > max(8.0, float(self.material.point_size)) ** 2:
            self._show_selection_message("No point at that location.")
            return

        # At the same screen position, prefer the point closest to the camera.
        nearby = np.flatnonzero(distance_sq <= distance_sq[closest] + 1.0)
        picked = nearby[np.argmin(camera_depth[nearby])]
        self._show_selected_point(int(candidates[picked]))

    def _show_selection_message(self, message):
        self.pick_hint.text = message
        self.window.post_redraw()

    def _show_selected_point(self, point_index):
        point = self.points[point_index]
        normal = self.normals[point_index]
        primitive_type = int(self.prim[point_index])
        surface_name = self._PRIMITIVE_NAMES.get(
            primitive_type, f"Unknown ({primitive_type})"
        )
        parameter_text = self._format_primitive_parameters(point_index, primitive_type)
        self._highlight_selected_point(point_index)
        self.pick_hint.text = "Click another visible point to inspect it."
        self.selection_info.text = (
            f"Index: {point_index}\n"
            f"XYZ: {point[0]:.5f}, {point[1]:.5f}, {point[2]:.5f}\n"
            f"Normal: {normal[0]:.5f}, {normal[1]:.5f}, {normal[2]:.5f}\n"
            f"Instance: {self.labels[point_index]}\n"
            f"Surface: {surface_name} (prim {primitive_type})\n"
            f"{parameter_text}"
        )
        self.window.post_redraw()

    def _format_primitive_parameters(self, point_index, primitive_type):
        parameter_slice = self._PARAMETER_SLICES.get(primitive_type)
        if parameter_slice is None:
            values = np.array2string(
                self.t_param[point_index],
                precision=5,
                separator=", ",
                max_line_width=70,
            )
            return f"T_param (full row): {values}"

        start, stop = parameter_slice
        values = np.array2string(
            self.t_param[point_index, start:stop],
            precision=5,
            separator=", ",
            max_line_width=32,
        )
        return f"T_param[{start}:{stop}]: {values}"

    def _highlight_selected_point(self, point_index):
        if self.selected_point_index is not None:
            self.scene_widget.scene.remove_geometry(self._SELECTION_GEOMETRY_NAME)

        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.selection_radius, resolution=12
        )
        marker.paint_uniform_color([1.0, 0.0, 0.0])
        marker.translate(self.points[point_index])
        self.selected_marker = marker
        self.selected_point_index = point_index
        self.scene_widget.scene.add_geometry(
            self._SELECTION_GEOMETRY_NAME,
            self.selected_marker,
            self.selection_material,
        )

    def _refresh_geometry(self):
        self.scene_widget.scene.remove_geometry(self._GEOMETRY_NAME)
        self.scene_widget.scene.add_geometry(
            self._GEOMETRY_NAME, self.point_cloud, self.material
        )

    def show_labels(self):
        print("Showing primitive instance labels")
        self.update_colors(self.label_colors)

    def show_prim(self):
        print("Showing primitive types")
        self.update_colors(self.prim_colors)

    def show_normals(self):
        print("Showing surface normals")
        self.update_colors(self.normal_colors)

    def on_layout(self, layout_context):
        rect = self.window.content_rect
        panel_width = 340
        self.scene_widget.frame = gui.Rect(
            rect.x, rect.y, rect.width - panel_width, rect.height
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
            self.scene_widget.setup_camera(
                60, self.bounds, self.bounds.get_center()
            )
            self.window.post_redraw()
