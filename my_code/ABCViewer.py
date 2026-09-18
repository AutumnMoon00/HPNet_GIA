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
        0: "Closed B-spline related",
        1: "Plane",
        2: "Open B-spline related",
        3: "Cone",
        4: "Cylinder",
        5: "Sphere",
        6: "Closed B-spline related",
        7: "Closed B-spline related",
        8: "Open B-spline related",
        9: "Closed B-spline related",
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
        self.pick_mode = False
        self.selected_point_index = None

        self.window = gui.Application.instance.create_window(
            "ABCParts Viewer", 1200, 800
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        render_scene = getattr(self.scene_widget.scene, "scene", None)
        self.mouse_picking_supported = (
            hasattr(self.scene_widget, "set_on_mouse")
            and render_scene is not None
            and hasattr(render_scene, "render_to_depth_image")
            and hasattr(self.scene_widget.scene, "camera")
        )
        if self.mouse_picking_supported:
            self.scene_widget.set_on_mouse(self.on_mouse)

        self.material = self._create_point_material()
        self.selection_material = self._create_point_material()
        self.selection_material.point_size = 12.0
        self.scene_widget.scene.add_geometry(
            self._GEOMETRY_NAME, self.point_cloud, self.material
        )

        bounds = self.point_cloud.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        em = self.window.theme.font_size
        self.panel = gui.Vert(
            0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)
        )
        self.panel.add_child(gui.Label("Color by"))

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

        if self.mouse_picking_supported:
            self.pick_button = gui.Button("Pick point")
            self.panel.add_child(self.pick_button)
            self.selection_info = gui.Label(
                "Select 'Pick point', then click a visible point."
            )
        else:
            self.panel.add_child(gui.Label("Point index"))
            self.point_index_slider = gui.Slider(gui.Slider.INT)
            self.point_index_slider.set_limits(0, len(self.points) - 1)
            self.point_index_slider.int_value = 0
            self.panel.add_child(self.point_index_slider)
            self.selection_info = gui.Label(
                "Use the point-index slider to inspect a point."
            )
        self.panel.add_child(self.selection_info)

        self.label_button.set_on_clicked(self.show_labels)
        self.prim_button.set_on_clicked(self.show_prim)
        self.normal_button.set_on_clicked(self.show_normals)
        self.point_size_slider.set_on_value_changed(self.set_point_size)
        if self.mouse_picking_supported:
            self.pick_button.set_on_clicked(self.toggle_pick_mode)
        else:
            self.point_index_slider.set_on_value_changed(self.show_point_by_index)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)

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
        self.selection_material.point_size = max(12.0, 2.0 * float(size))
        self._refresh_geometry()
        if self.selected_point_index is not None:
            self._refresh_selected_point()

    def show_point_by_index(self, point_index):
        self._show_selected_point(int(point_index))

    def toggle_pick_mode(self):
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.pick_button.text = "Stop picking"
            self.selection_info.text = "Click a visible point to inspect it."
            self.scene_widget.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
        else:
            self.pick_button.text = "Pick point"
            self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

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

        self.scene_widget.scene.scene.render_to_depth_image(
            lambda depth_image: self._select_depth_pixel(
                depth_image, click_x, click_y, frame.width, frame.height
            )
        )
        return gui.SceneWidget.EventCallbackResult.HANDLED

    def _select_depth_pixel(self, depth_image, click_x, click_y, view_width, view_height):
        depth = np.asarray(depth_image)
        height, width = depth.shape[:2]
        pixel_x = min(width - 1, max(0, int(click_x * width / view_width)))
        pixel_y = min(height - 1, max(0, int(click_y * height / view_height)))
        z = float(depth[pixel_y, pixel_x])
        if not np.isfinite(z) or z >= 1.0:
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self._show_selection_message("No point at that location.")
            )
            return

        world_point = self.scene_widget.scene.camera.unproject(
            pixel_x, pixel_y, z, width, height
        )
        point_index = int(np.argmin(np.sum((self.points - world_point) ** 2, axis=1)))
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self._show_selected_point(point_index)
        )

    def _show_selection_message(self, message):
        self.selection_info.text = message
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
            return "T_param: not available for B-spline-related surfaces."

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

        selected_point = o3d.geometry.PointCloud()
        selected_point.points = o3d.utility.Vector3dVector(
            self.points[point_index : point_index + 1]
        )
        selected_point.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
        self.selected_point = selected_point
        self.selected_point_index = point_index
        self.scene_widget.scene.add_geometry(
            self._SELECTION_GEOMETRY_NAME,
            self.selected_point,
            self.selection_material,
        )

    def _refresh_selected_point(self):
        self._highlight_selected_point(self.selected_point_index)

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
