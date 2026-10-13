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

if __package__:
    from .ABCViewer import ABCViewer, categorical_colors
else:
    from ABCViewer import ABCViewer, categorical_colors


DATASET_DIR = Path(r"C:\ABC_final")


def load_part(filepath):
    """Load and validate the point-cloud arrays required by the viewer."""
    required_keys = ("points", "normals", "labels", "prim", "T_param")
    with h5py.File(filepath, "r") as h5_file:
        missing_keys = [key for key in required_keys if key not in h5_file]
        if missing_keys:
            raise KeyError(f"{filepath} is missing dataset(s): {', '.join(missing_keys)}")

        print("Keys:", list(h5_file.keys()))
        for key in h5_file.keys():
            data = h5_file[key]
            print(f"{key:10s}", "shape =", data.shape, "dtype =", data.dtype)

        points = np.asarray(h5_file["points"][:], dtype=np.float64)
        normals = np.asarray(h5_file["normals"][:], dtype=np.float64)
        t_param = np.asarray(h5_file["T_param"][:], dtype=np.float64)
        labels = np.asarray(h5_file["labels"][:]).reshape(-1)
        prim = np.asarray(h5_file["prim"][:]).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("points must contain finite 3D coordinates.")
    if normals.shape != points.shape:
        raise ValueError(
            f"normals must have shape {points.shape}, got {normals.shape}."
        )
    if len(labels) != len(points) or len(prim) != len(points):
        raise ValueError(
            "labels and prim must contain one value per point "
            f"({len(points)}); got {len(labels)} and {len(prim)}."
        )
    return points, normals, labels, prim, t_param


def main():
    app = gui.Application.instance
    app.initialize()
    chooser = app.create_window("ABCParts Viewer", 1200, 800)
    layout = gui.Vert(12, gui.Margins(20, 20, 20, 20))
    layout.add_child(gui.Label("Choose how to open a model"))
    status_label = gui.Label("")
    layout.add_child(status_label)
    active_viewer = None

    def open_model(filepath):
        nonlocal active_viewer
        try:
            print(f"Selected: {filepath.name}")
            points, normals, labels, prim, t_param = load_part(filepath)

            print("\nNumber of points:", len(points))
            print("Primitive instances:", np.unique(labels))
            print("Number of instances:", len(np.unique(labels)))
            print("Number of primitives:", len(np.unique(prim)))
            for label in np.unique(labels):
                mask = labels == label
                print(
                    f"Instance {label}: {mask.sum()} points, "
                    f"primitive type = {np.unique(prim[mask])}"
                )

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.normals = o3d.utility.Vector3dVector(normals)

            label_colors = categorical_colors(labels, seed=42)
            prim_colors = categorical_colors(prim, seed=10)
            normal_colors = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
            point_cloud.colors = o3d.utility.Vector3dVector(label_colors)

            active_viewer = ABCViewer(
                point_cloud,
                label_colors,
                prim_colors,
                normal_colors,
                points,
                normals,
                labels,
                prim,
                t_param,
                model_name=filepath.name,
                window=chooser,
            )
            layout.visible = False
            chooser.set_needs_layout()
        except Exception as error:
            traceback.print_exc()
            status_label.text = f"Could not open {filepath.name}: {error}"
            chooser.set_needs_layout()
            if hasattr(chooser, "show_message_box"):
                chooser.show_message_box("Could not open model", str(error))

    def choose_random():
        try:
            files = [
                path
                for path in DATASET_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
            ]
        except OSError as error:
            status_label.text = f"Could not read {DATASET_DIR}: {error}"
            chooser.post_redraw()
            return
        if not files:
            status_label.text = f"No H5 files found in {DATASET_DIR}"
            chooser.post_redraw()
            return
        open_model(random.choice(files))

    def choose_file():
        file_button.enabled = False
        status_label.text = "Opening file explorer..."
        chooser.set_needs_layout()

        def run_file_picker():
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("native_file_picker.py")),
                        str(DATASET_DIR),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                path = result.stdout.decode("utf-8", errors="replace")
                error = result.stderr.decode("utf-8", errors="replace")
                failed = result.returncode != 0
            except Exception as exception:
                path = ""
                error = str(exception)
                failed = True

            def finish_selection():
                file_button.enabled = True
                if failed:
                    status_label.text = f"Could not open file explorer: {error}"
                    chooser.set_needs_layout()
                elif path:
                    status_label.text = ""
                    open_model(Path(path))
                else:
                    status_label.text = ""
                    chooser.set_needs_layout()

            app.post_to_main_thread(chooser, finish_selection)

        threading.Thread(target=run_file_picker, daemon=True).start()

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
    app.run()


if __name__ == "__main__":
    main()
