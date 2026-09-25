"""Open the system H5 file dialog outside the Open3D GUI process."""

import sys
import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            parent=root,
            title="Select an H5 model",
            initialdir=sys.argv[1],
            filetypes=[("H5 files", "*.h5 *.hdf5")],
        )
    finally:
        root.destroy()

    if path:
        sys.stdout.buffer.write(path.encode("utf-8"))


if __name__ == "__main__":
    main()
