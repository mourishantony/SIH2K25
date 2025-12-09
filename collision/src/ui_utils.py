from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from typing import Optional


def pick_video_file(title: str = "Select video file") -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path or None
