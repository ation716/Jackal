# -*- coding: utf-8 -*-
"""
alert_tem.py — Lightweight Tkinter dialog utilities.

Features:
  - SimpleDialog.auto_close : show a non-blocking popup that closes automatically
                               after a given number of seconds; supports an optional
                               callback executed on close.
  - SimpleDialog.confirm    : show a blocking yes/no confirmation dialog and return
                               the user's choice as a bool.
"""

from typing import Optional, Callable
import tkinter as tk
from tkinter import messagebox
import threading
import time


class SimpleDialog:
    """Lightweight dialog helper built on tkinter."""

    @staticmethod
    def auto_close(message: str,
                   title: str = "Notice",
                   seconds: int = 5,
                   on_close: Optional[Callable] = None):
        """
        Show a non-blocking popup that closes automatically after *seconds*.

        Runs in a daemon thread so it does not block the caller.

        Parameters
        ----------
        message  : text displayed in the popup body
        title    : window title bar text
        seconds  : how many seconds before the window auto-closes
        on_close : optional callback invoked just before the window is destroyed
        """

        def show():
            root = tk.Tk()
            root.title(title)
            root.geometry("600x300")

            tk.Label(root, text=message, pady=30).pack()
            tk.Label(root, text=f"Closing in {seconds} second(s)...").pack()

            def close():
                if on_close:
                    on_close()
                root.destroy()

            root.after(seconds * 1000, close)
            root.mainloop()

        threading.Thread(target=show, daemon=True).start()

    @staticmethod
    def confirm(message: str,
                title: str = "Confirm") -> bool:
        """
        Show a blocking yes/no confirmation dialog.

        Returns True if the user clicks Yes, False otherwise.

        Example:
            if SimpleDialog.confirm("Delete this item?"):
                delete_item()
        """
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno(title, message)
        root.destroy()
        return result


if __name__ == "__main__":
    SimpleDialog.auto_close("Long-running popup test", "Notice", 30)
    time.sleep(40)
