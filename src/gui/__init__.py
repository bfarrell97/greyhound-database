"""Graphical user interface (GUI) for the Greyhound Racing Analysis System.

This package contains the Tkinter-based GUI application:
- Main application window (5,700+ lines)
- Live Alpha Radar (top prospects)
- Active/Settled bet panels
- Manual bet placement controls
- Market monitoring and notifications

Modules:
    app: Main GUI application (GreyhoundApp class)
    temp_virtual_func: Temporary virtual function helpers

Example:
    >>> from src.gui.app import GreyhoundApp
    >>> import tkinter as tk
    >>> 
    >>> root = tk.Tk()
    >>> app = GreyhoundApp(root)
    >>> root.mainloop()
"""

__all__ = [
    'GreyhoundApp',
]
