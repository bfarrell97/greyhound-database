"""
Greyhound Racing Analysis System
Main entry point to launch the GUI application.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run GUI
from src.gui.app import GreyhoundRacingApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    # Start reconcile of live bets in background on startup
    try:
        import threading
        from scripts.reconcile_live_bets import reconcile_bets
        print("Starting reconcile of live bets in background...")
        threading.Thread(target=reconcile_bets, daemon=True).start()
    except Exception as e:
        print(f"Failed to start reconcile on startup: {e}")

    app = GreyhoundRacingApp(root)
    root.mainloop()
