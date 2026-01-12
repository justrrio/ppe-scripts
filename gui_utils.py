"""
GUI Utilities for folder selection using tkinter.
"""
import os
import tkinter as tk
from tkinter import filedialog


def select_folder(title: str = "Select Folder", initial_dir: str = None) -> str:
    """
    Open a folder selection dialog.
    
    Args:
        title: Dialog window title
        initial_dir: Initial directory to show
        
    Returns:
        Selected folder path, or empty string if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    folder = filedialog.askdirectory(
        title=title,
        initialdir=initial_dir or os.getcwd()
    )
    
    root.destroy()
    return folder


def select_file(
    title: str = "Select File",
    initial_dir: str = None,
    filetypes: list[tuple[str, str]] = None
) -> str:
    """
    Open a file selection dialog.
    
    Args:
        title: Dialog window title
        initial_dir: Initial directory to show
        filetypes: List of (description, pattern) tuples
        
    Returns:
        Selected file path, or empty string if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if filetypes is None:
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mkv *.mov"),
            ("All files", "*.*")
        ]
    
    file = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir or os.getcwd(),
        filetypes=filetypes
    )
    
    root.destroy()
    return file


if __name__ == "__main__":
    # Test the GUI utilities
    print("Testing folder selection...")
    folder = select_folder("Select a folder for testing")
    print(f"Selected folder: {folder}")
    
    print("\nTesting file selection...")
    file = select_file("Select a video file")
    print(f"Selected file: {file}")
