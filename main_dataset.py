"""
PPE Dataset Analysis Pipeline - Main Entry Point

This script analyzes extracted frames to determine if they are suitable
for training PPE (Personal Protective Equipment) object detection models.

Usage:
    python -m main_dataset                    # Interactive mode
    python -m main_dataset --gui              # GUI mode (folder picker)
    python -m main_dataset <source_folder>    # Direct mode
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.config import EXTRACTED_DIR
from Scripts.groq_client import GroqVisionClient
from Scripts.dataset_analyzer import analyze_and_organize_frames, restore_not_suitable_frames
from Scripts.gui_utils import select_folder


def print_banner():
    """Print application banner."""
    print()
    print("=" * 60)
    print(" PPE DATASET SUITABILITY ANALYZER")
    print(" Filter images suitable for PPE object detection training")
    print("=" * 60)
    print()


def list_available_folders() -> list[str]:
    """List available folders in Extracted directory."""
    folders = []
    if os.path.exists(EXTRACTED_DIR):
        for folder in os.listdir(EXTRACTED_DIR):
            folder_path = os.path.join(EXTRACTED_DIR, folder)
            if os.path.isdir(folder_path):
                # Count jpg files
                jpgs = [f for f in os.listdir(folder_path) 
                       if f.endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))]
                if jpgs:
                    folders.append((folder, len(jpgs)))
    return folders


def get_source_folder() -> str:
    """Get source folder from user input."""
    folders = list_available_folders()
    
    if not folders:
        print(f"No folders with images found in: {EXTRACTED_DIR}")
        return None
    
    print("Available folders in Extracted/:\n")
    for i, (folder, count) in enumerate(folders, 1):
        print(f"  {i}. {folder} ({count} images)")
    
    print()
    print("Enter folder number, name, or full path:")
    print()
    
    while True:
        choice = input("Folder (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return None
        
        # Check if it's a number (selection from list)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                folder_name = folders[idx][0]
                return os.path.join(EXTRACTED_DIR, folder_name)
        except ValueError:
            pass
        
        # Check if it's a full path
        if os.path.isdir(choice):
            return choice
        
        # Check if it's a folder name in Extracted/
        potential_path = os.path.join(EXTRACTED_DIR, choice)
        if os.path.isdir(potential_path):
            return potential_path
        
        print(f"Invalid selection: {choice}")
        print("Please enter a valid folder number, name, or full path")


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ('y', 'yes')


def main():
    """Main entry point with interactive mode."""
    print_banner()
    
    # Get source folder
    print("STEP 1: SELECT SOURCE FOLDER")
    print("-" * 40)
    
    source_folder = get_source_folder()
    if not source_folder:
        print("\nExiting...")
        return
    
    # Count frames
    frame_count = len([f for f in os.listdir(source_folder) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(source_folder, f))])
    
    print(f"\nSelected: {source_folder}")
    print(f"Images to analyze: {frame_count}")
    
    if not confirm_action("\nProceed with PPE dataset suitability analysis?"):
        print("Aborted.")
        return
    
    # Run analysis
    print("\n" + "STEP 2: ANALYZING IMAGES")
    print("-" * 40)
    
    client = GroqVisionClient()
    stats = analyze_and_organize_frames(source_folder, client)
    
    # Final summary
    print("=" * 60)
    print(" ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"""
Summary:
  Total analyzed: {stats['total']}
  Suitable for dataset: {stats['suitable']}
  Not suitable: {stats['not_suitable']} (moved to not-suitable/)
  Errors: {stats['errors']}
  
Output location: {source_folder}
""")


def gui_mode(dry_run: bool = False):
    """
    GUI mode - select source folder using file dialog.
    
    Args:
        dry_run: If True, only print what would be done
    """
    print_banner()
    print("GUI MODE - Select folder using dialog window\n")
    
    print("Select folder containing extracted frames...")
    source_folder = select_folder(
        title="Select Folder with Extracted Frames",
        initial_dir=EXTRACTED_DIR if os.path.exists(EXTRACTED_DIR) else None
    )
    
    if not source_folder:
        print("No folder selected. Exiting.")
        return
    
    print(f"Selected: {source_folder}")
    
    # Count frames
    frame_count = len([f for f in os.listdir(source_folder) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(source_folder, f))])
    
    if frame_count == 0:
        print(f"No .jpg images found in: {source_folder}")
        return
    
    print(f"Images to analyze: {frame_count}")
    print(f"Dry run: {dry_run}")
    
    if not confirm_action("\nProceed with PPE dataset suitability analysis?"):
        print("Aborted.")
        return
    
    client = GroqVisionClient()
    stats = analyze_and_organize_frames(source_folder, client, dry_run=dry_run)
    
    print(f"\nDone! Suitable: {stats['suitable']}, Not suitable: {stats['not_suitable']}")


def cli():
    """Command-line interface with arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze images for PPE dataset suitability"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Source folder containing images to analyze"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use GUI mode with folder picker dialog"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't move files"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore files from not-suitable folder back to main folder"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle restore mode
    if args.restore:
        if not args.source:
            print("Error: --restore requires a source folder")
            return
        
        source = args.source
        if not os.path.isabs(source):
            source = os.path.join(EXTRACTED_DIR, source)
        
        if not os.path.isdir(source):
            print(f"Error: Folder not found: {source}")
            return
        
        restore_not_suitable_frames(source)
        return
    
    # GUI mode
    if args.gui:
        gui_mode(dry_run=args.dry_run)
        return
    
    # Interactive mode if no source specified
    if not args.source:
        main()
        return
    
    # Direct mode with source argument
    source = args.source
    if not os.path.isabs(source):
        source = os.path.join(EXTRACTED_DIR, source)
    
    if not os.path.isdir(source):
        print(f"Error: Folder not found: {source}")
        return
    
    frame_count = len([f for f in os.listdir(source) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(source, f))])
    
    print(f"Source: {source}")
    print(f"Images: {frame_count}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    client = GroqVisionClient()
    stats = analyze_and_organize_frames(source, client, dry_run=args.dry_run)
    
    print(f"Done! Suitable: {stats['suitable']}, Not suitable: {stats['not_suitable']}")


if __name__ == "__main__":
    cli()
