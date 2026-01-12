"""
PPE Dataset Analysis Pipeline - Main Entry Point

Analyze extracted frames to determine if they are suitable
for training PPE (Personal Protective Equipment) object detection models.

Usage:
    python -m main_dataset
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.groq_client import GroqVisionClient
from Scripts.dataset_analyzer import analyze_and_organize_frames
from Scripts.gui_utils import select_folder


def print_banner():
    """Print application banner."""
    print()
    print("=" * 60)
    print(" PPE DATASET SUITABILITY ANALYZER")
    print(" Filter images suitable for PPE object detection training")
    print("=" * 60)
    print()


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ('y', 'yes')


def main():
    """Main entry point - GUI mode for folder selection."""
    print_banner()
    
    # Select source folder
    print("Select folder containing extracted frames...")
    source_folder = select_folder(
        title="Select Folder with Extracted Frames"
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
    
    if not confirm_action("\nProceed with PPE dataset suitability analysis?"):
        print("Aborted.")
        return
    
    # Run analysis
    print("\n" + "=" * 60)
    print("ANALYZING IMAGES")
    print("=" * 60)
    
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


if __name__ == "__main__":
    main()
