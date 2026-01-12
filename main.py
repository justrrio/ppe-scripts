"""
PPE Frame Extraction & Analysis Pipeline

Extract frames from videos and analyze them for PPE dataset suitability.
Unsuitable images are automatically moved to 'not-suitable/' folder.

Usage:
    python main.py
"""
import os

from config import FRAME_INTERVAL_SECONDS
from frame_extractor import extract_frames_from_videos
from groq_client import GroqVisionClient
from image_analyzer import analyze_and_filter_frames
from gui_utils import select_folder
from utils import sanitize_folder_name


def print_banner():
    """Print application banner."""
    print()
    print("=" * 60)
    print(" PPE FRAME EXTRACTION & ANALYSIS PIPELINE")
    print(" Extract frames and filter for PPE dataset training")
    print("=" * 60)
    print()


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ('y', 'yes')


def main(skip_analysis: bool = False):
    """
    Main entry point - GUI mode for folder selection.
    
    Args:
        skip_analysis: If True, skip PPE suitability analysis
    """
    print_banner()
    
    # Step 1: Select input folder
    print("Step 1: Select INPUT folder containing videos...")
    input_folder = select_folder(
        title="Select Input Folder (containing videos)"
    )
    
    if not input_folder:
        print("No input folder selected. Exiting.")
        return
    
    print(f"Input folder: {input_folder}")
    
    # List videos in folder
    videos = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ])
    
    if not videos:
        print(f"No video files found in: {input_folder}")
        return
    
    print(f"Found {len(videos)} video files")
    
    # Step 2: Select output folder
    print("\nStep 2: Select OUTPUT folder for extracted frames...")
    output_folder = select_folder(
        title="Select Output Folder (for extracted frames)"
    )
    
    if not output_folder:
        print("No output folder selected. Exiting.")
        return
    
    print(f"Output folder: {output_folder}")
    
    # Create output subfolder based on input folder name
    input_folder_name = os.path.basename(input_folder)
    output_name = sanitize_folder_name(input_folder_name or "extracted")
    final_output_dir = os.path.join(output_folder, output_name)
    
    print(f"Frames will be saved to: {final_output_dir}")
    
    # Confirm
    if not confirm_action("\nProceed with frame extraction?"):
        print("Aborted.")
        return
    
    # Step 3: Extract frames
    print("\n" + "=" * 60)
    print("EXTRACTING FRAMES")
    print("=" * 60)
    
    frames_dir = extract_frames_from_videos(
        videos,
        output_dir=final_output_dir,
        frame_interval_sec=FRAME_INTERVAL_SECONDS
    )
    
    if not frames_dir or not os.path.exists(frames_dir):
        print("Frame extraction failed!")
        return
    
    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"\nExtracted {frame_count} frames to: {frames_dir}")
    
    # Step 4: PPE Suitability Analysis (optional)
    if not skip_analysis:
        if not confirm_action("\nProceed with PPE suitability analysis using Groq AI?"):
            print("Frame extraction complete. Analysis skipped.")
            return
        
        client = GroqVisionClient()
        stats = analyze_and_filter_frames(frames_dir, client)
        
        print("\n" + "=" * 60)
        print(" PIPELINE COMPLETE")
        print("=" * 60)
        print(f"""
Summary:
  Videos processed: {len(videos)}
  Frames extracted: {frame_count}
  Suitable for dataset: {stats['suitable']}
  Not suitable: {stats['not_suitable']} (moved to not-suitable/)
  Errors: {stats['errors']}
  
Output location: {frames_dir}
""")
    else:
        print(f"\nExtraction complete! {frame_count} frames saved to {frames_dir}")


if __name__ == "__main__":
    main()
