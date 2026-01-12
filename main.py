"""
PPE Frame Extraction Pipeline - Main Entry Point

This script provides the main command-line interface for:
1. Collecting similar videos by filename pattern
2. Extracting frames at specified intervals
3. Detecting humans in frames using Groq AI
4. Organizing frames based on human presence

Usage:
    python -m main                           # Interactive mode
    python -m main --gui                     # GUI mode (folder picker)
    python -m main "video_reference.mp4"     # Quick mode
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.config import RAW_VIDEO_DIR, FRAME_INTERVAL_SECONDS, EXTRACTED_DIR
from Scripts.video_collector import collect_similar_videos, get_video_info, list_available_prefixes
from Scripts.frame_extractor import extract_frames_from_videos
from Scripts.groq_client import GroqVisionClient
from Scripts.human_detector import detect_and_organize_frames
from Scripts.gui_utils import select_folder


def print_banner():
    """Print application banner."""
    print()
    print("=" * 60)
    print(" PPE FRAME EXTRACTION PIPELINE")
    print(" Extract frames and detect human presence using Groq AI")
    print("=" * 60)
    print()


def get_user_input_video() -> str:
    """
    Get video reference from user.
    
    Returns:
        Selected video filename
    """
    print("Available video prefixes in source directory:\n")
    
    prefixes = list_available_prefixes()
    if not prefixes:
        print("No videos found in source directory!")
        print(f"Directory: {RAW_VIDEO_DIR}")
        return None
    
    for i, prefix in enumerate(prefixes, 1):
        print(f"  {i}. {prefix}")
    
    print()
    print("Enter a video filename to process videos with the same prefix.")
    print("Example: Winposh Regent_Camera 01_20251215110139_20251215110636.mp4")
    print()
    
    while True:
        filename = input("Video filename (or 'q' to quit): ").strip()
        
        if filename.lower() == 'q':
            return None
        
        if not filename:
            print("Please enter a filename")
            continue
        
        # Add .mp4 if missing
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
        
        return filename


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ('y', 'yes')


def main():
    """Main entry point for the pipeline."""
    print_banner()
    
    # Step 1: Get video reference from user
    print("STEP 1: SELECT VIDEO REFERENCE")
    print("-" * 40)
    
    reference = get_user_input_video()
    if not reference:
        print("\nExiting...")
        return
    
    # Step 2: Collect similar videos
    print("\n" + "STEP 2: COLLECTING VIDEOS")
    print("-" * 40)
    
    videos = collect_similar_videos(reference)
    info = get_video_info(videos)
    
    if not videos:
        print(f"\nNo videos found matching: {reference}")
        return
    
    print(f"\nFound {info['count']} matching videos ({info['total_size_mb']} MB total):")
    for v in info['videos'][:10]:  # Show first 10
        print(f"  - {v}")
    if len(info['videos']) > 10:
        print(f"  ... and {len(info['videos']) - 10} more")
    
    if not confirm_action("\nProceed with frame extraction?"):
        print("Aborted.")
        return
    
    # Step 3: Extract frames
    print("\n" + "STEP 3: EXTRACTING FRAMES")
    print("-" * 40)
    
    frames_dir = extract_frames_from_videos(
        videos,
        frame_interval_sec=FRAME_INTERVAL_SECONDS
    )
    
    if not frames_dir or not os.path.exists(frames_dir):
        print("Frame extraction failed!")
        return
    
    # Count extracted frames
    frame_count = len([f for f in os.listdir(frames_dir) 
                       if f.endswith('.jpg') and os.path.isfile(os.path.join(frames_dir, f))])
    
    print(f"\nExtracted {frame_count} frames to: {frames_dir}")
    
    if not confirm_action("\nProceed with human detection using Groq AI?"):
        print("Frame extraction complete. Human detection skipped.")
        return
    
    # Step 4: Human detection
    print("\n" + "STEP 4: DETECTING HUMANS")
    print("-" * 40)
    
    client = GroqVisionClient()
    stats = detect_and_organize_frames(frames_dir, client)
    
    # Final summary
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"""
Summary:
  Videos processed: {info['count']}
  Frames extracted: {stats['total']}
  Frames with humans: {stats['with_human']}
  Frames without humans: {stats['no_human']} (moved to no-human/)
  Errors: {stats['errors']}
  
Output location: {frames_dir}
""")


def quick_run(reference: str, skip_detection: bool = False):
    """
    Quick run mode without user prompts.
    
    Args:
        reference: Video reference filename
        skip_detection: If True, skip human detection step
    """
    print_banner()
    
    print(f"Quick run mode with reference: {reference}\n")
    
    # Collect videos
    videos = collect_similar_videos(reference)
    if not videos:
        print(f"No videos found matching: {reference}")
        return
    
    info = get_video_info(videos)
    print(f"Found {info['count']} videos ({info['total_size_mb']} MB)")
    
    # Extract frames
    frames_dir = extract_frames_from_videos(videos)
    
    if not frames_dir:
        print("Frame extraction failed!")
        return
    
    # Human detection (optional)
    if not skip_detection:
        client = GroqVisionClient()
        stats = detect_and_organize_frames(frames_dir, client)
        print(f"\nComplete! {stats['with_human']} with humans, {stats['no_human']} without")
    else:
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        print(f"\nExtraction complete! {frame_count} frames saved to {frames_dir}")


def gui_mode(skip_detection: bool = False):
    """
    GUI mode - select input folder and output folder using file dialogs.
    
    Args:
        skip_detection: If True, skip human detection step
    """
    print_banner()
    print("GUI MODE - Select folders using dialog windows\n")
    
    # Select input folder
    print("Step 1: Select INPUT folder containing videos...")
    input_folder = select_folder(
        title="Select Input Folder (containing videos)",
        initial_dir=RAW_VIDEO_DIR if os.path.exists(RAW_VIDEO_DIR) else None
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
    
    # Select output folder
    print("\nStep 2: Select OUTPUT folder for extracted frames...")
    output_folder = select_folder(
        title="Select Output Folder (for extracted frames)",
        initial_dir=EXTRACTED_DIR if os.path.exists(EXTRACTED_DIR) else None
    )
    
    if not output_folder:
        print("No output folder selected. Exiting.")
        return
    
    print(f"Output folder: {output_folder}")
    
    # Get output folder name from first video
    first_video = os.path.basename(videos[0])
    from Scripts.utils import get_video_prefix, sanitize_folder_name
    prefix = get_video_prefix(first_video)
    output_name = sanitize_folder_name(prefix or "extracted")
    final_output_dir = os.path.join(output_folder, output_name)
    
    print(f"Frames will be saved to: {final_output_dir}")
    
    # Confirm
    if not confirm_action("\nProceed with frame extraction?"):
        print("Aborted.")
        return
    
    # Extract frames
    frames_dir = extract_frames_from_videos(
        videos,
        output_folder_name=output_name,
        frame_interval_sec=FRAME_INTERVAL_SECONDS
    )
    
    # Override output directory to use selected folder
    import shutil
    if frames_dir and frames_dir != final_output_dir:
        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)
        shutil.move(frames_dir, final_output_dir)
        frames_dir = final_output_dir
    
    if not frames_dir or not os.path.exists(frames_dir):
        print("Frame extraction failed!")
        return
    
    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"\nExtracted {frame_count} frames to: {frames_dir}")
    
    # Human detection
    if not skip_detection:
        if not confirm_action("\nProceed with human detection using Groq AI?"):
            print("Frame extraction complete. Human detection skipped.")
            return
        
        client = GroqVisionClient()
        stats = detect_and_organize_frames(frames_dir, client)
        print(f"\nComplete! {stats['with_human']} with humans, {stats['no_human']} without")
    else:
        print(f"\nExtraction complete! {frame_count} frames saved to {frames_dir}")


def cli():
    """Command-line interface with argparse."""
    parser = argparse.ArgumentParser(
        description="PPE Frame Extraction Pipeline - Extract and analyze video frames"
    )
    parser.add_argument(
        "reference",
        nargs="?",
        help="Video reference filename for batch processing"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use GUI mode with folder picker dialogs"
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip human detection step (extraction only)"
    )
    
    args = parser.parse_args()
    
    if args.gui:
        gui_mode(skip_detection=args.skip_detection)
    elif args.reference:
        quick_run(args.reference, skip_detection=args.skip_detection)
    else:
        main()


if __name__ == "__main__":
    cli()

