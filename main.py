"""
PPE Frame Extraction Pipeline - Main Entry Point

This script provides the main command-line interface for:
1. Collecting similar videos by filename pattern
2. Extracting frames at specified intervals
3. Detecting humans in frames using Groq AI
4. Organizing frames based on human presence
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.config import RAW_VIDEO_DIR, FRAME_INTERVAL_SECONDS
from Scripts.video_collector import collect_similar_videos, get_video_info, list_available_prefixes
from Scripts.frame_extractor import extract_frames_from_videos
from Scripts.groq_client import GroqVisionClient
from Scripts.human_detector import detect_and_organize_frames


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


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        reference_file = sys.argv[1]
        skip_detect = "--skip-detection" in sys.argv
        quick_run(reference_file, skip_detection=skip_detect)
    else:
        main()
