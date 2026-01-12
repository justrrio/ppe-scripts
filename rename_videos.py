"""
Video Renaming Tool
Renames videos in a folder to the standard format:
{Location}_{Camera XX}_{StartTimestamp}_{EndTimestamp}.mp4

Usage:
    python -m Scripts.rename_videos
    python -m Scripts.rename_videos "path/to/folder" --location "Kapal1" --camera 1
"""
import os
import sys
import argparse
import cv2
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .config import RAW_VIDEO_DIR
except ImportError:
    from config import RAW_VIDEO_DIR


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        return frame_count / fps
    return 0


def generate_new_filename(
    location: str,
    camera_num: int,
    start_time: datetime,
    duration_seconds: float
) -> str:
    """Generate new filename in standard format."""
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    start_str = start_time.strftime("%Y%m%d%H%M%S")
    end_str = end_time.strftime("%Y%m%d%H%M%S")
    
    return f"{location}_Camera {camera_num:02d}_{start_str}_{end_str}.mp4"


def preview_renames(
    folder: str,
    location: str,
    camera_num: int,
    start_time: datetime
) -> list[tuple[str, str]]:
    """
    Preview the rename operations without executing them.
    
    Returns:
        List of tuples (old_name, new_name)
    """
    videos = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ])
    
    if not videos:
        print(f"No video files found in: {folder}")
        return []
    
    renames = []
    current_time = start_time
    
    for video in videos:
        video_path = os.path.join(folder, video)
        duration = get_video_duration(video_path)
        
        new_name = generate_new_filename(location, camera_num, current_time, duration)
        renames.append((video, new_name))
        
        # Next video starts where this one ends
        current_time += timedelta(seconds=duration)
    
    return renames


def execute_renames(folder: str, renames: list[tuple[str, str]]) -> int:
    """Execute the rename operations."""
    success_count = 0
    
    for old_name, new_name in renames:
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"  ✓ {old_name} → {new_name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {old_name}: Error - {e}")
    
    return success_count


def interactive_mode():
    """Run in interactive mode with user prompts."""
    print()
    print("=" * 60)
    print(" VIDEO RENAMING TOOL")
    print(" Rename videos to standard format for batch processing")
    print("=" * 60)
    print()
    
    # Get folder
    print(f"Default folder: {RAW_VIDEO_DIR}")
    folder_input = input("Folder path (Enter for default, or type path): ").strip()
    
    if folder_input:
        folder = folder_input
    else:
        folder = RAW_VIDEO_DIR
    
    if not os.path.isdir(folder):
        print(f"Error: Folder not found: {folder}")
        return
    
    # Count videos
    videos = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    print(f"\nFound {len(videos)} video files")
    
    if not videos:
        return
    
    # Get location
    print("\n" + "-" * 40)
    location = input("Location name (e.g., Kapal1, Dermaga2): ").strip()
    if not location:
        location = "Unknown"
    
    # Sanitize location (remove spaces, special chars)
    location = location.replace(" ", "_").replace("-", "_")
    
    # Get camera number
    camera_input = input("Camera number (default: 1): ").strip()
    try:
        camera_num = int(camera_input) if camera_input else 1
    except ValueError:
        camera_num = 1
    
    # Get start time
    print("\nStart time for first video:")
    print("  Format: YYYYMMDDHHMMSS (e.g., 20260112100000)")
    print("  Or press Enter for current time")
    time_input = input("Start time: ").strip()
    
    if time_input:
        try:
            start_time = datetime.strptime(time_input, "%Y%m%d%H%M%S")
        except ValueError:
            print("Invalid format, using current time")
            start_time = datetime.now()
    else:
        start_time = datetime.now()
    
    # Preview
    print("\n" + "=" * 60)
    print("PREVIEW")
    print("=" * 60)
    
    renames = preview_renames(folder, location, camera_num, start_time)
    
    if not renames:
        return
    
    print(f"\nThe following {len(renames)} files will be renamed:\n")
    for old_name, new_name in renames:
        print(f"  {old_name}")
        print(f"    → {new_name}")
        print()
    
    # Confirm
    confirm = input("Proceed with rename? (y/n): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Cancelled.")
        return
    
    # Execute
    print("\n" + "=" * 60)
    print("RENAMING")
    print("=" * 60 + "\n")
    
    success = execute_renames(folder, renames)
    
    print(f"\nComplete! {success}/{len(renames)} files renamed successfully.")
    
    if success > 0:
        # Show the command to run
        first_new_name = renames[0][1]
        print(f"\nTo process these videos, run:")
        print(f'  python -m main "{first_new_name}"')


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Rename videos to standard format for batch processing"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing videos to rename"
    )
    parser.add_argument(
        "--location", "-l",
        help="Location name (e.g., Kapal1)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=1,
        help="Camera number (default: 1)"
    )
    parser.add_argument(
        "--start-time", "-t",
        help="Start time in YYYYMMDDHHMMSS format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't rename"
    )
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments
    if not args.folder and not args.location:
        interactive_mode()
        return
    
    # CLI mode
    folder = args.folder or RAW_VIDEO_DIR
    
    if not os.path.isdir(folder):
        print(f"Error: Folder not found: {folder}")
        return
    
    if not args.location:
        print("Error: --location is required in CLI mode")
        return
    
    location = args.location.replace(" ", "_").replace("-", "_")
    
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, "%Y%m%d%H%M%S")
        except ValueError:
            print("Error: Invalid start time format. Use YYYYMMDDHHMMSS")
            return
    else:
        start_time = datetime.now()
    
    renames = preview_renames(folder, location, args.camera, start_time)
    
    if not renames:
        return
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Renaming {len(renames)} files:\n")
    
    if args.dry_run:
        for old_name, new_name in renames:
            print(f"  {old_name} → {new_name}")
    else:
        success = execute_renames(folder, renames)
        print(f"\nComplete! {success}/{len(renames)} files renamed.")


if __name__ == "__main__":
    cli()
