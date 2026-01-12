"""
Frame Extraction Module
Extracts frames from videos at specified intervals.
"""
import cv2
import os
from datetime import datetime

try:
    from .config import FRAME_INTERVAL_SECONDS
    from .utils import format_duration, ensure_dir, sanitize_folder_name
except ImportError:
    from config import FRAME_INTERVAL_SECONDS
    from utils import format_duration, ensure_dir, sanitize_folder_name


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    prefix: str,
    timestamp: str,
    start_index: int,
    frame_interval_sec: float = FRAME_INTERVAL_SECONDS
) -> int:
    """
    Extract frames from a single video at specified interval.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        prefix: Prefix for frame filenames (input folder name)
        timestamp: Timestamp string for filenames
        start_index: Starting index for frame numbering
        frame_interval_sec: Extract one frame every N seconds
        
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate frame interval based on FPS and desired seconds
    frame_interval = int(fps * frame_interval_sec)
    if frame_interval < 1:
        frame_interval = 1
    
    video_name = os.path.basename(video_path)
    print(f"  Processing: {video_name}")
    print(f"    FPS: {fps:.2f}, Duration: {format_duration(duration)}, Interval: every {frame_interval_sec}s")
    
    count = 0
    extracted_count = start_index
    frames_from_this_video = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            # Format: {prefix}_{timestamp}_{index}.jpg
            frame_filename = os.path.join(output_dir, f"{prefix}_{timestamp}_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            frames_from_this_video += 1
        
        count += 1
    
    cap.release()
    print(f"    Extracted {frames_from_this_video} frames")
    
    return frames_from_this_video


def extract_frames_from_videos(
    video_paths: list[str],
    output_dir: str,
    input_folder_name: str = None,
    frame_interval_sec: float = FRAME_INTERVAL_SECONDS
) -> str:
    """
    Extract frames from multiple videos into a single output folder.
    
    Frame naming format: {InputFolderName}_{Timestamp}_{Index}.jpg
    Timestamp format: MMHHDDMMYYYY (minute, hour, day, month, year)
    
    Args:
        video_paths: List of video file paths
        output_dir: Directory to save extracted frames
        input_folder_name: Name to use as prefix for frame files
        frame_interval_sec: Extract one frame every N seconds
        
    Returns:
        Path to output folder containing all extracted frames
    """
    if not video_paths:
        print("No videos to process")
        return None
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Generate timestamp at program start: MMHHDDMMYYYY
    now = datetime.now()
    timestamp = now.strftime("%M%H%d%m%Y")  # minute, hour, day, month, year
    
    # Sanitize prefix for filename
    if input_folder_name:
        prefix = sanitize_folder_name(input_folder_name)
    else:
        # Use first video's parent folder name
        first_video_dir = os.path.dirname(video_paths[0])
        prefix = sanitize_folder_name(os.path.basename(first_video_dir) or "frames")
    
    print(f"\n{'='*60}")
    print(f"FRAME EXTRACTION")
    print(f"{'='*60}")
    print(f"Output folder: {output_dir}")
    print(f"Videos to process: {len(video_paths)}")
    print(f"Frame interval: {frame_interval_sec} seconds")
    print(f"Frame prefix: {prefix}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}\n")
    
    total_frames = 0
    current_index = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}]", end="")
        frames = extract_frames_from_video(
            video_path, output_dir, prefix, timestamp, current_index, frame_interval_sec
        )
        total_frames += frames
        current_index += frames
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output folder: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir
