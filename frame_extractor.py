"""
Frame Extraction Module
Extracts frames from videos at specified intervals.
"""
import cv2
import os

try:
    from .config import FRAME_INTERVAL_SECONDS
    from .utils import format_duration, ensure_dir
except ImportError:
    from config import FRAME_INTERVAL_SECONDS
    from utils import format_duration, ensure_dir


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval_sec: float = FRAME_INTERVAL_SECONDS
) -> int:
    """
    Extract frames from a single video at specified interval.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
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
    
    # Get existing frame count in output dir for numbering continuation
    existing_frames = len([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    
    video_name = os.path.basename(video_path)
    print(f"  Processing: {video_name}")
    print(f"    FPS: {fps:.2f}, Duration: {format_duration(duration)}, Interval: every {frame_interval_sec}s")
    
    count = 0
    extracted_count = existing_frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        count += 1
    
    cap.release()
    frames_from_this_video = extracted_count - existing_frames
    print(f"    Extracted {frames_from_this_video} frames")
    
    return frames_from_this_video


def extract_frames_from_videos(
    video_paths: list[str],
    output_dir: str,
    frame_interval_sec: float = FRAME_INTERVAL_SECONDS
) -> str:
    """
    Extract frames from multiple videos into a single output folder.
    
    Args:
        video_paths: List of video file paths
        output_dir: Directory to save extracted frames
        frame_interval_sec: Extract one frame every N seconds
        
    Returns:
        Path to output folder containing all extracted frames
    """
    if not video_paths:
        print("No videos to process")
        return None
    
    # Create output directory
    ensure_dir(output_dir)
    
    print(f"\n{'='*60}")
    print(f"FRAME EXTRACTION")
    print(f"{'='*60}")
    print(f"Output folder: {output_dir}")
    print(f"Videos to process: {len(video_paths)}")
    print(f"Frame interval: {frame_interval_sec} seconds")
    print(f"{'='*60}\n")
    
    total_frames = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}]", end="")
        frames = extract_frames_from_video(video_path, output_dir, frame_interval_sec)
        total_frames += frames
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output folder: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir
