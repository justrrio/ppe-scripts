"""
Video Collection Module
Collects videos with matching prefix patterns from source directory.
"""
import os
import glob

try:
    from .utils import get_video_prefix
    from .config import RAW_VIDEO_DIR
except ImportError:
    from utils import get_video_prefix
    from config import RAW_VIDEO_DIR


def collect_similar_videos(reference_filename: str, source_dir: str = None) -> list[str]:
    """
    Collect all videos that have the same prefix as the reference video.
    
    Given a reference like "Winposh Regent_Camera 01_20251215110139_20251215110636.mp4",
    this will find all videos starting with "Winposh Regent_Camera 01_".
    
    Args:
        reference_filename: Reference video filename (name only, not full path)
        source_dir: Directory to search in (defaults to RAW_VIDEO_DIR from config)
        
    Returns:
        List of full paths to matching video files, sorted by name
    """
    if source_dir is None:
        source_dir = RAW_VIDEO_DIR
    
    # Extract prefix from reference
    prefix = get_video_prefix(reference_filename)
    
    if not prefix:
        print(f"Warning: Could not extract prefix from '{reference_filename}'")
        return []
    
    print(f"Looking for videos with prefix: '{prefix}'")
    
    # Find all matching videos
    matching_videos = []
    
    try:
        for filename in os.listdir(source_dir):
            if not filename.lower().endswith('.mp4'):
                continue
            
            file_prefix = get_video_prefix(filename)
            if file_prefix == prefix:
                full_path = os.path.join(source_dir, filename)
                matching_videos.append(full_path)
    except FileNotFoundError:
        print(f"Error: Directory not found: {source_dir}")
        return []
    
    # Sort by filename (which includes timestamp, so chronological order)
    matching_videos.sort()
    
    return matching_videos


def get_video_info(video_paths: list[str]) -> dict:
    """
    Get summary information about a list of videos.
    
    Args:
        video_paths: List of video file paths
        
    Returns:
        Dictionary with summary info
    """
    total_size = 0
    for path in video_paths:
        try:
            total_size += os.path.getsize(path)
        except OSError:
            pass
    
    return {
        "count": len(video_paths),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "videos": [os.path.basename(p) for p in video_paths]
    }


def list_available_prefixes(source_dir: str = None) -> list[str]:
    """
    List all unique video prefixes in the source directory.
    
    Args:
        source_dir: Directory to scan (defaults to RAW_VIDEO_DIR)
        
    Returns:
        List of unique prefixes
    """
    if source_dir is None:
        source_dir = RAW_VIDEO_DIR
    
    prefixes = set()
    
    try:
        for filename in os.listdir(source_dir):
            if filename.lower().endswith('.mp4'):
                prefix = get_video_prefix(filename)
                if prefix:
                    prefixes.add(prefix)
    except FileNotFoundError:
        print(f"Error: Directory not found: {source_dir}")
        return []
    
    return sorted(list(prefixes))


if __name__ == "__main__":
    # Test the video collector
    print("=== Video Collector Test ===\n")
    
    # List available prefixes
    print("Available video prefixes:")
    prefixes = list_available_prefixes()
    for i, prefix in enumerate(prefixes, 1):
        print(f"  {i}. {prefix}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with a specific reference
    test_ref = "Winposh Regent_Camera 01_20251215110139_20251215110636.mp4"
    print(f"Testing with reference: {test_ref}\n")
    
    videos = collect_similar_videos(test_ref)
    info = get_video_info(videos)
    
    print(f"Found {info['count']} matching videos ({info['total_size_mb']} MB total):")
    for v in info['videos']:
        print(f"  - {v}")
