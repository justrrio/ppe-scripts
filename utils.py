"""
Utility functions for PPE Frame Extraction Pipeline
"""
import os
import re


def get_video_prefix(filename: str) -> str:
    """
    Extract the base prefix from a video filename including the camera number.
    
    Example:
        "Winposh Regent_Camera 07_20251215110142_20251215110647.mp4"
        -> "Winposh Regent_Camera 07"
        
        "SMS Value_Camera 01_20251215134237_20251215153515.mp4"
        -> "SMS Value_Camera 01"
    
    The pattern extracts everything up to and including the camera number,
    which appears before the timestamp portions.
    
    Args:
        filename: Video filename (with or without path)
        
    Returns:
        Base prefix including camera identifier
    """
    # Get just the filename without path
    basename = os.path.basename(filename)
    
    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]
    
    # Pattern: {Source}_{Camera XX}_timestamp_timestamp
    # We want to extract {Source}_{Camera XX}
    # Camera number is typically 2 digits after "Camera " or "IPCamera "
    # Timestamps are 14 digits (YYYYMMDDHHMMSS format)
    
    # Match pattern: anything + Camera/IPCamera + space + digits + underscore + timestamp
    pattern = r'^(.+?(?:Camera|IPCamera)\s+\d+)_\d{14}_\d{14}$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        return match.group(1)
    
    # Fallback: try to find Camera XX pattern anywhere
    pattern2 = r'^(.+?(?:Camera|IPCamera)\s+\d+)'
    match2 = re.match(pattern2, name_without_ext)
    if match2:
        return match2.group(1)
    
    # Last fallback: return everything before the first timestamp-like pattern
    parts = name_without_ext.split('_')
    prefix_parts = []
    for part in parts:
        # Stop when we hit a 14-digit timestamp
        if re.match(r'^\d{14}$', part):
            break
        prefix_parts.append(part)
    
    return '_'.join(prefix_parts) if prefix_parts else name_without_ext


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a string to be used as a folder name.
    
    Args:
        name: Original string
        
    Returns:
        Safe folder name
    """
    # Replace spaces and special characters
    safe_name = re.sub(r'[^\w\-]', '_', name)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name


def format_duration(seconds: float) -> str:
    """
    Format seconds into human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    # Test the utility functions
    test_filenames = [
        "Winposh Regent_Camera 01_20251215110139_20251215110636.mp4",
        "SMS Value_Camera 04_20251216115805_20251216115808.mp4",
        "Steadfast_Camera 01_20251212161319_20251212170744.mp4",
    ]
    
    print("Testing get_video_prefix:")
    for fn in test_filenames:
        prefix = get_video_prefix(fn)
        print(f"  {fn}")
        print(f"  -> '{prefix}'")
        print()
