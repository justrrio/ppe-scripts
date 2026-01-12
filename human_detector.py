"""
Human Detection Module
Orchestrates human detection using Groq AI and organizes frames.
"""
import os
import shutil

try:
    from .config import BATCH_SIZE
    from .groq_client import GroqVisionClient
    from .utils import ensure_dir
except ImportError:
    from config import BATCH_SIZE
    from groq_client import GroqVisionClient
    from utils import ensure_dir


def detect_and_organize_frames(
    frames_dir: str,
    groq_client: GroqVisionClient = None,
    dry_run: bool = False
) -> dict:
    """
    Process all frames in a directory to detect humans and organize files.
    
    Frames without humans are moved to a 'no-human' subfolder.
    
    Args:
        frames_dir: Directory containing extracted frames
        groq_client: Groq client instance (creates one if not provided)
        dry_run: If True, only print what would be done without moving files
        
    Returns:
        Statistics dictionary with processing results
    """
    if groq_client is None:
        groq_client = GroqVisionClient()
    
    # Get all frame images
    all_frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and os.path.isfile(os.path.join(frames_dir, f))
    ])
    
    if not all_frames:
        print("No frames found to process")
        return {"total": 0, "with_human": 0, "no_human": 0, "errors": 0}
    
    # Create no-human subfolder
    no_human_dir = os.path.join(frames_dir, "no-human")
    if not dry_run:
        ensure_dir(no_human_dir)
    
    print(f"\n{'='*60}")
    print("HUMAN DETECTION")
    print(f"{'='*60}")
    print(f"Frames directory: {frames_dir}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated API calls: {(len(all_frames) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")
    
    stats = {
        "total": len(all_frames),
        "with_human": 0,
        "no_human": 0,
        "errors": 0,
        "moved_files": []
    }
    
    # Process in batches
    for batch_start in range(0, len(all_frames), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_frames))
        batch = all_frames[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_frames) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} frames...")
        
        # Analyze batch
        results = groq_client.analyze_images_batch(batch)
        
        # Process results
        for result in results:
            image_path = result["image_path"]
            has_human = result.get("has_human", True)
            confidence = result.get("confidence", 0.0)
            error = result.get("error")
            
            filename = os.path.basename(image_path)
            
            if error:
                stats["errors"] += 1
                print(f"  ⚠ {filename}: Error - {error}")
                continue
            
            if has_human:
                stats["with_human"] += 1
                print(f"  ✓ {filename}: Human detected (conf: {confidence:.2f})")
            else:
                stats["no_human"] += 1
                print(f"  ✗ {filename}: No human (conf: {confidence:.2f})")
                
                # Move to no-human folder
                if not dry_run:
                    dest_path = os.path.join(no_human_dir, filename)
                    try:
                        shutil.move(image_path, dest_path)
                        stats["moved_files"].append(filename)
                    except Exception as e:
                        print(f"    Error moving file: {e}")
                        stats["errors"] += 1
        
        print()
    
    print(f"{'='*60}")
    print("DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {stats['total']}")
    print(f"  With humans: {stats['with_human']}")
    print(f"  Without humans: {stats['no_human']} (moved to no-human/)")
    print(f"  Errors: {stats['errors']}")
    print(f"API calls made: {groq_client.get_stats()['total_requests']}")
    print(f"{'='*60}\n")
    
    return stats


def restore_no_human_frames(frames_dir: str) -> int:
    """
    Restore frames from no-human folder back to main folder.
    Useful for re-processing or undoing detection.
    
    Args:
        frames_dir: Main frames directory
        
    Returns:
        Number of files restored
    """
    no_human_dir = os.path.join(frames_dir, "no-human")
    
    if not os.path.exists(no_human_dir):
        print("No 'no-human' folder found")
        return 0
    
    files = [f for f in os.listdir(no_human_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    restored = 0
    for filename in files:
        src = os.path.join(no_human_dir, filename)
        dest = os.path.join(frames_dir, filename)
        try:
            shutil.move(src, dest)
            restored += 1
        except Exception as e:
            print(f"Error restoring {filename}: {e}")
    
    print(f"Restored {restored} files from no-human folder")
    return restored


if __name__ == "__main__":
    # Test the human detector
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from Scripts.config import EXTRACTED_DIR
    
    print("=== Human Detector Test ===\n")
    
    # Find any existing extracted folder with images
    test_dir = None
    if os.path.exists(EXTRACTED_DIR):
        for folder in os.listdir(EXTRACTED_DIR):
            folder_path = os.path.join(EXTRACTED_DIR, folder)
            if os.path.isdir(folder_path):
                # Check if folder has jpg files
                jpgs = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                if jpgs:
                    test_dir = folder_path
                    break
    
    if test_dir:
        # Do a dry run first
        print(f"Testing with folder: {os.path.basename(test_dir)}")
        print("Running in DRY RUN mode (no files will be moved)\n")
        stats = detect_and_organize_frames(test_dir, dry_run=True)
    else:
        print(f"No extracted folders with images found in: {EXTRACTED_DIR}")
        print("Run frame extraction first")
