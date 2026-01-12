"""
PPE Dataset Analyzer Module
Analyzes images to determine suitability for PPE object detection training.
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


def analyze_and_organize_frames(
    frames_dir: str,
    groq_client: GroqVisionClient = None,
    dry_run: bool = False
) -> dict:
    """
    Analyze all frames in a directory for PPE dataset suitability.
    
    Images not suitable for PPE detection training are moved to 'not-suitable/' subfolder.
    
    Args:
        frames_dir: Directory containing extracted frames
        groq_client: Groq client instance (creates one if not provided)
        dry_run: If True, only print what would be done without moving files
        
    Returns:
        Statistics dictionary with processing results
    """
    if groq_client is None:
        groq_client = GroqVisionClient()
    
    # Get all frame images (exclude not-suitable subfolder)
    all_frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and os.path.isfile(os.path.join(frames_dir, f))
    ])
    
    if not all_frames:
        print("No frames found to process")
        return {"total": 0, "suitable": 0, "not_suitable": 0, "errors": 0}
    
    # Create not-suitable subfolder
    not_suitable_dir = os.path.join(frames_dir, "not-suitable")
    if not dry_run:
        ensure_dir(not_suitable_dir)
    
    print(f"\n{'='*60}")
    print("PPE DATASET SUITABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Frames directory: {frames_dir}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated API calls: {(len(all_frames) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")
    
    stats = {
        "total": len(all_frames),
        "suitable": 0,
        "not_suitable": 0,
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
        
        # Analyze batch for dataset suitability
        results = groq_client.analyze_dataset_suitability_batch(batch)
        
        # Process results
        for result in results:
            image_path = result["image_path"]
            is_suitable = result.get("is_suitable", True)
            error = result.get("error")
            
            filename = os.path.basename(image_path)
            
            if error:
                stats["errors"] += 1
                print(f"  ⚠ {filename}: Error - {error}")
                continue
            
            if is_suitable:
                stats["suitable"] += 1
                print(f"  ✓ {filename}: Suitable for dataset")
            else:
                stats["not_suitable"] += 1
                print(f"  ✗ {filename}: Not suitable")
                
                # Move to not-suitable folder
                if not dry_run:
                    dest_path = os.path.join(not_suitable_dir, filename)
                    try:
                        shutil.move(image_path, dest_path)
                        stats["moved_files"].append(filename)
                    except Exception as e:
                        print(f"    Error moving file: {e}")
                        stats["errors"] += 1
        
        print()
    
    print(f"{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames analyzed: {stats['total']}")
    print(f"  Suitable for dataset: {stats['suitable']}")
    print(f"  Not suitable: {stats['not_suitable']} (moved to not-suitable/)")
    print(f"  Errors: {stats['errors']}")
    print(f"API calls made: {groq_client.get_stats()['total_requests']}")
    print(f"{'='*60}\n")
    
    return stats
    

def restore_not_suitable_frames(frames_dir: str) -> int:
    """
    Restore frames from not-suitable folder back to main folder.
    Useful for re-processing or undoing analysis.
    
    Args:
        frames_dir: Main frames directory
        
    Returns:
        Number of files restored
    """
    not_suitable_dir = os.path.join(frames_dir, "not-suitable")
    
    if not os.path.exists(not_suitable_dir):
        print("No 'not-suitable' folder found")
        return 0
    
    files = [f for f in os.listdir(not_suitable_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    restored = 0
    for filename in files:
        src = os.path.join(not_suitable_dir, filename)
        dest = os.path.join(frames_dir, filename)
        try:
            shutil.move(src, dest)
            restored += 1
        except Exception as e:
            print(f"Error restoring {filename}: {e}")
    
    print(f"Restored {restored} files from not-suitable folder")
    return restored


if __name__ == "__main__":
    # Test the dataset analyzer
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from Scripts.config import EXTRACTED_DIR
    
    print("=== PPE Dataset Analyzer Test ===\n")
    
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
        stats = analyze_and_organize_frames(test_dir, dry_run=True)
    else:
        print(f"No extracted folders with images found in: {EXTRACTED_DIR}")
        print("Run frame extraction first")
