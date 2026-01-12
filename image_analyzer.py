"""
Image Analyzer Module
Analyzes images for PPE dataset suitability and organizes them.
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


def analyze_and_filter_frames(
    frames_dir: str,
    groq_client: GroqVisionClient = None,
    dry_run: bool = False
) -> dict:
    """
    Analyze all frames in a directory for PPE dataset suitability.
    
    Images not suitable for PPE detection training are moved to 'not-suitable/' subfolder.
    
    Checks:
    - Human presence
    - PPE visibility
    - Image quality (blur, exposure)
    - Object size (too small/far)
    
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
        
        # Analyze batch
        results = groq_client.analyze_images_batch(batch)
        
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
                print(f"  ✓ {filename}: Suitable")
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
