import cv2
import os
import math

def extract_frames(video_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration} seconds")

    frame_interval = math.ceil(fps) # Extract 1 frame per second

    count = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if it's at the desired interval (approximately 1 frame per second)
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        count += 1

    cap.release()
    print(f"Finished extracting {extracted_count} frames to {output_dir}")

if __name__ == "__main__":
    video_input_path = "Edited/Kapal 1 (Mas Kael)/Winposh-CAM-01.mp4"
    
    # Derive output directory from video filename
    video_filename = os.path.basename(video_input_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    output_base_dir = "Extracted/Kapal 1 (Mas Kael)"
    output_destination_dir = os.path.join(output_base_dir, video_name_without_ext)

    extract_frames(video_input_path, output_destination_dir)
