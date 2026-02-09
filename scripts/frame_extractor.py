import os
import sys
import subprocess

def extract_frames(video_path, output_dir, threshold=0.3):
    """
    Extracts frames from video based on scene changes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ffmpeg command to extract frames where scene change is detected
    # select='gt(scene,threshold)' picks frames with significant changes
    # scale=1280:-1 ensures decent resolution
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',scale=1280:-1",
        "-vsync", "vfr",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Get timestamps of extracted frames
        # This is a bit tricky with ffmpeg directly, so we'll use a second pass to get timestamps if needed
        # For now, we'll just list the files
        frames = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        return frames
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e.stderr.decode()}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 frame_extractor.py <video_path> <output_dir> [threshold]")
        sys.exit(1)
    
    v_path = sys.argv[1]
    o_dir = sys.argv[2]
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    
    print(f"Extracting frames from {v_path} to {o_dir} with threshold {thresh}...")
    extracted = extract_frames(v_path, o_dir, thresh)
    print(f"Extracted {len(extracted)} frames.")
