"""
Prepare Official Test Set for Model Evaluation
Extracts frames from videos in List_of_testing_videos.txt
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
DATASET_ROOT = "/mnt/d/Celeb-DF-v2"
OUTPUT_ROOT = "/mnt/e/szakdoga/Deepfake/celeb_df_processed/test"
TEST_LIST_FILE = f"{DATASET_ROOT}/List_of_testing_videos.txt"

# Sampling parameters
FRAMES_PER_VIDEO = 8  # Extract 8 frames per video (same as training)


def read_test_list(test_list_file):
    """
    Read the official test list and parse video paths with labels

    Returns:
        dict: {'real': [paths], 'fake': [paths]}
    """
    test_videos = {'real': [], 'fake': []}

    print(f"Reading test list from {test_list_file}...")

    with open(test_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                label = int(parts[0])  # 0 = fake, 1 = real
                video_path = parts[1]  # e.g., "Celeb-synthesis/id1_id0_0007.mp4"

                # Construct full path
                full_path = Path(DATASET_ROOT) / video_path

                if full_path.exists():
                    if label == 1:
                        test_videos['real'].append(full_path)
                    else:
                        test_videos['fake'].append(full_path)
                else:
                    print(f"Warning: Video not found: {full_path}")

    return test_videos


def extract_frames_from_video(video_path, num_frames=8):
    """
    Extract evenly spaced frames from a video

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of numpy arrays (frames)
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: Video has 0 frames: {video_path}")
        cap.release()
        return frames

    # Extract evenly spaced frames
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def save_frame(frame, output_path):
    """Save a frame as JPEG"""
    # Convert RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def prepare_test_set():
    """
    Main function to prepare the test set
    """
    print("="*60)
    print("Official Test Set Preparation")
    print("="*60)

    # Read test list
    print(f"\n1. Reading test list...")
    test_videos = read_test_list(TEST_LIST_FILE)
    print(f"   Real videos: {len(test_videos['real'])}")
    print(f"   Fake videos: {len(test_videos['fake'])}")
    print(f"   Total: {len(test_videos['real']) + len(test_videos['fake'])}")

    if len(test_videos['real']) == 0 and len(test_videos['fake']) == 0:
        print("\nError: No test videos found!")
        return

    # Create output directories
    output_path = Path(OUTPUT_ROOT)
    for label in ['real', 'fake']:
        (output_path / label).mkdir(parents=True, exist_ok=True)

    # Process each class
    for label in ['real', 'fake']:
        videos = test_videos[label]

        if len(videos) == 0:
            print(f"\n2. No {label} videos to process, skipping...")
            continue

        print(f"\n2. Processing {label} videos...")
        print(f"   Total videos: {len(videos)}")

        # Extract frames
        print(f"   Extracting frames...")
        frame_count = 0

        for video_path in tqdm(videos, desc=f"   {label}"):
            frames = extract_frames_from_video(video_path, FRAMES_PER_VIDEO)
            video_name = video_path.stem

            for frame_idx, frame in enumerate(frames):
                output_file = output_path / label / f"{video_name}_frame{frame_idx:02d}.jpg"
                save_frame(frame, output_file)
                frame_count += 1

        print(f"   Extracted {frame_count} frames from {len(videos)} videos")

    # Print summary
    print("\n" + "="*60)
    print("Test Set Preparation Complete!")
    print("="*60)

    print(f"\nTEST SET:")
    for label in ['real', 'fake']:
        count = len(list((output_path / label).glob("*.jpg")))
        videos = len(test_videos[label])
        print(f"  {label}: {count} frames from {videos} videos")

    print(f"\nTest set saved to: {OUTPUT_ROOT}")
    print("="*60)


if __name__ == "__main__":
    prepare_test_set()
