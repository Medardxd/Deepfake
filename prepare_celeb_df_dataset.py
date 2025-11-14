"""
Celeb-DF-v2 Dataset Preparation for Fine-Tuning
Extracts frames from videos and creates balanced train/val splits
"""

import os
import cv2
from pathlib import Path
import random
from tqdm import tqdm
import shutil

# Configuration
DATASET_ROOT = "/mnt/d/Celeb-DF-v2"
OUTPUT_ROOT = "/mnt/e/szakdoga/Deepfake/celeb_df_processed"
TEST_LIST_FILE = f"{DATASET_ROOT}/List_of_testing_videos.txt"

# Sampling parameters
FRAMES_PER_VIDEO = 8  # Extract 8 frames per video
BALANCE_CLASSES = True  # Use equal number of real/fake videos
MAX_FAKE_VIDEOS = 890  # Match number of real videos

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def read_test_list(test_list_file):
    """Read the official test list to exclude from training"""
    test_videos = set()
    with open(test_list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_path = parts[1]  # Format: "0 Celeb-synthesis/id1_id0_0007.mp4"
                test_videos.add(video_path)
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

def collect_training_videos(dataset_root, test_videos):
    """
    Collect all videos not in the test set

    Returns:
        dict: {'real': [paths], 'fake': [paths]}
    """
    training_videos = {'real': [], 'fake': []}

    # Collect real videos from Celeb-real
    celeb_real_dir = Path(dataset_root) / "Celeb-real"
    for video_file in celeb_real_dir.glob("*.mp4"):
        rel_path = f"Celeb-real/{video_file.name}"
        if rel_path not in test_videos:
            training_videos['real'].append(video_file)

    # Collect real videos from YouTube-real
    youtube_real_dir = Path(dataset_root) / "YouTube-real"
    for video_file in youtube_real_dir.glob("*.mp4"):
        rel_path = f"YouTube-real/{video_file.name}"
        if rel_path not in test_videos:
            training_videos['real'].append(video_file)

    # Collect fake videos from Celeb-synthesis
    synthesis_dir = Path(dataset_root) / "Celeb-synthesis"
    for video_file in synthesis_dir.glob("*.mp4"):
        rel_path = f"Celeb-synthesis/{video_file.name}"
        if rel_path not in test_videos:
            training_videos['fake'].append(video_file)

    return training_videos

def create_dataset(dataset_root, output_root, test_list_file,
                   frames_per_video=8, balance=True, max_fake=890):
    """
    Main function to create the training dataset
    """
    print("="*60)
    print("Celeb-DF-v2 Dataset Preparation")
    print("="*60)

    # Read test list
    print(f"\n1. Reading test list from {test_list_file}")
    test_videos = read_test_list(test_list_file)
    print(f"   Found {len(test_videos)} videos in test set (will exclude from training)")

    # Collect training videos
    print(f"\n2. Collecting training videos...")
    training_videos = collect_training_videos(dataset_root, test_videos)
    print(f"   Real videos available: {len(training_videos['real'])}")
    print(f"   Fake videos available: {len(training_videos['fake'])}")

    # Balance classes if requested
    if balance and max_fake:
        print(f"\n3. Balancing dataset...")
        print(f"   Using {len(training_videos['real'])} real videos")
        print(f"   Sampling {max_fake} fake videos from {len(training_videos['fake'])} available")

        random.shuffle(training_videos['fake'])
        training_videos['fake'] = training_videos['fake'][:max_fake]

    # Create output directories
    output_path = Path(output_root)
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)

    # Process each class
    for label in ['real', 'fake']:
        videos = training_videos[label]
        random.shuffle(videos)

        # Split into train/val
        split_idx = int(len(videos) * TRAIN_RATIO)
        train_videos = videos[:split_idx]
        val_videos = videos[split_idx:]

        print(f"\n4. Processing {label} videos...")
        print(f"   Train: {len(train_videos)} videos")
        print(f"   Val: {len(val_videos)} videos")

        # Process training videos
        print(f"   Extracting frames from training videos...")
        frame_count = 0
        for video_path in tqdm(train_videos, desc=f"   {label}/train"):
            frames = extract_frames_from_video(video_path, frames_per_video)
            video_name = video_path.stem

            for frame_idx, frame in enumerate(frames):
                output_file = output_path / 'train' / label / f"{video_name}_frame{frame_idx:02d}.jpg"
                save_frame(frame, output_file)
                frame_count += 1

        print(f"   Extracted {frame_count} training frames")

        # Process validation videos
        print(f"   Extracting frames from validation videos...")
        frame_count = 0
        for video_path in tqdm(val_videos, desc=f"   {label}/val"):
            frames = extract_frames_from_video(video_path, frames_per_video)
            video_name = video_path.stem

            for frame_idx, frame in enumerate(frames):
                output_file = output_path / 'val' / label / f"{video_name}_frame{frame_idx:02d}.jpg"
                save_frame(frame, output_file)
                frame_count += 1

        print(f"   Extracted {frame_count} validation frames")

    # Print summary
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)

    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for label in ['real', 'fake']:
            count = len(list((output_path / split / label).glob("*.jpg")))
            print(f"  {label}: {count} frames")

    print(f"\nDataset saved to: {output_root}")
    print("="*60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Run dataset preparation
    create_dataset(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        test_list_file=TEST_LIST_FILE,
        frames_per_video=FRAMES_PER_VIDEO,
        balance=BALANCE_CLASSES,
        max_fake=MAX_FAKE_VIDEOS
    )
