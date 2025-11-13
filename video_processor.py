"""
Video Processing Utility for Deepfake Detection
Handles video frame extraction for analysis
"""

import cv2
import tempfile
import os
from typing import List, Tuple
from PIL import Image
import numpy as np


class VideoProcessor:
    """Extracts frames from video files for deepfake analysis"""

    # Configuration constants
    MAX_DURATION_SECONDS = 60  # Maximum video duration: 1 minute
    MAX_WIDTH = 1920  # Maximum resolution width (1080p)
    MAX_HEIGHT = 1920  # Maximum resolution height
    MIN_WIDTH = 854  # Minimum resolution width (480p)
    MIN_HEIGHT = 480  # Minimum resolution height

    @staticmethod
    def validate_video(video_path: str) -> dict:
        """
        Validate video duration and properties

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'error': str (if invalid),
                'duration': float,
                'width': int,
                'height': int
            }
        """
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            return {'valid': False, 'error': 'Could not open video file'}

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()

        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0

        # Validate duration
        if duration > VideoProcessor.MAX_DURATION_SECONDS:
            return {
                'valid': False,
                'error': f'Video duration ({duration:.1f}s) exceeds maximum allowed duration ({VideoProcessor.MAX_DURATION_SECONDS}s)',
                'duration': duration,
                'width': width,
                'height': height
            }

        # Validate minimum resolution (works for both landscape and portrait)
        # Check if either orientation meets the minimum requirements
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        min_required = min(VideoProcessor.MIN_WIDTH, VideoProcessor.MIN_HEIGHT)
        max_required = max(VideoProcessor.MIN_WIDTH, VideoProcessor.MIN_HEIGHT)

        if min_dimension < min_required or max_dimension < max_required:
            return {
                'valid': False,
                'error': f'Video resolution ({width}x{height}) is too low for reliable analysis. Minimum resolution required: {VideoProcessor.MIN_WIDTH}x{VideoProcessor.MIN_HEIGHT} (480p) in any orientation',
                'duration': duration,
                'width': width,
                'height': height
            }

        return {
            'valid': True,
            'duration': duration,
            'width': width,
            'height': height
        }

    @staticmethod
    def downscale_video(input_path: str, output_path: str, max_width: int = None, max_height: int = None) -> dict:
        """
        Downscale video to reduce resolution and file size

        Args:
            input_path: Path to input video
            output_path: Path to save downscaled video
            max_width: Maximum width (default: MAX_WIDTH)
            max_height: Maximum height (default: MAX_HEIGHT)

        Returns:
            Dictionary with original and new dimensions
        """
        if max_width is None:
            max_width = VideoProcessor.MAX_WIDTH
        if max_height is None:
            max_height = VideoProcessor.MAX_HEIGHT

        # Open input video
        video = cv2.VideoCapture(input_path)

        if not video.isOpened():
            raise ValueError("Could not open input video")

        # Get original properties
        original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate new dimensions (maintain aspect ratio)
        scale_factor = min(max_width / original_width, max_height / original_height, 1.0)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Make dimensions even (required for some codecs)
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1

        # If no downscaling needed, just copy the file
        if scale_factor >= 1.0:
            video.release()
            import shutil
            shutil.copy(input_path, output_path)
            return {
                'original_width': original_width,
                'original_height': original_height,
                'new_width': original_width,
                'new_height': original_height,
                'downscaled': False
            }

        # Create video writer with downscaled dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        # Process each frame
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            writer.write(resized_frame)

        video.release()
        writer.release()

        return {
            'original_width': original_width,
            'original_height': original_height,
            'new_width': new_width,
            'new_height': new_height,
            'downscaled': True
        }

    @staticmethod
    def extract_frames(video_path: str, frame_interval_seconds: float = 0.5) -> List[Image.Image]:
        """
        Extract ALL frames from a video file at regular intervals

        Args:
            video_path: Path to the video file
            frame_interval_seconds: Interval between frames in seconds (default: 0.5)

        Returns:
            List of PIL Image objects (all frames at specified interval)

        Raises:
            ValueError: If video cannot be opened or has no frames
        """
        frames = []

        # Open video file
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if total_frames == 0:
            video.release()
            raise ValueError("Video file contains no frames")

        # Calculate frame interval based on FPS
        # Extract 1 frame every frame_interval_seconds seconds
        if fps > 0:
            frame_step = int(fps * frame_interval_seconds)
        else:
            frame_step = 15  # Default to 15 frames (assuming 30fps = 0.5s)

        # Ensure at least 1 frame step
        frame_step = max(1, frame_step)

        # Generate frame indices: 0, frame_step, 2*frame_step, ... for ENTIRE video
        frame_indices = list(range(0, total_frames, frame_step))

        # Extract ALL frames at the specified interval
        for target_idx in frame_indices:
            # Seek to the target frame
            video.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

            ret, frame = video.read()
            if not ret:
                continue

            # Convert BGR (OpenCV format) to RGB (PIL format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

        video.release()

        if len(frames) == 0:
            raise ValueError("Could not extract any frames from video")

        return frames

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get metadata information about a video file

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video metadata
        """
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            return {"error": "Could not open video"}

        info = {
            "fps": video.get(cv2.CAP_PROP_FPS),
            "total_frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": 0
        }

        # Calculate duration
        if info["fps"] > 0:
            info["duration_seconds"] = info["total_frames"] / info["fps"]

        video.release()
        return info

    @staticmethod
    def save_uploaded_video(uploaded_file) -> str:
        """
        Save an uploaded video file to a temporary location

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Path to the saved temporary file
        """
        # Create a temporary file with the same extension
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name


def analyze_extracted_frames(frames: List[Image.Image], detector) -> dict:
    """
    Analyze pre-extracted frames for deepfakes

    Args:
        frames: List of PIL Image objects (already extracted)
        detector: DeepfakeDetector or MultiStageDetector instance

    Returns:
        Dictionary containing analysis results for all frames
    """
    # Analyze each frame
    frame_results = []

    for idx, frame in enumerate(frames):
        # MultiStageDetector can handle PIL Images directly
        # For backward compatibility, also support file-based detectors
        try:
            # Try to pass PIL Image directly (MultiStageDetector supports this)
            result = detector.analyze(frame)
        except:
            # Fallback: Save to temp file (for old-style detectors)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                frame.save(tmp.name, 'JPEG')
                result = detector.analyze(tmp.name)
                os.unlink(tmp.name)

        # Add frame metadata
        result['frame_number'] = idx + 1
        result['frame_image'] = frame  # Store the PIL Image for display
        frame_results.append(result)

    # Calculate aggregate statistics
    deepfake_count = sum(1 for r in frame_results if r.get('is_deepfake', False))
    avg_confidence = sum(r.get('confidence', 0) for r in frame_results) / len(frame_results)

    # Calculate category statistics (for multi-stage detection)
    category_counts = {}
    detector_counts = {}

    for r in frame_results:
        # Count categories
        category = r.get('category', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1

        # Count detectors used
        detector_used = r.get('detector_used', 'unknown')
        detector_counts[detector_used] = detector_counts.get(detector_used, 0) + 1

    # Stricter overall verdict: require 60% of frames to be fake
    fake_percentage = deepfake_count / len(frame_results)
    overall_verdict = 'LIKELY DEEPFAKE' if fake_percentage > 0.6 else 'LIKELY AUTHENTIC'

    return {
        'frames_analyzed': len(frame_results),
        'frame_results': frame_results,
        'frames': frames,  # Include all extracted frames
        'aggregate': {
            'deepfake_frames': deepfake_count,
            'authentic_frames': len(frame_results) - deepfake_count,
            'average_confidence': avg_confidence,
            'overall_verdict': overall_verdict,
            'fake_percentage': round(fake_percentage * 100, 1),
            'category_breakdown': category_counts,
            'detector_breakdown': detector_counts
        }
    }


def analyze_video_frames(video_path: str, detector, num_frames: int = 5) -> dict:
    """
    Extract frames from video and analyze each for deepfakes

    Args:
        video_path: Path to video file
        detector: DeepfakeDetector instance
        num_frames: Number of frames to analyze

    Returns:
        Dictionary containing analysis results for all frames, including frame images
    """
    processor = VideoProcessor()

    # Get video info
    video_info = processor.get_video_info(video_path)

    # Extract frames
    frames = processor.extract_frames(video_path, num_frames)

    # Analyze each frame
    frame_results = []

    for idx, frame in enumerate(frames):
        # Save frame to temporary file for detector
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            frame.save(tmp.name, 'JPEG')

            # Analyze frame
            result = detector.analyze(tmp.name)
            result['frame_number'] = idx + 1
            result['frame_image'] = frame  # Store the PIL Image for display
            frame_results.append(result)

            # Clean up temp file
            os.unlink(tmp.name)

    # Calculate aggregate statistics
    deepfake_count = sum(1 for r in frame_results if r.get('is_deepfake', False))
    avg_confidence = sum(r.get('confidence', 0) for r in frame_results) / len(frame_results)

    return {
        'video_info': video_info,
        'frames_analyzed': len(frame_results),
        'frame_results': frame_results,
        'frames': frames,  # Include all extracted frames
        'aggregate': {
            'deepfake_frames': deepfake_count,
            'authentic_frames': len(frame_results) - deepfake_count,
            'average_confidence': avg_confidence,
            'overall_verdict': 'LIKELY DEEPFAKE' if deepfake_count > len(frame_results) / 2 else 'LIKELY AUTHENTIC'
        }
    }
