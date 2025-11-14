"""
Face Detection Module using MTCNN
Detects and crops faces from images for specialized deepfake analysis
"""

from PIL import Image, ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    """
    MTCNN-based face detector for extracting face regions from images.

    Detects faces in frames and returns cropped face regions suitable
    for specialized face deepfake detection models.
    """

    def __init__(self, min_face_size=40, thresholds=None, device=None, verbose=False):
        """
        Initialize the face detector.

        Args:
            min_face_size: Minimum face size to detect in pixels (default: 40)
            thresholds: MTCNN detection thresholds [P, R, O] (default: [0.6, 0.7, 0.7])
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            verbose: Print debug output during detection (default: False)
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds or [0.6, 0.7, 0.7]
        self.verbose = verbose

        # Force CPU for MTCNN due to RTX 5070 incompatibility
        # TODO: Remove this when PyTorch supports sm_120
        if device is None:
            self.device = "cpu"  # Force CPU to avoid CUDA sm_120 issues
        else:
            self.device = device

        self.mtcnn = None
        self._load_detector()

    def _load_detector(self):
        """
        Load the MTCNN face detector.
        Model will be downloaded on first run.
        """
        try:
            if self.verbose:
                print(f"Loading MTCNN face detector")
                print(f"Using device: {self.device}")
                print("Note: First run will download the model, please wait...")

            # Initialize MTCNN
            self.mtcnn = MTCNN(
                min_face_size=self.min_face_size,
                thresholds=self.thresholds,
                device=self.device,
                keep_all=True,  # Detect all faces
                post_process=False  # We'll handle post-processing ourselves
            )

            if self.verbose:
                print("MTCNN detector loaded successfully!")

        except Exception as e:
            print(f"Error loading MTCNN detector: {e}")
            print("Face detection will not be available.")
            self.mtcnn = None

    def detect_faces(self, image, return_all=False):
        """
        Detect faces in an image.

        Args:
            image: PIL Image object or path to image file
            return_all: If True, return all detected faces; if False, return only the largest (default: False)

        Returns:
            Dictionary with detection results:
            {
                'success': bool,
                'faces_found': int,
                'boxes': list of [x1, y1, x2, y2],  # Face bounding boxes
                'confidences': list of float,  # Detection confidence scores
                'landmarks': list of facial landmarks (optional)
            }
        """
        try:
            # Handle image path
            if isinstance(image, str):
                image = Image.open(image)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Check if detector is loaded
            if self.mtcnn is None:
                return {
                    'success': False,
                    'error': 'MTCNN detector not loaded',
                    'faces_found': 0,
                    'boxes': [],
                    'confidences': []
                }

            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

            # Handle no faces detected
            if boxes is None or len(boxes) == 0:
                if self.verbose:
                    print("No faces detected in image")

                return {
                    'success': True,
                    'faces_found': 0,
                    'boxes': [],
                    'confidences': [],
                    'landmarks': []
                }

            # Sort by confidence (highest first)
            sorted_indices = np.argsort(probs)[::-1]
            boxes = boxes[sorted_indices]
            probs = probs[sorted_indices]
            landmarks = landmarks[sorted_indices] if landmarks is not None else None

            # If return_all is False, only keep the largest/most confident face
            if not return_all:
                boxes = boxes[:1]
                probs = probs[:1]
                landmarks = landmarks[:1] if landmarks is not None else None

            if self.verbose:
                print(f"Detected {len(boxes)} face(s)")
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    print(f"  Face {i+1}: confidence={prob:.3f}, box={box}")

            return {
                'success': True,
                'faces_found': len(boxes),
                'boxes': boxes.tolist(),
                'confidences': probs.tolist(),
                'landmarks': landmarks.tolist() if landmarks is not None else []
            }

        except Exception as e:
            if self.verbose:
                print(f"Error during face detection: {e}")

            return {
                'success': False,
                'error': str(e),
                'faces_found': 0,
                'boxes': [],
                'confidences': []
            }

    def crop_face(self, image, box, margin=0.2):
        """
        Crop a face region from an image with optional margin.

        Args:
            image: PIL Image object
            box: Face bounding box [x1, y1, x2, y2]
            margin: Margin to add around face as fraction of face size (default: 0.2 = 20%)

        Returns:
            Cropped PIL Image of the face
        """
        # Handle image path
        if isinstance(image, str):
            image = Image.open(image)

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)

        # Calculate crop coordinates with margin
        crop_x1 = max(0, int(x1 - margin_x))
        crop_y1 = max(0, int(y1 - margin_y))
        crop_x2 = min(image.width, int(x2 + margin_x))
        crop_y2 = min(image.height, int(y2 + margin_y))

        # Crop image
        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        return cropped

    def detect_and_crop_faces(self, image, return_all=False, margin=0.2):
        """
        Detect faces and return cropped face images.

        Args:
            image: PIL Image object or path to image file
            return_all: If True, return all detected faces; if False, return only the largest (default: False)
            margin: Margin to add around face as fraction of face size (default: 0.2)

        Returns:
            Dictionary with cropped faces:
            {
                'success': bool,
                'faces_found': int,
                'cropped_faces': list of PIL Images,  # Cropped face images
                'boxes': list of [x1, y1, x2, y2],  # Original bounding boxes
                'confidences': list of float  # Detection confidence scores
            }
        """
        # Handle image path
        if isinstance(image, str):
            image = Image.open(image)

        # Detect faces
        detection_result = self.detect_faces(image, return_all=return_all)

        if not detection_result['success'] or detection_result['faces_found'] == 0:
            return {
                'success': detection_result['success'],
                'faces_found': 0,
                'cropped_faces': [],
                'boxes': [],
                'confidences': [],
                'error': detection_result.get('error', 'No faces detected')
            }

        # Crop each detected face
        cropped_faces = []
        for box in detection_result['boxes']:
            cropped = self.crop_face(image, box, margin=margin)
            cropped_faces.append(cropped)

        return {
            'success': True,
            'faces_found': len(cropped_faces),
            'cropped_faces': cropped_faces,
            'boxes': detection_result['boxes'],
            'confidences': detection_result['confidences']
        }

    def visualize_detections(self, image, boxes, confidences=None, save_path=None):
        """
        Draw bounding boxes on image to visualize face detections.

        Args:
            image: PIL Image object
            boxes: List of face bounding boxes [x1, y1, x2, y2]
            confidences: Optional list of confidence scores
            save_path: Optional path to save visualization (if None, returns image)

        Returns:
            PIL Image with drawn bounding boxes (if save_path is None)
        """
        # Handle image path
        if isinstance(image, str):
            image = Image.open(image)

        # Create a copy to draw on
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Draw each box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Add confidence label if available
            if confidences is not None and i < len(confidences):
                label = f"{confidences[i]:.2f}"
                draw.text((x1, y1 - 10), label, fill="green")

        # Save or return
        if save_path:
            img_draw.save(save_path)
            if self.verbose:
                print(f"Visualization saved to {save_path}")
        else:
            return img_draw


# Example usage and testing
if __name__ == '__main__':
    import os

    detector = FaceDetector(verbose=True)

    # Test with a sample image (if available)
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        result = detector.detect_and_crop_faces(test_image)
        print("\nFace Detection Result:")
        print(f"Faces found: {result['faces_found']}")

        if result['faces_found'] > 0:
            # Save cropped faces
            for i, face in enumerate(result['cropped_faces']):
                face.save(f'cropped_face_{i}.jpg')
                print(f"Saved cropped_face_{i}.jpg")
    else:
        print("No test image found. Create a test_image.jpg to test the detector.")
