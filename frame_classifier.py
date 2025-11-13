"""
Frame Classification Module using CLIP
Categorizes video frames into content types for specialized deepfake detection
"""

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class FrameClassifier:
    """
    CLIP-based frame classifier for categorizing image content.

    Uses zero-shot classification to determine if a frame contains:
    - Human faces/people
    - Objects/other content

    This enables routing frames to specialized deepfake detectors.
    """

    # Category definitions for zero-shot classification
    CATEGORIES = {
        'human_face': "a photo of a person or human face",
        'other': "a photo of an object, landscape, or scene"
    }

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, verbose=False):
        """
        Initialize the frame classifier.

        Args:
            model_name: HuggingFace CLIP model name (default: openai/clip-vit-base-patch32)
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            verbose: Print debug output during inference (default: False)
        """
        self.model_name = model_name
        self.verbose = verbose

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """
        Load the CLIP model and processor.
        Model will be downloaded on first run.
        """
        try:
            if self.verbose:
                print(f"Loading CLIP model: {self.model_name}")
                print(f"Using device: {self.device}")
                print("Note: First run will download the model (~600MB), please wait...")

            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            if self.verbose:
                print("CLIP model loaded successfully!")

        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Frame classification will not be available.")
            self.model = None
            self.processor = None

    def classify(self, image):
        """
        Classify an image into one of the predefined categories.

        Args:
            image: PIL Image object or path to image file

        Returns:
            Dictionary with classification results:
            {
                'category': str,  # 'human_face' or 'other'
                'confidence': float,  # 0-1 confidence score
                'all_scores': dict,  # Scores for all categories
                'success': bool
            }
        """
        try:
            # Handle image path
            if isinstance(image, str):
                image = Image.open(image)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Check if model is loaded
            if self.model is None or self.processor is None:
                return {
                    'success': False,
                    'error': 'CLIP model not loaded',
                    'category': 'other',  # Fallback to general detector
                    'confidence': 0.0
                }

            # Prepare text descriptions for categories
            text_inputs = list(self.CATEGORIES.values())

            # Process inputs
            inputs = self.processor(
                text=text_inputs,
                images=image,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Calculate similarity scores
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Convert to numpy
            probs_np = probs.cpu().numpy()[0]

            # Map scores to category names
            category_keys = list(self.CATEGORIES.keys())
            all_scores = {
                category_keys[i]: float(probs_np[i])
                for i in range(len(category_keys))
            }

            # Get top category
            top_idx = np.argmax(probs_np)
            top_category = category_keys[top_idx]
            top_confidence = float(probs_np[top_idx])

            if self.verbose:
                print(f"Classification scores: {all_scores}")
                print(f"Top category: {top_category} (confidence: {top_confidence:.3f})")

            return {
                'success': True,
                'category': top_category,
                'confidence': top_confidence,
                'all_scores': all_scores
            }

        except Exception as e:
            if self.verbose:
                print(f"Error during classification: {e}")

            return {
                'success': False,
                'error': str(e),
                'category': 'other',  # Fallback to general detector
                'confidence': 0.0
            }

    def classify_batch(self, images):
        """
        Classify multiple images at once (more efficient than one-by-one).

        Args:
            images: List of PIL Image objects or image paths

        Returns:
            List of classification result dictionaries
        """
        # For simplicity, process one by one
        # Could be optimized to use true batching later
        return [self.classify(img) for img in images]


# Example usage and testing
if __name__ == '__main__':
    import os

    classifier = FrameClassifier(verbose=True)

    # Test with a sample image (if available)
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        result = classifier.classify(test_image)
        print("\nClassification Result:")
        print(result)
    else:
        print("No test image found. Create a test_image.jpg to test the classifier.")
