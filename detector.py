import os
from PIL import Image
import numpy as np
from transformers import pipeline


class DeepfakeDetector:
    """
    AI-Generated Image Detection module using HuggingFace's pre-trained model.

    Uses: Ateeqq/ai-vs-human-image-detector
    - SiglipForImageClassification fine-tuned for AI vs Human detection
    - 99.23% accuracy on test data
    - Works on ALL image types (faces, landscapes, nature, art, etc.)
    - Trained on latest AI generators (Midjourney v6.1, FLUX, SD 3.5, GPT-4o)
    """

    def __init__(self, model_name=None):
        """
        Initialize the detector.

        Args:
            model_name: HuggingFace model name (default: Ateeqq/ai-vs-human-image-detector)
        """
        if model_name is None:
            model_name = "Ateeqq/ai-vs-human-image-detector"

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load the HuggingFace AI detection model.
        Model will be downloaded on first run (~370MB).
        """
        try:
            print(f"Loading AI image detection model: {self.model_name}")
            print("Note: First run will download the model (~370MB), please wait...")

            self.model = pipeline(
                'image-classification',
                model=self.model_name,
                device=-1  # Use CPU (-1), change to 0 for GPU if available
            )

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to placeholder mode...")
            self.model = None

    def _preprocess_image(self, image_path):
        """
        Preprocess image for model input.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image data
        """
        # Open and convert image
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get basic image info
        width, height = img.size

        return {
            'image': img,
            'width': width,
            'height': height,
            'format': img.format,
            'mode': img.mode
        }

    def _run_inference(self, preprocessed_data):
        """
        Run the ML model inference.

        Args:
            preprocessed_data: Preprocessed image data

        Returns:
            Model predictions with confidence scores
        """
        img = preprocessed_data['image']

        # Use the HuggingFace model if loaded
        if self.model is not None:
            try:
                # Run model prediction
                predictions = self.model(img)

                # Debug: Print raw predictions to understand format
                print("=" * 50)
                print("RAW MODEL OUTPUT:")
                print(predictions)
                print("=" * 50)

                # Extract results
                # Expected format: [{'label': 'ai', 'score': 0.95}, {'label': 'hum', 'score': 0.05}]
                ai_score = 0.0
                human_score = 0.0

                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']

                    print(f"Label: {label}, Score: {score}")

                    if 'ai' in label or 'fake' in label or 'generated' in label or 'artificial' in label:
                        ai_score = score
                    elif 'hum' in label or 'real' in label or 'authentic' in label or 'natural' in label:
                        human_score = score

                print(f"Final scores - AI-Generated: {ai_score}, Human: {human_score}")

                return {
                    'ai_confidence': float(ai_score),
                    'human_confidence': float(human_score),
                    'predictions': predictions,
                    'using_real_model': True
                }

            except Exception as e:
                print(f"Error during inference: {e}")
                print("Falling back to placeholder...")

        # Fallback: placeholder logic if model fails or not loaded
        img_array = np.array(img)
        avg_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        dummy_confidence = min(0.5 + (std_color[0] / 255.0) * 0.5, 0.99)

        return {
            'ai_confidence': float(dummy_confidence),
            'human_confidence': float(1.0 - dummy_confidence),
            'predictions': [],
            'using_real_model': False
        }

    def analyze(self, image_path):
        """
        Analyze an image for AI-generation detection.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate image exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Image file not found'
                }

            # Preprocess image
            preprocessed = self._preprocess_image(image_path)

            # Run inference
            predictions = self._run_inference(preprocessed)

            # Extract AI confidence
            ai_confidence = predictions['ai_confidence']
            is_deepfake = ai_confidence > 0.5

            # Determine confidence level label
            if ai_confidence > 0.85 or ai_confidence < 0.15:
                confidence_label = 'high'
            elif ai_confidence > 0.65 or ai_confidence < 0.35:
                confidence_label = 'medium'
            else:
                confidence_label = 'low'

            # Prepare response
            result = {
                'success': True,
                'is_deepfake': is_deepfake,
                'confidence': round(ai_confidence * 100, 2),
                'confidence_label': confidence_label,
                'verdict': 'AI-GENERATED' if is_deepfake else 'APPEARS AUTHENTIC',
                'image_info': {
                    'width': preprocessed['width'],
                    'height': preprocessed['height'],
                    'format': preprocessed['format']
                },
                'model_info': {
                    'using_real_model': predictions.get('using_real_model', False),
                    'model_name': self.model_name if predictions.get('using_real_model', False) else 'Placeholder'
                }
            }

            # Add note if using placeholder
            if not predictions.get('using_real_model', False):
                result['note'] = 'Warning: Using placeholder detection. Install dependencies and restart to use the real AI model.'
            else:
                result['note'] = f'Analysis performed using {self.model_name} (99.23% accuracy). Works on all image types.'

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Example usage and testing
if __name__ == '__main__':
    detector = DeepfakeDetector()

    # Test with a sample image (if available)
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        result = detector.analyze(test_image)
        print("Analysis Result:")
        print(result)
    else:
        print("No test image found. Create a test_image.jpg to test the detector.")
