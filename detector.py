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

    def __init__(self, model_name=None, verbose=False):
        """
        Initialize the detector.

        Args:
            model_name: HuggingFace model name (default: Ateeqq/ai-vs-human-image-detector)
            verbose: Print debug output during inference (default: False)
        """
        if model_name is None:
            model_name = "Ateeqq/ai-vs-human-image-detector"

        self.model_name = model_name
        self.model = None
        self.verbose = verbose
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

                # Debug: Print raw predictions to understand format (only if verbose)
                if self.verbose:
                    print("=" * 50)
                    print("RAW MODEL OUTPUT:")
                    print(predictions)
                    print("=" * 50)

                # Extract results
                # Expected format: [{'label': 'ai', 'score': 0.95}, {'label': 'hum', 'score': 0.05}]
                # DEBUG: Print raw predictions to diagnose label issue
                print("\n" + "=" * 60)
                print("DEBUG - GENERAL DETECTOR RAW OUTPUT:")
                for pred in predictions:
                    print(f"  Label: '{pred['label']}' -> Score: {pred['score']:.4f}")
                print("=" * 60 + "\n")

                ai_score = 0.0
                human_score = 0.0

                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']

                    if 'ai' in label or 'fake' in label or 'generated' in label or 'artificial' in label:
                        ai_score = score
                    elif 'hum' in label or 'real' in label or 'authentic' in label or 'natural' in label:
                        human_score = score

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
            # Use higher threshold to reduce false positives (70% instead of 50%)
            is_deepfake = ai_confidence > 0.7

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


class FaceDeepfakeDetector:
    """
    Specialized deepfake detector for human faces using SigLIP.

    Uses: prithivMLmods/deepfake-detector-model-v1
    - Fine-tuned from google/siglip-base-patch16-512 for binary deepfake classification
    - Optimized for cropped face images
    - Detects face swaps, AI-synthesized faces, and facial manipulations
    - 94.44% accuracy (precision: 97.18% fake, 92.01% real)
    """

    def __init__(self, model_name=None, verbose=False):
        """
        Initialize the face deepfake detector.

        Args:
            model_name: HuggingFace model name (default: prithivMLmods/deepfake-detector-model-v1)
            verbose: Print debug output during inference (default: False)
        """
        if model_name is None:
            model_name = "prithivMLmods/deepfake-detector-model-v1"

        self.model_name = model_name
        self.model = None
        self.verbose = verbose
        self._load_model()

    def _load_model(self):
        """
        Load the face deepfake detection model.
        Model will be downloaded on first run.
        Supports LoRA fine-tuned models from models/celeb_df_finetuned/
        """
        try:
            import os
            from pathlib import Path

            # Check for fine-tuned model
            finetuned_path = Path("models/celeb_df_finetuned")

            if finetuned_path.exists() and (finetuned_path / "adapter_model.safetensors").exists():
                print(f"🎯 Found fine-tuned LoRA model at {finetuned_path}")
                print(f"Loading base model: {self.model_name}")
                print("Note: First run will download the base model, please wait...")

                # Import PEFT for LoRA
                try:
                    from peft import PeftModel
                    from transformers import AutoModelForImageClassification, AutoImageProcessor

                    # Load base model
                    base_model = AutoModelForImageClassification.from_pretrained(
                        self.model_name,
                        num_labels=2,
                        ignore_mismatched_sizes=True
                    )

                    # Load LoRA adapters
                    print(f"Loading LoRA adapters from {finetuned_path}...")
                    peft_model = PeftModel.from_pretrained(base_model, str(finetuned_path))

                    # Merge LoRA weights into base model for pipeline compatibility
                    print("Merging LoRA weights into base model...")
                    model = peft_model.merge_and_unload()

                    # Create pipeline with merged model
                    processor = AutoImageProcessor.from_pretrained(str(finetuned_path))

                    self.model = pipeline(
                        'image-classification',
                        model=model,
                        image_processor=processor,
                        device=-1  # Use CPU (-1), change to 0 for GPU if available
                    )

                    print("✅ Fine-tuned LoRA model loaded successfully!")
                    print("   Model trained on Celeb-DF-v2 (85.9% accuracy, 83.8% F1)")

                except ImportError:
                    print("⚠️  PEFT library not found. Installing...")
                    import subprocess
                    subprocess.check_call(["pip", "install", "-q", "peft"])
                    print("Please restart the script to use the fine-tuned model.")
                    # Fall back to base model
                    self.model = pipeline(
                        'image-classification',
                        model=self.model_name,
                        device=-1
                    )

            else:
                # Use original base model
                print(f"Loading face deepfake detection model: {self.model_name}")
                print("Note: First run will download the model, please wait...")
                print("💡 Tip: Fine-tuned model not found. Place it in models/celeb_df_finetuned/ for better accuracy.")

                self.model = pipeline(
                    'image-classification',
                    model=self.model_name,
                    device=-1  # Use CPU (-1), change to 0 for GPU if available
                )

                print("Face deepfake detection model loaded successfully!")

        except Exception as e:
            print(f"Error loading face model: {e}")
            print("Falling back to placeholder mode...")
            self.model = None

    def _preprocess_image(self, image_path):
        """
        Preprocess face image for model input.

        Args:
            image_path: Path to the face image file

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
        Run the face deepfake detection model inference.

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

                # Debug: Print raw predictions (only if verbose)
                if self.verbose:
                    print("=" * 50)
                    print("RAW FACE MODEL OUTPUT:")
                    print(predictions)
                    print("=" * 50)

                # Extract results
                # Expected format: [{'label': 'REAL' or 'FAKE', 'score': ...}]
                # DEBUG: Print raw predictions to diagnose label issue
                print("\n" + "=" * 60)
                print("DEBUG - FACE DETECTOR RAW OUTPUT:")
                for pred in predictions:
                    print(f"  Label: '{pred['label']}' -> Score: {pred['score']:.4f}")
                print("=" * 60 + "\n")

                fake_score = 0.0
                real_score = 0.0

                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']

                    if 'fake' in label or 'deepfake' in label or 'synthetic' in label or 'ai' in label:
                        fake_score = score
                    elif 'real' in label or 'authentic' in label or 'genuine' in label:
                        real_score = score

                return {
                    'ai_confidence': float(fake_score),
                    'human_confidence': float(real_score),
                    'predictions': predictions,
                    'using_real_model': True
                }

            except Exception as e:
                print(f"Error during face inference: {e}")
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
        Analyze a face image for deepfake detection.

        Args:
            image_path: Path to the face image file

        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate image exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Face image file not found'
                }

            # Preprocess image
            preprocessed = self._preprocess_image(image_path)

            # Run inference
            predictions = self._run_inference(preprocessed)

            # Extract deepfake confidence
            ai_confidence = predictions['ai_confidence']
            real_confidence = predictions.get('human_confidence', 1.0 - ai_confidence)

            # Use higher threshold to reduce false positives (70% instead of 50%)
            is_deepfake = ai_confidence > 0.7

            # Confidence in the actual prediction made
            prediction_confidence = ai_confidence if is_deepfake else real_confidence

            # Determine confidence level label
            if prediction_confidence > 0.85:
                confidence_label = 'high'
            elif prediction_confidence > 0.65:
                confidence_label = 'medium'
            else:
                confidence_label = 'low'

            # Prepare response
            result = {
                'success': True,
                'is_deepfake': is_deepfake,
                'confidence': round(prediction_confidence * 100, 2),
                'confidence_label': confidence_label,
                'verdict': 'DEEPFAKE FACE' if is_deepfake else 'AUTHENTIC FACE',
                'fake_probability': round(ai_confidence * 100, 2),
                'real_probability': round(real_confidence * 100, 2),
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
                result['note'] = 'Warning: Using placeholder detection. Install dependencies and restart to use the real face model.'
            else:
                result['note'] = f'Face analysis performed using {self.model_name} (specialized for faces).'

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class MultiStageDetector:
    """
    Multi-stage deepfake detector that categorizes frames and applies specialized detectors.

    Pipeline:
    1. Classify frame category (human face vs other) using CLIP
    2. If human face: detect & crop face → specialized face deepfake detector
    3. If other: general AI-generated image detector
    """

    def __init__(self, verbose=False):
        """
        Initialize the multi-stage detector.

        Args:
            verbose: Print debug output during processing (default: False)
        """
        from frame_classifier import FrameClassifier
        from face_detector import FaceDetector

        self.verbose = verbose

        # Initialize all components
        print("Initializing Multi-Stage Deepfake Detection System...")

        # Stage 1: Frame classifier
        print("Loading frame classifier (CLIP)...")
        self.frame_classifier = FrameClassifier(verbose=verbose)

        # Stage 2a: Face detector (for face cropping)
        print("Loading face detector (MTCNN)...")
        self.face_detector = FaceDetector(verbose=verbose)

        # Stage 2b: Specialized detectors
        print("Loading specialized deepfake detectors...")
        self.face_deepfake_detector = FaceDeepfakeDetector(verbose=verbose)
        self.general_detector = DeepfakeDetector(verbose=verbose)

        print("Multi-Stage Detection System ready!")

    def analyze(self, image_path):
        """
        Analyze an image using the multi-stage detection pipeline.

        Args:
            image_path: Path to the image file or PIL Image object

        Returns:
            Dictionary with comprehensive analysis results including:
            - Category classification
            - Deepfake detection results
            - Detector used
            - Face detection info (if applicable)
        """
        import tempfile

        try:
            # Handle PIL Image objects - save temporarily
            temp_file = None
            if isinstance(image_path, Image.Image):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                image_path.save(temp_file.name, 'JPEG')
                image_path = temp_file.name

            # Stage 1: Classify frame category
            if self.verbose:
                print("\n=== Stage 1: Frame Classification ===")

            classification = self.frame_classifier.classify(image_path)

            if not classification['success']:
                # Fallback to general detector if classification fails
                if self.verbose:
                    print("Classification failed, using general detector")

                result = self.general_detector.analyze(image_path)
                result['category'] = 'unknown'
                result['category_confidence'] = 0.0
                result['detector_used'] = 'general'

                if temp_file:
                    os.unlink(temp_file.name)

                return result

            category = classification['category']
            category_confidence = classification['confidence']

            if self.verbose:
                print(f"Category: {category} (confidence: {category_confidence:.3f})")

            # Stage 2: Apply specialized detector based on category
            if category == 'human_face':
                # Human face pipeline
                if self.verbose:
                    print("\n=== Stage 2a: Face Detection & Cropping ===")

                # Detect and crop face
                face_result = self.face_detector.detect_and_crop_faces(
                    image_path,
                    return_all=False,  # Only analyze the most prominent face
                    margin=0.2
                )

                if not face_result['success'] or face_result['faces_found'] == 0:
                    # No face detected, fallback to general detector
                    if self.verbose:
                        print("No face detected, falling back to general detector")

                    result = self.general_detector.analyze(image_path)
                    result['category'] = 'human_face_attempt'
                    result['category_confidence'] = category_confidence
                    result['detector_used'] = 'general (fallback)'
                    result['face_detection_failed'] = True

                    if temp_file:
                        os.unlink(temp_file.name)

                    return result

                # Face detected successfully
                cropped_face = face_result['cropped_faces'][0]
                face_box = face_result['boxes'][0]
                face_confidence = face_result['confidences'][0]

                if self.verbose:
                    print(f"Face detected with confidence: {face_confidence:.3f}")
                    print(f"Face box: {face_box}")
                    print("\n=== Stage 2b: Face Deepfake Analysis ===")

                # Save cropped face temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cropped_face.save(tmp.name, 'JPEG')
                    face_path = tmp.name

                # Analyze with specialized face detector
                result = self.face_deepfake_detector.analyze(face_path)

                # Clean up temp face file
                os.unlink(face_path)

                # Add metadata
                result['category'] = 'human_face'
                result['category_confidence'] = category_confidence
                result['detector_used'] = 'face_specialized'
                result['face_info'] = {
                    'face_detected': True,
                    'face_confidence': face_confidence,
                    'face_box': face_box
                }

            else:
                # Other content (objects, landscapes, etc.)
                if self.verbose:
                    print(f"\n=== Stage 2: General AI Detection (Category: {category}) ===")

                # Use general AI-generated image detector
                result = self.general_detector.analyze(image_path)

                # Add metadata
                result['category'] = category
                result['category_confidence'] = category_confidence
                result['detector_used'] = 'general'

            # Clean up temp file if created
            if temp_file:
                os.unlink(temp_file.name)

            if self.verbose:
                print(f"\n=== Final Result ===")
                print(f"Category: {result['category']}")
                print(f"Verdict: {result['verdict']}")
                print(f"Confidence: {result['confidence']}%")
                print(f"Detector used: {result['detector_used']}")

            return result

        except Exception as e:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            return {
                'success': False,
                'error': str(e),
                'category': 'error',
                'detector_used': 'none'
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
