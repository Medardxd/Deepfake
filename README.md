# Deepfake Detection System

A web-based deepfake detection tool built with Flask and Python. This defensive security application allows users to upload images and analyze them for potential AI-generated or manipulated content.

## Features

- Clean, modern web interface with drag-and-drop upload
- Image upload and analysis
- Confidence scoring and detailed results
- Modular architecture for easy ML model integration
- RESTful API for backend processing
- Responsive design for mobile and desktop

## Project Structure

```
deepfake-detector/
├── app.py                 # Flask web server
├── detector.py            # Deepfake detection logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── css/
│   │   └── styles.css    # Styles
│   └── js/
│       └── app.js        # Frontend JavaScript
├── uploads/              # Uploaded images (created automatically)
└── models/               # ML models directory
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. Navigate to the project directory:
```bash
cd deepfake-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Linux/Mac:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image using the drag-and-drop interface or file selector

4. Click "Analyze Image" to process the image

5. View the results showing whether the image appears to be authentic or a deepfake

## API Endpoints

### POST `/api/analyze`
Upload and analyze an image for deepfake detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` file (PNG, JPG, JPEG, GIF, BMP, max 16MB)

**Response:**
```json
{
    "success": true,
    "is_deepfake": false,
    "confidence": 75.23,
    "confidence_label": "high",
    "verdict": "APPEARS AUTHENTIC",
    "image_info": {
        "width": 1920,
        "height": 1080,
        "format": "JPEG"
    },
    "note": "This is a placeholder implementation..."
}
```

### GET `/api/health`
Check if the backend service is running.

**Response:**
```json
{
    "status": "ok",
    "detector": "ready"
}
```

## Integrating a Real ML Model

The current implementation uses a placeholder for the deepfake detection logic. To integrate a real ML model:

1. **Train or obtain a deepfake detection model**
   - TensorFlow/Keras (.h5, .pb)
   - PyTorch (.pt, .pth)
   - ONNX (.onnx)

2. **Place your model in the `models/` directory**

3. **Update `detector.py`:**

```python
# Example for TensorFlow/Keras
def _load_model(self, model_path):
    import tensorflow as tf
    self.model = tf.keras.models.load_model(model_path)

# Example for PyTorch
def _load_model(self, model_path):
    import torch
    self.model = torch.load(model_path)
    self.model.eval()
```

4. **Update preprocessing in `_preprocess_image()`:**
   - Resize images to match your model's input size
   - Normalize pixel values
   - Convert to appropriate format (numpy array, tensor, etc.)

5. **Update inference in `_run_inference()`:**
   - Replace placeholder logic with actual model prediction
   - Process model output to get confidence scores

6. **Uncomment relevant dependencies in `requirements.txt`:**
```txt
tensorflow==2.15.0  # For TensorFlow models
# or
torch==2.1.1        # For PyTorch models
```

## Future Enhancements

- Video analysis with frame extraction
- Batch processing for multiple images
- Model comparison (test multiple models)
- Detailed heatmaps showing manipulated regions
- User authentication and history
- Database for storing analysis results
- Advanced reporting and statistics

## Security Notice

This tool is designed for defensive security purposes only. It should be used to:
- Detect potentially manipulated media
- Verify authenticity of images
- Research deepfake detection techniques
- Educational purposes

## Technology Stack

- **Backend:** Flask (Python web framework)
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **Image Processing:** Pillow (PIL)
- **ML Ready:** Supports TensorFlow, PyTorch, ONNX integration

## Troubleshooting

### Port already in use
If port 5000 is already in use, change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### File upload errors
- Ensure the `uploads/` directory exists and is writable
- Check file size (max 16MB by default)
- Verify file format is supported

### Module import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

This project is for educational and defensive security purposes.

## Contributing

To contribute to this project:
1. Test with various image types
2. Implement real ML models
3. Improve UI/UX
4. Add video processing capabilities
5. Enhance error handling

## Contact

For questions or issues, please create an issue in the project repository.
