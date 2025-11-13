# Deepfake Detection System

A multi-stage deepfake detection system for video analysis. This tool uses CLIP for content categorization and specialized AI detectors to identify deepfakes and AI-generated content in videos.

## Features

- **Video Analysis**: Upload videos (up to 60 seconds, max 200MB)
- **Multi-Stage Detection Pipeline**:
  - CLIP categorizes each frame (human face vs. other content)
  - MTCNN detects and crops faces from frames
  - Specialized detectors for different content types
- **High Accuracy**: 94.44% for faces, 99.23% for general content
- **Frame-by-Frame Analysis**: See results for every frame
- **Automatic Downscaling**: Videos > 1920x1080 are automatically downscaled
- **Web Interface**: Clean Streamlit-based UI
- **Docker Support**: Easy deployment with Docker/Docker Compose

## How It Works

```
Video Upload → Validation → Downscaling (if needed) → Frame Extraction (0.5s intervals)
    ↓
For Each Frame:
    ↓
CLIP Classification → "human_face" or "other"
    ↓                        ↓
Face Pipeline          General Pipeline
    ↓                        ↓
MTCNN Detection       General AI Detector
    ↓                   (99.23% accuracy)
Crop Face
    ↓
Face Deepfake Detector
(94.44% accuracy)
    ↓
Aggregate Results → Overall Verdict
```

## Models Used

| Component | Model | Purpose | Accuracy |
|-----------|-------|---------|----------|
| **Frame Classifier** | openai/clip-vit-base-patch32 | Categorize content | Zero-shot |
| **Face Detector** | MTCNN (facenet-pytorch) | Detect & crop faces | Detection only |
| **Face Deepfake Detector** | prithivMLmods/deepfake-detector-model-v1 | Detect face deepfakes | 94.44% |
| **General AI Detector** | Ateeqq/ai-vs-human-image-detector | Detect AI-generated content | 99.23% |

**Total Model Size**: ~1.4GB (downloaded on first run)

## Project Structure

```
Deepfake/
├── streamlit_app.py          # Main Streamlit web interface
├── detector.py               # Multi-stage detector & specialized detectors
├── video_processor.py        # Video handling, frame extraction
├── frame_classifier.py       # CLIP-based content categorization
├── face_detector.py          # MTCNN face detection & cropping
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose setup
├── uml_class_simple.puml    # Class diagram
└── uml_pipeline_simple.puml # Pipeline flow diagram
```

## Installation & Usage

### Option 1: Docker (Recommended)

**Prerequisites**: Docker & Docker Compose installed

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Access at http://localhost:8501
```

**First run**: Models will download (~1.4GB, takes 5-15 minutes)

### Option 2: Local Installation

**Prerequisites**: Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

## Video Requirements

- **Formats**: MP4, AVI, MOV, MKV, WEBM
- **Max Duration**: 60 seconds
- **Max File Size**: 200MB
- **Min Resolution**: 854x480 (480p)
- **Max Resolution**: 1920x1080 (1080p) - auto-downscaled if larger

## Analysis Process

1. **Upload Video**: Drag and drop or select video file
2. **Validation**: System checks duration and resolution
3. **Downscaling**: If resolution > 1920x1080, video is automatically downscaled
4. **Frame Extraction**: Extracts 1 frame every 0.5 seconds
5. **Multi-Stage Detection**:
   - CLIP categorizes each frame
   - If "human_face": MTCNN crops face → Face Deepfake Detector
   - If "other": General AI Detector
   - Fallback to General Detector if no face found
6. **Aggregation**: Overall verdict based on % of fake frames (>60% = LIKELY DEEPFAKE)
7. **Results**: Frame-by-frame analysis + overall verdict

## Technology Stack

- **Frontend**: Streamlit 1.30+
- **ML Framework**: PyTorch 2.6+ (CPU-only)
- **Transformers**: HuggingFace Transformers 4.40+
- **Video Processing**: OpenCV 4.8
- **Face Detection**: facenet-pytorch (MTCNN)
- **Content Classification**: CLIP (openai/clip-vit-base-patch32)
- **Deployment**: Docker, Docker Compose
- **Language**: Python 3.10

## System Requirements

### Local Installation
- Python 3.10+
- 8GB RAM minimum
- 10GB disk space (for models)
- Internet connection (first run only)

### Docker Installation
- Docker 20.10+
- Docker Compose 1.29+
- 8GB RAM
- 10GB disk space

## Configuration

### Video Processor Settings
Located in `video_processor.py`:
```python
MAX_DURATION_SECONDS = 60    # Max video duration
MAX_WIDTH = 1920             # Max resolution width
MAX_HEIGHT = 1080            # Max resolution height
MIN_WIDTH = 854              # Min resolution width (480p)
MIN_HEIGHT = 480             # Min resolution height
```

### Detection Thresholds
Located in `detector.py`:
```python
# Both detectors use 70% threshold
is_deepfake = ai_confidence > 0.7

# Video verdict threshold (in video_processor.py)
overall_verdict = 'LIKELY DEEPFAKE' if fake_percentage > 0.6 else 'LIKELY AUTHENTIC'
```

## Docker Details

See `DOCKER_README.md` for complete Docker documentation including:
- Resource limits and configuration
- Troubleshooting
- Production deployment tips
- Model caching

## UML Diagrams

Two simple diagrams are provided:
- `uml_class_simple.puml` - Class structure
- `uml_pipeline_simple.puml` - Pipeline flow

View at: http://www.plantuml.com/plantuml/uml/

## Limitations & Future Considerations

### Current Limitations
- **Video only**: No single image analysis
- **60 second limit**: Longer videos must be trimmed
- **CPU-only**: No GPU acceleration (for compatibility)
- **Model age**: Trained on 2023-2024 era deepfakes/AI content
- **Temporal analysis**: Analyzes frames independently (no motion analysis)

### Obsolescence Risk
Detection models are trained on specific generation techniques. Newer AI models (Sora 2, latest diffusion models, advanced face swaps) may not be effectively detected without retraining on current datasets.

**Estimated Relevance**: 6-18 months without model updates

### Mitigation Strategies
To extend relevance:
- Regular model retraining on new synthetic media
- Ensemble of multiple detector versions
- Complementary metadata/forensics analysis
- Authentication-based approaches (C2PA, blockchain verification)

## Performance

- **First Run**: 5-15 minutes (model download)
- **Video Processing**: ~2-5 seconds per frame
- **Memory Usage**: 4-8GB during analysis
- **Recommended**: 4 CPU cores for smooth operation

## Troubleshooting

### Docker Issues
```bash
# Check logs
docker-compose logs deepfake-detector

# Restart
docker-compose restart

# Rebuild
docker-compose build --no-cache
```

### Port Conflicts
Change port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Out of Memory
Increase Docker memory limit or reduce video resolution/duration

### Models Not Downloading
Check internet connection and HuggingFace availability:
```bash
curl -I https://huggingface.co
```

## Security Notice

This tool is designed for:
- ✅ Detecting manipulated media
- ✅ Verifying content authenticity
- ✅ Research and educational purposes
- ✅ Defensive security applications

**Not for**:
- ❌ Creating deepfakes
- ❌ Malicious content generation
- ❌ Privacy invasion

## API Structure

While primarily a web UI, the detection system can be used programmatically:

```python
from detector import MultiStageDetector
from video_processor import VideoProcessor, analyze_extracted_frames

# Initialize
detector = MultiStageDetector(verbose=False)
processor = VideoProcessor()

# Process video
frames = processor.extract_frames('video.mp4', frame_interval_seconds=0.5)
results = analyze_extracted_frames(frames, detector)

# Get verdict
print(results['aggregate']['overall_verdict'])
print(f"{results['aggregate']['fake_percentage']}% of frames are deepfakes")
```

## Contributing

Areas for improvement:
1. GPU acceleration support
2. Longer video support (>60s)
3. Temporal consistency analysis
4. Model ensemble approaches
5. Real-time video stream analysis
6. Additional model integrations

## License

Educational and defensive security purposes.

## Citation

If using this project for research:

```
Deepfake Detection System (2024-2025)
Multi-stage video deepfake detection using CLIP, MTCNN, and specialized detectors
```

**Models**:
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
- MTCNN: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- Face Detector: prithivMLmods/deepfake-detector-model-v1
- General Detector: Ateeqq/ai-vs-human-image-detector

## Acknowledgments

- OpenAI for CLIP
- HuggingFace for model hosting and Transformers library
- facenet-pytorch for MTCNN implementation
- Model creators: prithivMLmods, Ateeqq

---

**Last Updated**: 2025
**Status**: Active Development
**Version**: 1.0
