import streamlit as st
from PIL import Image
import os
import tempfile
from detector import DeepfakeDetector, MultiStageDetector
from video_processor import VideoProcessor, analyze_extracted_frames

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .authentic {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .ai-generated {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .frame-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem;
    }
    /* Limit video preview height */
    [data-testid="stVideo"] {
        max-height: 400px;
    }
    [data-testid="stVideo"] video {
        max-height: 400px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

# Initialize detector (load model once)
if 'detector' not in st.session_state:
    with st.spinner('Loading multi-stage deepfake detection system... (first run may take a few minutes)'):
        st.session_state.detector = MultiStageDetector(verbose=False)

# Header
st.markdown('<h1 class="main-header">Deepfake Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Detect AI-generated videos</p>', unsafe_allow_html=True)

# Video Analysis
st.subheader("Upload a Video")
st.info("ℹ️ **Analysis:** All frames will be analyzed (1 frame every 0.5 seconds)")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Supported formats: MP4, AVI, MOV, MKV, WEBM. Max 1 minute duration.",
    key="video_uploader"
)

if uploaded_video is not None:
    # Check file size (max 200MB)
    max_file_size_mb = 200
    file_size_mb = uploaded_video.size / (1024 * 1024)

    if file_size_mb > max_file_size_mb:
        st.error(f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({max_file_size_mb} MB). Please upload a smaller video.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Uploaded Video:**")
        st.video(uploaded_video)

    with col2:
        st.write("**Video Info:**")
        st.info(f"Filename: {uploaded_video.name}\n\nSize: {file_size_mb:.1f} MB")

    if st.button("Analyze Video", type="primary", width='stretch'):
        # First, save and validate the video
        with st.spinner('Validating video...'):
            try:
                # Save video
                processor = VideoProcessor()
                video_path = processor.save_uploaded_video(uploaded_video)

                # Validate duration
                validation = processor.validate_video(video_path)

                if not validation['valid']:
                    st.error(validation['error'])
                    os.unlink(video_path)
                    st.stop()

                # Get video info
                video_info = processor.get_video_info(video_path)

                # Downscale if necessary
                if validation['width'] > processor.MAX_WIDTH or validation['height'] > processor.MAX_HEIGHT:
                    with st.spinner(f'Downscaling video from {validation["width"]}x{validation["height"]}...'):
                        downscaled_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        scale_info = processor.downscale_video(video_path, downscaled_path)

                        # Clean up original, use downscaled
                        os.unlink(video_path)
                        video_path = downscaled_path

                        st.info(f"Video downscaled from {scale_info['original_width']}x{scale_info['original_height']} to {scale_info['new_width']}x{scale_info['new_height']}")

                        # Update video info with new dimensions
                        video_info = processor.get_video_info(video_path)

            except Exception as e:
                st.error(f"Error validating video: {str(e)}")
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.unlink(video_path)
                st.stop()

        # Show video metadata
        st.write("**Video Details:**")
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
        with info_cols[1]:
            st.metric("FPS", f"{video_info['fps']:.1f}")
        with info_cols[2]:
            st.metric("Frames", video_info['total_frames'])
        with info_cols[3]:
            st.metric("Duration", f"{video_info['duration_seconds']:.1f}s")

        # Extract frames
        with st.spinner('Extracting frames from video (1 frame every 0.5 seconds)...'):
            try:
                st.write("---")
                st.subheader("Extracted Frames")
                frames = processor.extract_frames(video_path)
                st.caption(f"📊 Analyzing {len(frames)} frames from this video")

                # Display extracted frames in a grid (smaller thumbnails)
                cols_per_row = 10
                for i in range(0, len(frames), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(frames):
                            with col:
                                st.image(frames[idx], width='stretch')
                                st.caption(f"Frame {idx + 1}", unsafe_allow_html=False)

            except Exception as e:
                st.error(f"Error extracting frames: {str(e)}")
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.unlink(video_path)
                st.stop()

        # Now analyze the frames
        with st.spinner('Analyzing frames for deepfakes...'):
            try:
                # Analyze the already-extracted frames
                results = analyze_extracted_frames(frames, st.session_state.detector)
                os.unlink(video_path)

                # Overall verdict
                st.write("---")
                st.subheader("Overall Verdict")

                aggregate = results['aggregate']
                overall_is_fake = aggregate['overall_verdict'] == 'LIKELY DEEPFAKE'
                verdict_class = "ai-generated" if overall_is_fake else "authentic"
                verdict_emoji = "🔴" if overall_is_fake else "🟢"

                st.markdown(f'<div class="result-box {verdict_class}">', unsafe_allow_html=True)
                st.markdown(f"### {verdict_emoji} {aggregate['overall_verdict']}")
                st.markdown(f"**{aggregate['deepfake_frames']}** out of **{results['frames_analyzed']}** frames detected as AI-generated ({aggregate.get('fake_percentage', 0)}%)")
                st.markdown('</div>', unsafe_allow_html=True)

                # Show category and detector breakdown
                if 'category_breakdown' in aggregate and 'detector_breakdown' in aggregate:
                    st.write("**Analysis Breakdown:**")
                    breakdown_cols = st.columns(2)

                    with breakdown_cols[0]:
                        st.write("*Categories Detected:*")
                        for category, count in aggregate['category_breakdown'].items():
                            category_label = {
                                'human_face': 'Human Face',
                                'other': 'Object/Scene',
                                'unknown': 'Unknown'
                            }.get(category, category)
                            st.write(f"- {category_label}: {count} frame(s)")

                    with breakdown_cols[1]:
                        st.write("*Detectors Used:*")
                        for detector, count in aggregate['detector_breakdown'].items():
                            detector_label = {
                                'face_specialized': 'Face Specialist (ViT)',
                                'general': 'General AI Detector',
                                'general (fallback)': 'General (Fallback)'
                            }.get(detector, detector)
                            st.write(f"- {detector_label}: {count} frame(s)")

                # Frame-by-frame results
                st.write("---")
                st.subheader("Frame Analysis Results")

                # Display results in a grid (compact layout)
                frame_results = results['frame_results']
                results_cols_per_row = 10

                for i in range(0, len(frame_results), results_cols_per_row):
                    cols = st.columns(results_cols_per_row)

                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(frame_results):
                            frame_result = frame_results[idx]
                            frame_num = frame_result['frame_number']
                            is_fake = frame_result['is_deepfake']
                            confidence = frame_result['confidence']
                            frame_img = frame_result['frame_image']

                            with col:
                                # Show frame thumbnail (smaller)
                                st.image(frame_img, width='stretch')

                                # Show verdict (compact)
                                verdict_emoji = "🔴" if is_fake else "🟢"
                                verdict_text = "Fake" if is_fake else "Real"
                                st.markdown(f"**#{frame_num}** {verdict_emoji} {verdict_text}")
                                st.caption(f"Confidence: {confidence}%")

                                # Show category and detector (compact)
                                category = frame_result.get('category', 'unknown')
                                category_icon = '👤' if category == 'human_face' else '🖼️'
                                detector = frame_result.get('detector_used', 'unknown')
                                detector_short = 'Face' if 'face' in detector else 'General'

                                st.caption(f"{category_icon} {detector_short}")

                # Show success message at the end
                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.unlink(video_path)

# Sidebar with information
with st.sidebar:
    st.header("About")

    st.write("""
    This tool uses a **multi-stage detection system** to analyze videos for deepfakes.

    **How it works:**
    1. **Frame Classification:** CLIP categorizes each frame
    2. **Specialized Detection:**
       - Human faces → Face-specific ViT detector
       - Other content → General AI detector

    **Models Used:**
    - **CLIP:** openai/clip-vit-base-patch32 (categorization)
    - **Face Detector:** MTCNN (face detection & cropping)
    - **Face Deepfake:** prithivMLmods/deepfake-detector-model-v1 (94.44% accuracy)
    - **General AI:** Ateeqq/ai-vs-human-image-detector (99.23% accuracy)

    **Note:** This multi-stage approach provides better accuracy by using specialized models for different content types.
    """)

    st.write("---")
    st.caption("Built with Streamlit, HuggingFace Transformers & CLIP")
