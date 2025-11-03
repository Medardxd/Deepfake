import streamlit as st
from PIL import Image
import os
import tempfile
from detector import DeepfakeDetector

# Page configuration
st.set_page_config(
    page_title="AI Image Detector - Testing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .authentic {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .ai-generated {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    with st.spinner('🔄 Loading AI detection model... (this may take ~1 minute on first run)'):
        st.session_state.detector = DeepfakeDetector()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model Info")
    st.info("""
    **Model**: Ateeqq/ai-vs-human-image-detector

    **Accuracy**: 99.23%

    **Detects**: All image types (faces, landscapes, art, etc.)

    **Trained on**: Midjourney v6.1, FLUX, SD 3.5, GPT-4o
    """)

    st.subheader("Testing Features")
    show_debug = st.checkbox("Show Debug Info", value=False)
    show_raw_scores = st.checkbox("Show Raw Model Output", value=False)

    st.subheader("Batch Testing")
    st.write("Upload multiple images to test at once")

# Main header
st.markdown('<h1 class="main-header">🔍 AI Image Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Development & Testing Interface | Powered by Streamlit</p>', unsafe_allow_html=True)

# Create tabs for different testing modes
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Testing", "📊 About"])

# Tab 1: Single Image Testing
with tab1:
    st.header("Upload an Image to Analyze")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        help="Supported formats: PNG, JPG, JPEG, GIF, BMP (Max 16MB)"
    )

    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Uploaded Image")
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')

            # Show image metadata
            st.caption(f"📏 Size: {image.size[0]} x {image.size[1]} | Format: {image.format}")

        with col2:
            st.subheader("🔬 Analysis")

            # Analyze button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner('🧠 Analyzing image...'):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Run detection
                    result = st.session_state.detector.analyze(tmp_path)

                    # Clean up temp file
                    os.unlink(tmp_path)

                    # Display results
                    if result['success']:
                        # Verdict
                        is_ai = result['is_deepfake']
                        verdict_class = "ai-generated" if is_ai else "authentic"
                        verdict_color = "🔴" if is_ai else "🟢"

                        st.markdown(f'<div class="result-box {verdict_class}">', unsafe_allow_html=True)
                        st.markdown(f"### {verdict_color} {result['verdict']}")
                        st.markdown(f"**Confidence**: {result['confidence']}%")
                        st.markdown(f"**Level**: {result['confidence_label'].upper()}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Metrics
                        st.subheader("📊 Detailed Metrics")

                        metric_col1, metric_col2 = st.columns(2)

                        with metric_col1:
                            st.metric(
                                label="AI Probability",
                                value=f"{result['confidence']}%",
                                delta=f"{result['confidence'] - 50:+.2f}% from threshold"
                            )

                        with metric_col2:
                            st.metric(
                                label="Human Probability",
                                value=f"{100 - result['confidence']:.2f}%"
                            )

                        # Progress bar visualization
                        st.subheader("📈 Confidence Visualization")
                        st.progress(result['confidence'] / 100)

                        # Model info
                        if show_debug:
                            st.subheader("🔧 Debug Information")
                            st.json(result)

                        # Note
                        st.info(f"ℹ️ {result.get('note', 'Analysis complete')}")

                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")

# Tab 2: Batch Testing
with tab2:
    st.header("Batch Image Analysis")
    st.write("Upload multiple images to test them all at once - great for testing model performance!")

    uploaded_files = st.file_uploader(
        "Choose multiple images",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files:
        st.write(f"📦 {len(uploaded_files)} images uploaded")

        if st.button("🚀 Analyze All Images", type="primary"):
            results_summary = {"ai": 0, "human": 0, "total": len(uploaded_files)}

            # Progress bar for batch processing
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Results container
            results_container = st.container()

            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                # Save and analyze
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                result = st.session_state.detector.analyze(tmp_path)
                os.unlink(tmp_path)

                # Count results
                if result['success']:
                    if result['is_deepfake']:
                        results_summary['ai'] += 1
                    else:
                        results_summary['human'] += 1

                # Display individual result
                with results_container:
                    col1, col2, col3 = st.columns([2, 3, 1])
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, width=150)
                    with col2:
                        st.write(f"**{uploaded_file.name}**")
                        if result['success']:
                            verdict_emoji = "🔴" if result['is_deepfake'] else "🟢"
                            st.write(f"{verdict_emoji} {result['verdict']}")
                            st.write(f"Confidence: {result['confidence']}%")
                        else:
                            st.write("❌ Error analyzing")
                    with col3:
                        if result['success']:
                            st.metric("Score", f"{result['confidence']}%")

                    st.divider()

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            # Summary
            st.success("✅ Batch analysis complete!")

            st.subheader("📊 Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Total Images", results_summary['total'])
            with summary_col2:
                st.metric("AI-Generated", results_summary['ai'])
            with summary_col3:
                st.metric("Authentic", results_summary['human'])

# Tab 3: About
with tab3:
    st.header("📖 About This Testing Interface")

    st.markdown("""
    ## Purpose
    This Streamlit interface is designed for **rapid testing and development** of the AI image detection system.

    ## Why Use This?
    - ✅ **Fast Iteration**: Test new features without touching HTML/CSS/JS
    - ✅ **Easy Debugging**: See raw outputs and debug info easily
    - ✅ **Batch Testing**: Test multiple images at once
    - ✅ **Clean Separation**: ML logic testing separate from UI development

    ## Workflow
    1. **Test here first** - Verify ML functionality works correctly
    2. **Debug easily** - Use debug mode to see raw outputs
    3. **Batch test** - Test model on multiple images
    4. **Port to Flask** - Once verified, implement in production UI

    ## Technical Details
    - **Framework**: Streamlit (Python-only web framework)
    - **Model**: Same `detector.py` used by Flask app
    - **Purpose**: Development & Testing ONLY
    - **Production**: Use Flask app for final product

    ## Features Not in Flask Version
    - 🔄 Batch image processing
    - 🐛 Debug mode with raw outputs
    - 📊 Visual confidence metrics
    - ⚡ Faster prototyping of new features

    ## Model Information
    - **Name**: Ateeqq/ai-vs-human-image-detector
    - **Architecture**: SiglipForImageClassification
    - **Accuracy**: 99.23%
    - **Training Data**: 60k AI + 60k Human images
    - **Detects**: Midjourney, FLUX, Stable Diffusion, GPT-4o, and more

    ---

    💡 **Pro Tip**: Use this for testing, then implement features in the Flask app for a polished production interface!
    """)

    st.subheader("🔗 Comparison")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.markdown("""
        ### Streamlit (This App)
        - ✅ Pure Python
        - ✅ Rapid prototyping
        - ✅ Easy debugging
        - ✅ Batch processing
        - ❌ Less customizable
        - ❌ Different look/feel
        """)

    with comparison_col2:
        st.markdown("""
        ### Flask (Production)
        - ✅ Full control
        - ✅ Custom design
        - ✅ Professional UI
        - ✅ Better for portfolio
        - ❌ More complex
        - ❌ Slower iteration
        """)

# Footer
st.divider()
st.caption("🔬 Development & Testing Interface | Built with Streamlit | Using detector.py from Flask app")
