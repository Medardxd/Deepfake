from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from detector import DeepfakeDetector

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize detector
detector = DeepfakeDetector()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Handle image upload and analysis"""

    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

    try:
        # Save file with secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Analyze image
        result = detector.analyze(filepath)

        # Clean up uploaded file (optional - comment out if you want to keep them)
        # os.remove(filepath)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'detector': 'ready'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
