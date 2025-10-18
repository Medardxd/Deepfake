// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const removeButton = document.getElementById('removeButton');
const analyzeButton = document.getElementById('analyzeButton');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const analyzeAnotherButton = document.getElementById('analyzeAnotherButton');
const tryAgainButton = document.getElementById('tryAgainButton');

// State
let selectedFile = null;

// Event Listeners
dropZone.addEventListener('click', (e) => {
    // Only trigger file input if clicking the drop zone itself, not the label
    if (e.target === dropZone || e.target.closest('.drop-zone') === dropZone && !e.target.closest('.file-label')) {
        fileInput.click();
    }
});
fileInput.addEventListener('change', handleFileSelect);
removeButton.addEventListener('click', resetUpload);
analyzeButton.addEventListener('click', analyzeImage);
analyzeAnotherButton.addEventListener('click', resetAll);
tryAgainButton.addEventListener('click', resetAll);

// Drag and Drop Events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File Handling Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a PNG, JPG, JPEG, GIF, or BMP image.');
        return;
    }

    // Validate file size (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 16MB. Please upload a smaller image.');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
        analyzeButton.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewSection.classList.add('hidden');
    analyzeButton.classList.add('hidden');
    dropZone.classList.remove('hidden');
}

function resetAll() {
    resetUpload();
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    loadingSpinner.classList.add('hidden');
}

// Analysis Function
async function analyzeImage() {
    if (!selectedFile) {
        showError('No file selected.');
        return;
    }

    // Hide everything and show loading
    previewSection.classList.add('hidden');
    analyzeButton.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Send to backend
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Hide loading
        loadingSpinner.classList.add('hidden');

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        loadingSpinner.classList.add('hidden');
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    }
}

// Display Results
function displayResults(data) {
    const verdictCard = document.getElementById('verdictCard');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictText = document.getElementById('verdictText');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceScore = document.getElementById('confidenceScore');
    const confidenceLevel = document.getElementById('confidenceLevel');
    const imageSize = document.getElementById('imageSize');
    const imageFormat = document.getElementById('imageFormat');
    const noteText = document.getElementById('noteText');

    // Set verdict styling
    verdictCard.classList.remove('authentic', 'deepfake');
    if (data.is_deepfake) {
        verdictCard.classList.add('deepfake');
        verdictIcon.innerHTML = `
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="15" y1="9" x2="9" y2="15"></line>
            <line x1="9" y1="9" x2="15" y2="15"></line>
        `;
    } else {
        verdictCard.classList.add('authentic');
        verdictIcon.innerHTML = `
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
        `;
    }

    // Set text content
    verdictText.textContent = data.verdict;
    confidenceText.textContent = `Analysis complete with ${data.confidence}% confidence`;
    confidenceScore.textContent = `${data.confidence}%`;
    confidenceLevel.textContent = data.confidence_label.toUpperCase();

    // Set image info
    if (data.image_info) {
        imageSize.textContent = `${data.image_info.width} x ${data.image_info.height}`;
        imageFormat.textContent = data.image_info.format || 'Unknown';
    }

    // Set note if present
    if (data.note) {
        noteText.textContent = data.note;
        document.getElementById('noteSection').classList.remove('hidden');
    } else {
        document.getElementById('noteSection').classList.add('hidden');
    }

    // Show results
    resultsSection.classList.remove('hidden');
}

// Error Handling
function showError(message) {
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;

    previewSection.classList.add('hidden');
    analyzeButton.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSpinner.classList.add('hidden');
    errorSection.classList.remove('hidden');
}

// Health Check (optional - to verify backend is running)
async function checkBackendHealth() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            console.log('Backend is ready');
        }
    } catch (error) {
        console.warn('Backend not responding:', error);
    }
}

// Initialize
checkBackendHealth();
