# Quick Start Guide

## Two Versions Available

Your project now has TWO interfaces for the AI image detector:

### 1. 🚀 Streamlit (Testing & Development)
**Purpose**: Fast testing, debugging, and prototyping
**File**: `streamlit_app.py`

### 2. 🎨 Flask (Production)
**Purpose**: Polished, professional web interface
**File**: `app.py`

---

## Installation

First time setup (only need to do this once):

```bash
cd Deepfake
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Streamlit
pip install streamlit

# Or reinstall all dependencies
pip install -r requirements.txt
```

---

## Running Streamlit (Testing Version)

```bash
# Make sure venv is activated
source venv/bin/activate

# Run Streamlit
streamlit run streamlit_app.py
```

**Opens at**: http://localhost:8501

### Features:
- ✅ Single image testing
- ✅ Batch image processing
- ✅ Debug mode (see raw model outputs)
- ✅ Visual metrics and progress bars
- ✅ Fast prototyping

---

## Running Flask (Production Version)

```bash
# Make sure venv is activated
source venv/bin/activate

# Run Flask
python app.py
```

**Opens at**: http://localhost:5000

### Features:
- ✅ Custom professional UI
- ✅ Drag-and-drop upload
- ✅ Polished design
- ✅ Production-ready

---

## When to Use Which?

### Use Streamlit When:
- 🧪 Testing new features
- 🐛 Debugging detection issues
- 📦 Batch testing multiple images
- ⚡ Need to iterate quickly
- 🔬 Experimenting with model parameters

### Use Flask When:
- 🎯 Showing to others (demo/presentation)
- 📱 Need custom design
- 🚀 Ready for deployment
- 💼 Portfolio/thesis work

---

## Workflow Example

1. **Test in Streamlit** - Add new feature, verify it works
2. **Debug** - Use debug mode to check outputs
3. **Batch test** - Test on multiple images
4. **Port to Flask** - Once working, add to Flask UI
5. **Polish** - Make it look good in production

---

## Project Structure

```
Deepfake/
├── streamlit_app.py      # Streamlit testing interface
├── app.py                # Flask production interface
├── detector.py           # Shared ML logic (used by BOTH!)
├── requirements.txt      # All dependencies
├── templates/            # Flask HTML templates
├── static/              # Flask CSS/JS
├── uploads/             # Uploaded images storage
└── models/              # Downloaded ML models
```

---

## Tips

- **Both share `detector.py`**: Any changes to ML logic affect both versions
- **Only run ONE at a time**: Don't run Flask and Streamlit simultaneously (confusing)
- **Streamlit auto-reloads**: Change code = instant refresh
- **Flask needs restart**: Change code = restart server

---

## Common Commands

```bash
# Install Streamlit
pip install streamlit

# Run Streamlit
streamlit run streamlit_app.py

# Run Flask
python app.py

# Check if Streamlit installed
streamlit --version

# Stop server
CTRL + C
```

---

## Need Help?

- **Streamlit docs**: https://docs.streamlit.io
- **Flask docs**: https://flask.palletsprojects.com
- **Model info**: Check the "About" tab in Streamlit app

---

🎉 **You're all set!** Try both versions and see which workflow you prefer!
