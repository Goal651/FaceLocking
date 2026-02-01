# Face Recognition: ArcFace ONNX + 5-Point Alignment

A modular, CPU-first face recognition system that transforms faces into comparable vectors through a transparent pipeline.

## 🚀 Quick Start

```bash
# 1. Setup
git clone https://github.com/goal651/onnx-arcface-face-recognition.git
cd onnx-arcface-face-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Initialize
python init_project.py

# 3. Download ArcFace model
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm buffalo_l.zip *.onnx

# 4. Test components
python -m src.camera           # Camera test
python -m src.detect           # Face detection
python -m src.align            # Alignment check
```

## Core Workflow

### 1. **Enroll People**
```bash
python -m src.enroll
```
- Enter person's name
- Capture 15+ samples (SPACE for manual, A for auto)
- Press S to save

### 2. **Evaluate Threshold**
```bash
python -m src.evaluate
```
- Analyzes genuine/impostor distances
- Suggests optimal threshold

### 3. **Recognize in Real-Time**
```bash
python -m src.recognize
```
- Live webcam recognition
- Controls: +/- adjust threshold, R reload DB, Q quit

## Pipeline Architecture

**Enrollment:**
```
Camera → Detect → 5-Point Landmarks → Align (112×112) → ArcFace Embedding → Store
```

**Recognition:**
```
Camera → Detect → Landmarks → Align → Embedding → Compare → Threshold Decision
```

## Key Features

- **CPU-Only**: No GPU required
- **5-Point Alignment**: Eyes, nose, mouth corners → consistent pose
- **ArcFace ONNX**: 512-D embeddings, L2-normalized
- **Modular Design**: Each stage testable/replaceable
- **Real-Time**: Live webcam with adjustable thresholds

## 🔧 Tech Stack

- **Detection**: OpenCV Haar Cascade
- **Landmarks**: MediaPipe FaceMesh (5 points)
- **Embedding**: ArcFace ResNet-50 (ONNX format)
- **Matching**: Cosine similarity on normalized vectors

## Quick Debug

```bash
# Camera not working? Check:
python -m src.camera

# No faces detected?
python -m src.detect

# Poor recognition?
python -m src.evaluate  # Check threshold
```

## Project Structure
```
data/           # Enrolled faces + database
models/         # embedder_arcface.onnx
src/            # Modular pipeline components
init_project.py # Setup script
```

**Minimum**: Enroll 2+ people with 5+ samples each for meaningful evaluation.

---