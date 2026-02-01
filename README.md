# Face Recognition: ArcFace ONNX + 5-Point Alignment + Face Locking

A modular, CPU-first face recognition system that transforms faces into comparable vectors through a transparent pipeline, now enhanced with **Face Locking** capabilities for behavior tracking.

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

# 5. Test Face Locking (NEW!)
python test_face_locking.py    # Verify face locking setup
```

## 🔒 NEW: Face Locking Feature

The Face Locking system recognizes a specific enrolled identity and tracks their actions over time.

### Quick Start with Face Locking

```bash
# 1. Enroll yourself first
python -m src.enroll

# 2. Run Face Locking System
python -m src.face_locking
```

### Face Locking Controls
- **L** - Toggle lock on/off for target identity
- **S** - Save action history to file
- **R** - Reload face database
- **Q** - Quit system

### What Face Locking Does

1. **Target Selection**: Focuses on one enrolled identity (configurable)
2. **Face Locking**: When target appears with high confidence, locks onto them
3. **Stable Tracking**: Maintains lock even during brief recognition failures
4. **Action Detection**: Detects and records:
   - Face movement (left/right/up/down)
   - Eye blinks with timing
   - Smiles and expressions
5. **History Recording**: Saves timestamped action log to files

### Action History Files

Files are automatically saved as: `{identity}_history_{timestamp}.txt`

Example: `wilson_history_20260201143022.txt`

Each record includes:
- Precise timestamp
- Action type (movement, blink, smile)
- Detailed description
- Quantitative values (speed, ratios, etc.)

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

### 3. **Standard Recognition**
```bash
python -m src.recognize
```
- Live webcam recognition
- Controls: +/- adjust threshold, R reload DB, Q quit

### 4. **Face Locking (NEW)**
```bash
python -m src.face_locking
```
- Locks onto specific enrolled identity
- Tracks actions and behaviors
- Records detailed action history

## Pipeline Architecture

**Enrollment:**
```
Camera → Detect → 5-Point Landmarks → Align (112×112) → ArcFace Embedding → Store
```

**Recognition:**
```
Camera → Detect → Landmarks → Align → Embedding → Compare → Threshold Decision
```

**Face Locking (NEW):**
```
Camera → Detect → Recognize → Lock Target → Track Actions → Record History
                                    ↓
                            [Movement, Blinks, Smiles]
```

## Key Features

- **CPU-Only**: No GPU required
- **5-Point Alignment**: Eyes, nose, mouth corners → consistent pose
- **ArcFace ONNX**: 512-D embeddings, L2-normalized
- **Modular Design**: Each stage testable/replaceable
- **Real-Time**: Live webcam with adjustable thresholds
- **🔒 Face Locking**: Target-specific behavior tracking
- **📊 Action Detection**: Movement, blinks, expressions
- **📝 History Recording**: Timestamped action logs

## 🔧 Tech Stack

- **Detection**: OpenCV Haar Cascade
- **Landmarks**: MediaPipe FaceMesh (5 points)
- **Embedding**: ArcFace ResNet-50 (ONNX format)
- **Matching**: Cosine similarity on normalized vectors
- **Action Detection**: Advanced algorithms for movement, blinks, smiles
- **Tracking**: Stable face locking with confidence smoothing

## Face Locking Technical Details

### Action Detection Algorithms

1. **Movement Detection**:
   - Position smoothing over multiple frames
   - Velocity calculation with direction classification
   - Threshold-based significant movement detection

2. **Blink Detection**:
   - Eye Aspect Ratio (EAR) calculation
   - Consecutive frame validation
   - False positive filtering

3. **Smile Detection**:
   - Mouth geometry analysis
   - Width-to-height ratio calculation
   - Temporal smoothing for stability

### Tracking Stability

- **Position-based tracking**: Maintains lock using spatial continuity
- **Confidence history**: Smooths recognition confidence over time
- **Timeout handling**: Releases lock after extended absence
- **Re-acquisition**: Automatically re-locks when target reappears

## Quick Debug

```bash
# Camera not working? Check:
python -m src.camera

# No faces detected?
python -m src.detect

# Poor recognition?
python -m src.evaluate  # Check threshold

# Face locking issues?
python test_face_locking.py  # Run test suite
```

## Project Structure
```
data/           # Enrolled faces + database + action histories
├── db/         # Face database (embeddings)
├── enroll/     # Enrollment photos
└── *_history_*.txt  # Action history files (NEW)

models/         # embedder_arcface.onnx
src/            # Modular pipeline components
├── face_locking.py      # Face locking system (NEW)
├── action_detection.py  # Action detection algorithms (NEW)
├── recognize.py         # Multi-face recognition
├── enroll.py           # Enrollment system
└── ...

init_project.py         # Setup script
test_face_locking.py    # Face locking test suite (NEW)
```

## Configuration

### Target Identity Selection

Edit `src/face_locking.py` to change target:

```python
# Change target identity
system = FaceLockingSystem(target_identity="YourName")
```

### Action Detection Tuning

Modify thresholds in `src/action_detection.py`:

```python
# Movement sensitivity
self.movement_threshold = 25  # pixels

# Blink detection
self.eye_aspect_ratio_threshold = 0.25

# Smile detection  
self.smile_ratio_threshold = 1.8
```

**Minimum**: Enroll 2+ people with 5+ samples each for meaningful evaluation.

---

## Assignment Compliance

This implementation fulfills all Term-02 Week-04 requirements:

✅ **Manual Face Selection**: Configurable target identity  
✅ **Face Locking**: Locks onto enrolled identity with visual feedback  
✅ **Stable Tracking**: Maintains lock across frames with timeout handling  
✅ **Action Detection**: Movement (left/right), blinks, smiles  
✅ **Action History**: Timestamped files with format `{face}_history_{timestamp}.txt`  
✅ **CPU-Only**: No GPU requirements  
✅ **Existing Pipeline**: Built on top of working recognition system  

### Action History File Format

```
Face Locking Action History
Target Identity: Wilson
Session Start: 2026-02-01 14:30:22.123
Session End: 2026-02-01 14:35:45.678
Total Actions: 15
--------------------------------------------------

2026-02-01 14:30:22.123 | lock_initiated | Locked onto Wilson at position (320, 240)
2026-02-01 14:30:25.456 | face_moved_right | Face moved 45.2px in right direction | Value: 23.1
2026-02-01 14:30:28.789 | eye_blink | Blink detected (EAR: 0.234, count: 1) | Value: 0.234
2026-02-01 14:30:32.012 | smile | Smile detected (mouth ratio: 2.15) | Value: 2.150
...
```