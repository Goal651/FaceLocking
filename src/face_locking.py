# src/face_locking.py
"""
Face Locking Feature Implementation

This module extends the face recognition system with face locking capabilities:
- Manual face selection for a specific enrolled identity
- Stable face tracking across frames
- Action detection (movement, blinks, smiles)
- Action history recording to timestamped files in ./history/
- Display all detected faces with status (LOCKED/UNLOCKED/UNKNOWN)
- Enhanced smile detection with adjustable thresholds
- Real-time smile scoring and visual feedback
- Improved face detection to avoid false positives (mouths, etc.)
- Scalable UI with noise reduction

Usage:
    python -m src.face_locking

Controls:
    q: quit
    r: reload database
    l: toggle lock on/off for target identity
    +/-: adjust smile detection threshold
    F1/F2: adjust face detection sensitivity (reduce false positives)
    m: toggle mirror mode (natural camera view)
    M: toggle landmarks display
    C: toggle confidence display
    d: toggle detailed UI information
    [/]: adjust window scaling
    s: save current action history to ./history/

Features:
    - Green border: Locked target face
    - Cyan border: Target identity (unlocked)
    - Orange border: Known person (not target)
    - Red border: Unknown person
    - Gray border: Rejected detection (debug mode)
    - Real-time smile detection with score display
    - Enhanced mouth geometry analysis for better smile detection
    - Face validation to filter out mouth detections
    - Adjustable face detection parameters
    - Clean, scalable UI with reduced visual noise
    - Mirror mode for natural camera experience (default ON)
    - Proper movement direction detection in mirror mode

Admin Tools:
    Run 'python admin.py' to:
    - List all enrolled users
    - Delete enrolled users
    - View database statistics
    - Backup database
    - Clean up corrupted entries
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import cv2
import numpy as np

# Import existing modules
from .recognize import (
    HaarFaceMesh5pt, ArcFaceEmbedderONNX, FaceDBMatcher, 
    FaceDet, MatchResult, load_db_npz, cosine_distance
)
from .haar_5pt import align_face_5pt
from .action_detection import AdvancedActionDetector, ActionClassifier


@dataclass
class ActionRecord:
    """Single action record with timestamp"""
    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None


@dataclass
class FaceTracker:
    """Tracks a locked face across frames"""
    identity: str
    last_position: Tuple[int, int]  # center (x, y)
    last_seen_frame: int
    confidence_history: deque  # recent recognition confidences
    position_history: deque    # recent positions for movement detection
    blink_state: str          # "open", "closed", "unknown"
    blink_counter: int
    smile_state: bool
    lock_start_time: float
    
    def __post_init__(self):
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, 'position_history') or self.position_history is None:
            self.position_history = deque(maxlen=5)


class ActionDetector:
    """Detects face actions: movement, blinks, smiles - DEPRECATED"""
    
    def __init__(self):
        print("Warning: Using deprecated ActionDetector. Use AdvancedActionDetector instead.")
        # Movement thresholds
        self.movement_threshold = 30  # pixels
        
        # Blink detection (simple eye aspect ratio based)
        self.blink_threshold = 0.25
        self.blink_frames = 3  # consecutive frames to confirm blink
        
        # Smile detection (mouth aspect ratio based)
        self.smile_threshold = 0.6
        
    def detect_movement(self, current_pos: Tuple[int, int], 
                       position_history: deque) -> Optional[str]:
        """Detect left/right movement based on position history"""
        if len(position_history) < 3:
            return None
            
        # Get recent positions
        positions = list(position_history)
        positions.append(current_pos)
        
        # Calculate movement trend
        x_positions = [pos[0] for pos in positions[-4:]]
        if len(x_positions) < 4:
            return None
            
        # Check for consistent movement
        x_diff = x_positions[-1] - x_positions[0]
        
        if abs(x_diff) > self.movement_threshold:
            if x_diff > 0:
                return "face_moved_right"
            else:
                return "face_moved_left"
        
        return None
    
    def detect_blink(self, landmarks: np.ndarray) -> Tuple[str, bool]:
        """
        Detect eye blink using eye aspect ratio
        landmarks: (5, 2) array with [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Simple blink detection based on eye positions
            # In a real implementation, you'd use more sophisticated eye landmarks
            # For now, we'll use a simplified approach
            
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Simulate blink detection (in practice, you'd analyze eye opening)
            # This is a placeholder - real blink detection would need more eye landmarks
            blink_detected = eye_distance < 50  # simplified threshold
            
            if blink_detected:
                return "closed", True
            else:
                return "open", False
                
        except Exception:
            return "unknown", False
    
    def detect_smile(self, landmarks: np.ndarray) -> bool:
        """
        Detect smile using mouth landmarks
        landmarks: (5, 2) array with [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        try:
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            nose = landmarks[2]
            
            # Calculate mouth width vs height ratio
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            mouth_center = (left_mouth + right_mouth) / 2
            mouth_to_nose = np.linalg.norm(mouth_center - nose)
            
            # Simple smile detection based on mouth width relative to face
            smile_ratio = mouth_width / max(mouth_to_nose, 1)
            
            return smile_ratio > self.smile_threshold
            
        except Exception:
            return False


class FaceLockingSystem:
    """Main face locking system with improved UI and better face detection"""
    
    def __init__(self, target_identity: str = "Wilson", 
                 db_path: str = "data/db/face_db.npz",
                 model_path: str = "models/embedder_arcface.onnx",
                 window_scale: float = 1.0,
                 mirror_mode: bool = True):
        
        self.target_identity = target_identity
        self.db_path = Path(db_path)
        self.window_scale = window_scale
        self.mirror_mode = mirror_mode  # Mirror the camera view
        
        # UI settings
        self.show_detailed_info = True
        self.show_landmarks = True
        self.show_confidence = True
        self.ui_alpha = 0.8  # transparency for UI elements
        
        # Face detection settings to avoid mouth detection
        self.min_face_size = (80, 80)  # Increased minimum face size
        self.max_face_size = (400, 400)  # Maximum face size to avoid false positives
        self.face_aspect_ratio_range = (0.7, 1.4)  # Valid face aspect ratios
        
        # Initialize components with improved settings
        self.detector = HaarFaceMesh5pt(min_size=self.min_face_size, debug=False)
        self.embedder = ArcFaceEmbedderONNX(model_path=model_path, debug=False)
        
        # Load database and matcher
        db = load_db_npz(self.db_path)
        self.matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
        
        # Face tracking
        self.tracker: Optional[FaceTracker] = None
        self.is_locked = False
        self.frame_count = 0
        self.lock_timeout = 30  # frames to wait before releasing lock
        
        # Action detection
        self.action_detector = AdvancedActionDetector()
        self.action_classifier = ActionClassifier()
        self.action_history: List[ActionRecord] = []
        
        # UI state
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.1  # update UI every 100ms to reduce flicker
        
        # Face validation cache to reduce false positives
        self.face_validation_cache = {}
        self.cache_timeout = 2.0  # seconds
        
        # Verify target identity exists
        if target_identity not in db:
            available = list(db.keys())
            raise ValueError(f"Target identity '{target_identity}' not found in database. "
                           f"Available: {available}")
        
        print(f"Face Locking System initialized for: {target_identity}")
        print(f"Database contains {len(db)} identities: {list(db.keys())}")
        print(f"Window scale: {window_scale}x")
        print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
        print(f"Face detection: min_size={self.min_face_size}, max_size={self.max_face_size}")
    
    def _validate_face_detection(self, face: FaceDet, frame: np.ndarray) -> bool:
        """Validate face detection to filter out false positives like mouths"""
        
        # Check face size constraints
        face_width = face.x2 - face.x1
        face_height = face.y2 - face.y1
        
        # Size validation
        if face_width < self.min_face_size[0] or face_height < self.min_face_size[1]:
            return False
        
        if face_width > self.max_face_size[0] or face_height > self.max_face_size[1]:
            return False
        
        # Aspect ratio validation (faces should be roughly rectangular)
        aspect_ratio = face_width / face_height
        if not (self.face_aspect_ratio_range[0] <= aspect_ratio <= self.face_aspect_ratio_range[1]):
            return False
        
        # Landmark validation - check if landmarks make sense for a face
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = face.kps
            
            # Check if we have the expected number of landmarks
            if len(landmarks) != 5:
                return False
            
            # Validate landmark positions relative to face box
            for lm in landmarks:
                x, y = lm
                # Landmarks should be within the face bounding box (with some tolerance)
                if not (face.x1 - 10 <= x <= face.x2 + 10 and face.y1 - 10 <= y <= face.y2 + 10):
                    return False
            
            # Check eye-mouth geometry (basic face structure validation)
            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
            
            # Eyes should be horizontally aligned (roughly)
            eye_y_diff = abs(left_eye[1] - right_eye[1])
            if eye_y_diff > face_height * 0.15:  # Eyes too misaligned
                return False
            
            # Mouth should be below eyes
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            if mouth_center_y <= eye_center_y:  # Mouth above eyes - invalid
                return False
            
            # Nose should be between eyes and mouth vertically
            if not (eye_center_y < nose[1] < mouth_center_y):
                return False
            
            # Check if mouth is too large relative to face (might be detecting mouth as face)
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            if mouth_width > face_width * 0.8:  # Mouth too wide for face
                return False
        
        # Position validation - avoid detections at extreme edges
        frame_h, frame_w = frame.shape[:2]
        face_center_x = (face.x1 + face.x2) / 2
        face_center_y = (face.y1 + face.y2) / 2
        
        # Face should not be too close to edges
        edge_margin = 20
        if (face_center_x < edge_margin or face_center_x > frame_w - edge_margin or
            face_center_y < edge_margin or face_center_y > frame_h - edge_margin):
            return False
        
        return True
    
    def _get_face_center(self, face: FaceDet) -> Tuple[int, int]:
        """Get center point of face bounding box"""
        center_x = (face.x1 + face.x2) // 2
        center_y = (face.y1 + face.y2) // 2
        return (center_x, center_y)
    
    def _is_same_face(self, face: FaceDet, tracker: FaceTracker) -> bool:
        """Check if detected face matches tracked face based on position"""
        current_center = self._get_face_center(face)
        last_center = tracker.last_position
        
        distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                          (current_center[1] - last_center[1])**2)
        
        # Allow some movement but not too much
        return distance < 100  # pixels
    
    def _record_action(self, action_type: str, description: str, value: Optional[float] = None):
        """Record an action to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        action = ActionRecord(
            timestamp=timestamp,
            action_type=action_type,
            description=description,
            value=value
        )
        self.action_history.append(action)
        print(f"[ACTION] {action.timestamp}: {action.description}")
    
    def _save_action_history(self) -> str:
        """Save action history to file in history folder"""
        if not self.action_history:
            print("No actions to save")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.target_identity.lower()}_history_{timestamp}.txt"
        filepath = Path("history") / filename
        
        # Ensure history directory exists
        filepath.parent.mkdir(exist_ok=True)
        
        # Write history
        with open(filepath, 'w') as f:
            f.write(f"Face Locking Action History\n")
            f.write(f"Target Identity: {self.target_identity}\n")
            f.write(f"Session Start: {self.action_history[0].timestamp if self.action_history else 'N/A'}\n")
            f.write(f"Session End: {self.action_history[-1].timestamp if self.action_history else 'N/A'}\n")
            f.write(f"Total Actions: {len(self.action_history)}\n")
            f.write("-" * 50 + "\n\n")
            
            for action in self.action_history:
                line = f"{action.timestamp} | {action.action_type} | {action.description}"
                if action.value is not None:
                    line += f" | Value: {action.value:.3f}"
                f.write(line + "\n")
        
        print(f"Action history saved to: {filepath}")
        return str(filepath)
    
    def _create_ui_overlay(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a semi-transparent overlay for UI elements"""
        h, w = frame_shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create semi-transparent background for UI areas
        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Bottom controls
        cv2.rectangle(overlay, (0, h - 160), (w, h), (0, 0, 0), -1)
        
        # Side legend (if needed)
        if self.show_detailed_info:
            cv2.rectangle(overlay, (0, 120), (300, h - 160), (0, 0, 0), -1)
        
        return overlay
    
    def _draw_clean_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                        color: Tuple[int, int, int], scale: float = 0.6, thickness: int = 1):
        """Draw text with background for better readability"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(img, (pos[0] - 2, pos[1] - text_h - 2), 
                     (pos[0] + text_w + 2, pos[1] + baseline + 2), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, pos, font, scale, color, thickness)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for face locking with improved UI and face validation"""
        self.frame_count += 1
        
        # Mirror the frame if mirror mode is enabled (for natural camera view)
        if self.mirror_mode:
            frame = cv2.flip(frame, 1)  # Horizontal flip
        
        # Scale frame if needed
        if self.window_scale != 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * self.window_scale), int(h * self.window_scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        vis = frame.copy()
        
        # Create UI overlay
        current_time = time.time()
        if current_time - self.last_ui_update > self.ui_update_interval:
            self.last_ui_update = current_time
        
        # Detect faces with validation
        detected_faces = self.detector.detect(frame, max_faces=5)
        
        # Filter out invalid detections (like mouths)
        valid_faces = []
        for face in detected_faces:
            if self._validate_face_detection(face, frame):
                valid_faces.append(face)
            else:
                # Draw rejected detection in debug mode
                if self.show_detailed_info:
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (128, 128, 128), 1)
                    self._draw_clean_text(vis, "REJECTED", (face.x1, face.y1 - 5), (128, 128, 128), 0.3, 1)
        
        # If we have a lock, try to maintain it
        if self.is_locked and self.tracker:
            self._update_locked_face(frame, valid_faces, vis)
        else:
            # Look for target identity to lock onto
            self._search_for_target(frame, valid_faces, vis)
        
        # Draw clean UI
        self._draw_clean_ui(vis)
        
        return vis
    
    def _update_locked_face(self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray):
        """Update tracking for locked face and display all faces"""
        target_face = None
        
        # Process all faces in the frame
        for face in faces:
            # Recognize each face
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            match_result = self.matcher.match(emb)
            
            # Check if this is our locked target
            is_locked_target = (self._is_same_face(face, self.tracker) and 
                              match_result.accepted and 
                              match_result.name == self.target_identity)
            
            if is_locked_target:
                target_face = face
                self.tracker.confidence_history.append(match_result.similarity)
                # Draw locked face (will be done separately)
            else:
                # Draw unlocked faces
                self._draw_unlocked_face(vis, face, match_result)
        
        if target_face:
            # Update tracker
            current_center = self._get_face_center(target_face)
            self.tracker.last_position = current_center
            self.tracker.last_seen_frame = self.frame_count
            self.tracker.position_history.append(current_center)
            
            # Detect actions
            self._detect_actions(target_face)
            
            # Draw locked face
            self._draw_locked_face(vis, target_face)
            
        else:
            # Face lost - check timeout
            frames_since_seen = self.frame_count - self.tracker.last_seen_frame
            if frames_since_seen > self.lock_timeout:
                self._release_lock()
                self._record_action("lock_lost", f"Face disappeared for {frames_since_seen} frames")
    
    def _draw_unlocked_face(self, vis: np.ndarray, face: FaceDet, match_result: MatchResult):
        """Draw unlocked faces with clean, minimal UI"""
        # Determine face status and colors
        if match_result.accepted:
            if match_result.name == self.target_identity:
                color = (0, 255, 255)  # Cyan for target but not locked
                status = "TARGET"
            else:
                color = (255, 165, 0)  # Orange for known but not target
                status = match_result.name.upper()
        else:
            color = (0, 0, 255)  # Red for unknown
            status = "UNKNOWN"
        
        # Draw face rectangle (clean, thin border)
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        
        # Draw minimal status text with background
        if self.show_detailed_info:
            self._draw_clean_text(vis, status, (face.x1, face.y1 - 10), color, 0.5, 1)
            
            # Show confidence only if detailed info is enabled
            if self.show_confidence:
                conf_text = f"{match_result.similarity:.2f}"
                self._draw_clean_text(vis, conf_text, (face.x2 - 50, face.y1 - 10), color, 0.4, 1)
        
        # Draw landmarks only if enabled and for known faces
        if self.show_landmarks and match_result.accepted:
            for x, y in face.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 1, color, -1)
        
        # Minimal smile detection display
        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(face.kps, time.time())
        if smile_detected:
            # Simple smile indicator - just a small icon
            smile_x, smile_y = face.x2 - 25, face.y1 + 15
            cv2.circle(vis, (smile_x, smile_y), 8, (0, 255, 255), 1)
            cv2.circle(vis, (smile_x - 3, smile_y - 2), 1, (0, 255, 255), -1)  # left eye
            cv2.circle(vis, (smile_x + 3, smile_y - 2), 1, (0, 255, 255), -1)  # right eye
            cv2.ellipse(vis, (smile_x, smile_y + 2), (4, 2), 0, 0, 180, (0, 255, 255), 1)  # smile
    
    def _search_for_target(self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray):
        """Search for target identity to lock onto and display all faces with clean UI"""
        for face in faces:
            # Recognize face
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            match_result = self.matcher.match(emb)
            
            # Determine face status and colors
            is_target = (match_result.accepted and match_result.name == self.target_identity)
            
            if is_target:
                color = (0, 255, 0)  # Green for target
                status = "TARGET"
            elif match_result.accepted:
                color = (255, 165, 0)  # Orange for known but not target
                status = match_result.name.upper()
            else:
                color = (0, 0, 255)  # Red for unknown
                status = "UNKNOWN"
            
            # Draw face rectangle (clean)
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
            
            # Draw status with clean text
            if self.show_detailed_info:
                self._draw_clean_text(vis, status, (face.x1, face.y1 - 10), color, 0.5, 1)
                
                # Show confidence
                if self.show_confidence:
                    self._draw_clean_text(vis, f"{match_result.similarity:.2f}", 
                                         (face.x2 - 50, face.y1 - 10), color, 0.4, 1)
            
            # Draw landmarks for known faces only
            if self.show_landmarks and match_result.accepted:
                for x, y in face.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 1, color, -1)
            
            # Clean smile detection display
            smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(face.kps, time.time())
            if smile_detected:
                # Simple smile indicator
                smile_x, smile_y = face.x2 - 20, face.y1 + 15
                cv2.circle(vis, (smile_x, smile_y), 6, (0, 255, 255), 1)
                cv2.circle(vis, (smile_x - 2, smile_y - 1), 1, (0, 255, 255), -1)  # left eye
                cv2.circle(vis, (smile_x + 2, smile_y - 1), 1, (0, 255, 255), -1)  # right eye
                cv2.ellipse(vis, (smile_x, smile_y + 1), (3, 2), 0, 0, 180, (0, 255, 255), 1)  # smile
            
            # Check if this is our target and we should lock
            if (is_target and match_result.similarity > 0.7):  # High confidence threshold for locking
                self._initiate_lock(face, match_result.similarity)
    
    def _initiate_lock(self, face: FaceDet, confidence: float):
        """Initiate face lock on target"""
        center = self._get_face_center(face)
        
        self.tracker = FaceTracker(
            identity=self.target_identity,
            last_position=center,
            last_seen_frame=self.frame_count,
            confidence_history=deque([confidence], maxlen=10),
            position_history=deque([center], maxlen=5),
            blink_state="unknown",
            blink_counter=0,
            smile_state=False,
            lock_start_time=time.time()
        )
        
        self.is_locked = True
        self._record_action("lock_initiated", f"Locked onto {self.target_identity} at position {center}")
        print(f"🔒 LOCKED onto {self.target_identity}")
    
    def _release_lock(self):
        """Release face lock"""
        if self.tracker:
            lock_duration = time.time() - self.tracker.lock_start_time
            self._record_action("lock_released", f"Lock held for {lock_duration:.1f} seconds")
        
        self.tracker = None
        self.is_locked = False
        print("🔓 Lock RELEASED")
    
    def _detect_actions(self, face: FaceDet):
        """Detect actions on locked face using advanced detection"""
        if not self.tracker:
            return
        
        current_time = time.time()
        current_center = self._get_face_center(face)
        
        # Movement detection
        movement_info = self.action_detector.detect_movement_advanced(current_center, current_time)
        if movement_info and self.action_classifier.should_record_action(movement_info["direction"], current_time):
            # Adjust movement direction for mirror mode
            if self.mirror_mode:
                direction = movement_info["direction"]
                if "left" in direction:
                    direction = direction.replace("left", "right")
                elif "right" in direction:
                    direction = direction.replace("right", "left")
                movement_info["direction"] = direction
            
            action_type = self.action_classifier.classify_movement(movement_info)
            description = f"Face moved {movement_info['distance']:.1f}px in {movement_info['direction'].split('_')[2]} direction"
            self._record_action(action_type, description, movement_info["speed"])
        
        # Blink detection
        blink_detected, eye_metrics = self.action_detector.detect_blink_advanced(face.kps, current_time)
        if blink_detected and self.action_classifier.should_record_action("eye_blink", current_time):
            self.tracker.blink_counter += 1
            description = f"Blink detected (EAR: {eye_metrics.avg_eye_ratio:.3f}, count: {self.tracker.blink_counter})"
            self._record_action("eye_blink", description, eye_metrics.avg_eye_ratio)
        
        # Smile detection
        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(face.kps, current_time)
        if smile_detected != self.tracker.smile_state:
            if smile_detected and self.action_classifier.should_record_action("smile", current_time):
                description = f"Smile detected (mouth ratio: {mouth_metrics.mouth_ratio:.2f})"
                self._record_action("smile", description, mouth_metrics.mouth_ratio)
            elif not smile_detected and self.action_classifier.should_record_action("smile_end", current_time):
                self._record_action("smile_end", "Smile ended")
            self.tracker.smile_state = smile_detected
    
    def _draw_locked_face(self, vis: np.ndarray, face: FaceDet):
        """Draw locked face with clean, prominent highlighting"""
        # Thick green border for locked face
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 3)
        
        # Clean lock icon
        lock_x, lock_y = face.x1 + 5, face.y1 + 5
        cv2.rectangle(vis, (lock_x, lock_y), (lock_x + 15, lock_y + 12), (0, 255, 0), 2)
        cv2.circle(vis, (lock_x + 7, lock_y + 4), 4, (0, 255, 0), 2)
        
        # Clean status text
        self._draw_clean_text(vis, f"LOCKED: {self.target_identity}", 
                             (face.x1, face.y1 - 15), (0, 255, 0), 0.7, 2)
        
        # Confidence (only if detailed info enabled)
        if self.show_confidence and self.tracker and self.tracker.confidence_history:
            avg_conf = np.mean(list(self.tracker.confidence_history))
            self._draw_clean_text(vis, f"{avg_conf:.2f}", 
                                 (face.x2 - 50, face.y1 - 15), (0, 255, 0), 0.5, 1)
        
        # Enhanced smile detection with clean display
        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(face.kps, time.time())
        
        if smile_detected:
            # Prominent smile indicator
            smile_x, smile_y = face.x1 + 30, face.y1 + 25
            cv2.circle(vis, (smile_x, smile_y), 12, (0, 255, 255), 2)
            cv2.circle(vis, (smile_x - 4, smile_y - 3), 2, (0, 255, 255), -1)  # left eye
            cv2.circle(vis, (smile_x + 4, smile_y - 3), 2, (0, 255, 255), -1)  # right eye
            cv2.ellipse(vis, (smile_x, smile_y + 2), (6, 4), 0, 0, 180, (0, 255, 255), 2)  # smile
            
            # Smile score
            if self.show_detailed_info:
                self._draw_clean_text(vis, f"SMILE {mouth_metrics.mouth_ratio:.1f}", 
                                     (face.x1, face.y2 + 15), (0, 255, 255), 0.5, 1)
        elif self.show_detailed_info:
            # Show current smile score when not smiling
            self._draw_clean_text(vis, f"Score: {mouth_metrics.mouth_ratio:.1f}", 
                                 (face.x1, face.y2 + 15), (150, 150, 150), 0.4, 1)
        
        # Show blink counter (minimal)
        if self.show_detailed_info and self.tracker:
            self._draw_clean_text(vis, f"Blinks: {self.tracker.blink_counter}", 
                                 (face.x2 - 80, face.y2 + 15), (0, 255, 0), 0.4, 1)
        
        # Draw landmarks (only key points for locked face)
        if self.show_landmarks:
            # Highlight mouth landmarks for smile detection
            if len(face.kps) >= 5:
                left_mouth = face.kps[3].astype(int)
                right_mouth = face.kps[4].astype(int)
                cv2.circle(vis, tuple(left_mouth), 3, (0, 255, 255), -1)  # left mouth corner
                cv2.circle(vis, tuple(right_mouth), 3, (0, 255, 255), -1)  # right mouth corner
                cv2.line(vis, tuple(left_mouth), tuple(right_mouth), (0, 255, 255), 1)  # mouth line
                
                # Other landmarks (smaller)
                for i, (x, y) in enumerate(face.kps.astype(int)):
                    if i not in [3, 4]:  # skip mouth corners (already drawn)
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    def _draw_clean_ui(self, vis: np.ndarray):
        """Draw clean, minimal user interface"""
        h, w = vis.shape[:2]
        
        # Create semi-transparent overlay for UI
        overlay = vis.copy()
        
        # Top status bar with background
        status_bg = np.zeros((50, w, 3), dtype=np.uint8)
        status_bg[:] = (0, 0, 0)
        
        # Status information
        status = "🔒 LOCKED" if self.is_locked else "🔍 SEARCHING"
        status_color = (0, 255, 0) if self.is_locked else (0, 255, 255)
        
        # Main status
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        mirror_indicator = "🪞" if self.mirror_mode else "📷"
        self._draw_clean_text(vis, f"Face Locking: {status} | Target: {self.target_identity} {mirror_indicator}", 
                             (10, 25), status_color, 0.7, 2)
        
        # Action count (minimal)
        if len(self.action_history) > 0:
            self._draw_clean_text(vis, f"Actions: {len(self.action_history)}", 
                                 (w - 150, 25), (255, 255, 255), 0.5, 1)
        
        # Bottom controls (only show when needed)
        if self.show_detailed_info:
            # Controls background
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            
            # Compact controls
            controls = "L:Lock | S:Save | R:Reload | +/-:Smile | F1/F2:Detection | m:Mirror | M:Landmarks | D:Details | Q:Quit"
            self._draw_clean_text(vis, controls, (10, h - 50), (200, 200, 200), 0.45, 1)
            
            # Current settings
            mirror_text = "Mirror" if self.mirror_mode else "Normal"
            settings = f"Smile: {self.action_detector.smile_ratio_threshold:.1f} | Scale: {self.window_scale:.1f}x | Face: {self.min_face_size[0]}px | {mirror_text}"
            self._draw_clean_text(vis, settings, (10, h - 25), (150, 150, 150), 0.4, 1)
        else:
            # Minimal controls
            cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            self._draw_clean_text(vis, "D:Show Details | Q:Quit", (10, h - 10), (200, 200, 200), 0.4, 1)
        
        # Side legend (compact, only when detailed)
        if self.show_detailed_info and not self.is_locked:
            legend_x = 10
            legend_y = 70
            
            # Compact legend
            legend_items = [
                ("●", (0, 255, 0), "Locked"),
                ("●", (0, 255, 255), "Target"), 
                ("●", (255, 165, 0), "Known"),
                ("●", (0, 0, 255), "Unknown")
            ]
            
            for i, (symbol, color, label) in enumerate(legend_items):
                y_pos = legend_y + (i * 20)
                self._draw_clean_text(vis, f"{symbol} {label}", (legend_x, y_pos), color, 0.4, 1)
    
    def toggle_ui_details(self):
        """Toggle detailed UI information"""
        self.show_detailed_info = not self.show_detailed_info
        print(f"UI Details: {'ON' if self.show_detailed_info else 'OFF'}")
    
    def toggle_landmarks(self):
        """Toggle landmark display"""
        self.show_landmarks = not self.show_landmarks
        print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
    
    def toggle_confidence(self):
        """Toggle confidence display"""
        self.show_confidence = not self.show_confidence
        print(f"Confidence: {'ON' if self.show_confidence else 'OFF'}")
    
    def adjust_window_scale(self, increase: bool):
        """Adjust window scaling"""
        if increase:
            self.window_scale = min(2.0, self.window_scale + 0.1)
        else:
            self.window_scale = max(0.5, self.window_scale - 0.1)
        print(f"Window scale: {self.window_scale:.1f}x")
    
    def toggle_mirror_mode(self):
        """Toggle mirror mode for natural camera view"""
        self.mirror_mode = not self.mirror_mode
        print(f"Mirror mode: {'ON' if self.mirror_mode else 'OFF'}")
        self._record_action("mirror_toggle", f"Mirror mode {'enabled' if self.mirror_mode else 'disabled'}")
    
    def adjust_face_detection_sensitivity(self, increase: bool):
        """Adjust face detection sensitivity to reduce false positives"""
        if increase:
            # More sensitive - smaller minimum size, wider aspect ratio range
            self.min_face_size = (max(60, self.min_face_size[0] - 10), max(60, self.min_face_size[1] - 10))
            self.face_aspect_ratio_range = (max(0.5, self.face_aspect_ratio_range[0] - 0.1), 
                                          min(2.0, self.face_aspect_ratio_range[1] + 0.1))
        else:
            # Less sensitive - larger minimum size, narrower aspect ratio range
            self.min_face_size = (min(120, self.min_face_size[0] + 10), min(120, self.min_face_size[1] + 10))
            self.face_aspect_ratio_range = (min(0.9, self.face_aspect_ratio_range[0] + 0.1), 
                                          max(1.2, self.face_aspect_ratio_range[1] - 0.1))
        
        print(f"Face detection - Min size: {self.min_face_size}, Aspect ratio: {self.face_aspect_ratio_range}")
        self._record_action("detection_adjust", f"Face detection sensitivity adjusted")
    
    def adjust_smile_threshold(self, increase: bool):
        """Adjust smile detection threshold"""
        if increase:
            self.action_detector.smile_ratio_threshold += 0.1
        else:
            self.action_detector.smile_ratio_threshold = max(0.5, self.action_detector.smile_ratio_threshold - 0.1)
        
        print(f"Smile threshold adjusted to: {self.action_detector.smile_ratio_threshold:.1f}")
        self._record_action("threshold_adjust", f"Smile threshold set to {self.action_detector.smile_ratio_threshold:.1f}")
    
    def toggle_lock(self):
        """Manually toggle lock state"""
        if self.is_locked:
            self._release_lock()
        else:
            print(f"Manual lock toggle - searching for {self.target_identity}")
    
    def reload_database(self):
        """Reload face database"""
        db = load_db_npz(self.db_path)
        self.matcher.reload_from(self.db_path)
        print(f"Database reloaded: {len(db)} identities")


def main():
    """Main face locking demo with improved UI"""
    
    # Initialize system with scaling
    try:
        system = FaceLockingSystem(target_identity="Wilson", window_scale=1.0, mirror_mode=True)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("FACE LOCKING SYSTEM - Enhanced UI & Face Detection")
    print("="*60)
    print(f"Target Identity: {system.target_identity}")
    print(f"Window Scale: {system.window_scale}x")
    print(f"Mirror Mode: {'ON' if system.mirror_mode else 'OFF'}")
    print(f"Face Detection: Min size {system.min_face_size}")
    print("\nControls:")
    print("  L - Toggle lock on/off")
    print("  S - Save action history to ./history/")
    print("  R - Reload database")
    print("  +/- - Adjust smile threshold")
    print("  F1/F2 - Adjust face detection sensitivity")
    print("  m - Toggle mirror mode (natural camera view)")
    print("  M - Toggle landmarks display")
    print("  C - Toggle confidence display")
    print("  D - Toggle detailed UI")
    print("  [ / ] - Adjust window scale")
    print("  Q - Quit")
    print("\nAdmin Tools:")
    print("  Run 'python admin.py' to manage enrolled users")
    print("="*60)
    
    # Create resizable window
    cv2.namedWindow("Face Locking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Locking System", 1280, 720)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            vis = system.process_frame(frame)
            
            # Display
            cv2.imshow("Face Locking System", vis)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('l') or key == ord('L'):
                system.toggle_lock()
            elif key == ord('s') or key == ord('S'):
                filepath = system._save_action_history()
                if filepath:
                    print(f"History saved to: {filepath}")
            elif key == ord('r') or key == ord('R'):
                system.reload_database()
            elif key == ord('+') or key == ord('='):
                system.adjust_smile_threshold(increase=True)
            elif key == ord('-') or key == ord('_'):
                system.adjust_smile_threshold(increase=False)
            elif key == ord('d') or key == ord('D'):
                system.toggle_ui_details()
            elif key == ord('['):
                system.adjust_window_scale(increase=False)
            elif key == ord(']'):
                system.adjust_window_scale(increase=True)
            elif key == ord('c') or key == ord('C'):
                system.toggle_confidence()
            elif key == ord('m'):  # lowercase m for mirror
                system.toggle_mirror_mode()
            elif key == ord('M'):  # uppercase M for landmarks (keep existing)
                system.toggle_landmarks()
            elif key == 65470:  # F1 key
                system.adjust_face_detection_sensitivity(increase=False)
            elif key == 65471:  # F2 key
                system.adjust_face_detection_sensitivity(increase=True)
    
    finally:
        # Save final history
        if system.action_history:
            system._save_action_history()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nFace Locking System terminated")
        print(f"Final session stats:")
        print(f"  - Total actions recorded: {len(system.action_history)}")
        print(f"  - Target identity: {system.target_identity}")
        print(f"  - Final smile threshold: {system.action_detector.smile_ratio_threshold:.1f}")


if __name__ == "__main__":
    main()