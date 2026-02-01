# src/face_locking.py
"""
Face Locking Feature Implementation

This module extends the face recognition system with face locking capabilities:
- Manual face selection for a specific enrolled identity
- Stable face tracking across frames
- Action detection (movement, blinks, smiles)
- Action history recording to timestamped files

Usage:
    python -m src.face_locking

Controls:
    q: quit
    r: reload database
    l: toggle lock on/off for target identity
    +/-: adjust recognition threshold
    s: save current action history
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
    """Main face locking system"""
    
    def __init__(self, target_identity: str = "Wilson", 
                 db_path: str = "data/db/face_db.npz",
                 model_path: str = "models/embedder_arcface.onnx"):
        
        self.target_identity = target_identity
        self.db_path = Path(db_path)
        
        # Initialize components
        self.detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
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
        
        # Verify target identity exists
        if target_identity not in db:
            available = list(db.keys())
            raise ValueError(f"Target identity '{target_identity}' not found in database. "
                           f"Available: {available}")
        
        print(f"Face Locking System initialized for: {target_identity}")
        print(f"Database contains {len(db)} identities: {list(db.keys())}")
    
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
        """Save action history to file"""
        if not self.action_history:
            print("No actions to save")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.target_identity.lower()}_history_{timestamp}.txt"
        filepath = Path("data") / filename
        
        # Ensure data directory exists
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for face locking"""
        self.frame_count += 1
        vis = frame.copy()
        
        # Detect faces
        faces = self.detector.detect(frame, max_faces=5)
        
        # If we have a lock, try to maintain it
        if self.is_locked and self.tracker:
            self._update_locked_face(frame, faces, vis)
        else:
            # Look for target identity to lock onto
            self._search_for_target(frame, faces, vis)
        
        # Draw UI
        self._draw_ui(vis)
        
        return vis
    
    def _update_locked_face(self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray):
        """Update tracking for locked face"""
        target_face = None
        
        # Try to find the same face based on position
        for face in faces:
            if self._is_same_face(face, self.tracker):
                # Verify it's still the target identity
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                emb = self.embedder.embed(aligned)
                match_result = self.matcher.match(emb)
                
                if match_result.accepted and match_result.name == self.target_identity:
                    target_face = face
                    self.tracker.confidence_history.append(match_result.similarity)
                    break
        
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
    
    def _search_for_target(self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray):
        """Search for target identity to lock onto"""
        for face in faces:
            # Recognize face
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            match_result = self.matcher.match(emb)
            
            # Draw recognition result
            color = (0, 255, 0) if match_result.accepted else (0, 0, 255)
            label = match_result.name if match_result.name else "Unknown"
            
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
            cv2.putText(vis, f"{label} ({match_result.similarity:.2f})", 
                       (face.x1, face.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check if this is our target and we should lock
            if (match_result.accepted and 
                match_result.name == self.target_identity and 
                match_result.similarity > 0.7):  # High confidence threshold for locking
                
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
        """Draw locked face with special highlighting"""
        # Thick green border for locked face
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 4)
        
        # Lock icon (simple)
        lock_x, lock_y = face.x1 + 10, face.y1 + 10
        cv2.rectangle(vis, (lock_x, lock_y), (lock_x + 20, lock_y + 15), (0, 255, 0), 2)
        cv2.circle(vis, (lock_x + 10, lock_y + 5), 5, (0, 255, 0), 2)
        
        # Status text
        status_text = f"LOCKED: {self.target_identity}"
        cv2.putText(vis, status_text, (face.x1, face.y1 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Confidence
        if self.tracker and self.tracker.confidence_history:
            avg_conf = np.mean(list(self.tracker.confidence_history))
            conf_text = f"Conf: {avg_conf:.2f}"
            cv2.putText(vis, conf_text, (face.x1, face.y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw landmarks
        for x, y in face.kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    def _draw_ui(self, vis: np.ndarray):
        """Draw user interface elements"""
        h, w = vis.shape[:2]
        
        # Status header
        status = "LOCKED" if self.is_locked else "SEARCHING"
        color = (0, 255, 0) if self.is_locked else (0, 255, 255)
        
        header = f"Face Locking: {status} | Target: {self.target_identity}"
        cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Action count
        action_text = f"Actions recorded: {len(self.action_history)}"
        cv2.putText(vis, action_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        controls = [
            "Controls:",
            "L - Toggle lock",
            "S - Save history", 
            "R - Reload DB",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            y_pos = h - 120 + (i * 20)
            cv2.putText(vis, control, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
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
    """Main face locking demo"""
    
    # Initialize system
    try:
        system = FaceLockingSystem(target_identity="Wilson")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    
    print("\n" + "="*60)
    print("FACE LOCKING SYSTEM")
    print("="*60)
    print(f"Target Identity: {system.target_identity}")
    print("\nControls:")
    print("  L - Toggle lock on/off")
    print("  S - Save action history")
    print("  R - Reload database")
    print("  Q - Quit")
    print("="*60)
    
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
            
            if key == ord('q'):
                break
            elif key == ord('l') or key == ord('L'):
                system.toggle_lock()
            elif key == ord('s') or key == ord('S'):
                filepath = system._save_action_history()
                if filepath:
                    print(f"History saved to: {filepath}")
            elif key == ord('r') or key == ord('R'):
                system.reload_database()
    
    finally:
        # Save final history
        if system.action_history:
            system._save_action_history()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nFace Locking System terminated")


if __name__ == "__main__":
    main()