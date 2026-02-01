#!/usr/bin/env python3
"""
Test script for Face Locking System

This script tests the face locking functionality without requiring a camera.
It verifies that all components are properly imported and initialized.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.face_locking import FaceLockingSystem, ActionRecord, FaceTracker
        print("✓ Face locking system imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import face locking system: {e}")
        return False
    
    try:
        from src.action_detection import AdvancedActionDetector, ActionClassifier
        print("✓ Action detection modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import action detection: {e}")
        return False
    
    try:
        from src.recognize import HaarFaceMesh5pt, ArcFaceEmbedderONNX, FaceDBMatcher
        print("✓ Recognition modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import recognition modules: {e}")
        return False
    
    return True

def test_database():
    """Test database loading"""
    print("\nTesting database...")
    
    try:
        from src.recognize import load_db_npz
        from pathlib import Path
        
        db_path = Path("data/db/face_db.npz")
        if not db_path.exists():
            print(f"✗ Database file not found: {db_path}")
            return False
        
        db = load_db_npz(db_path)
        print(f"✓ Database loaded with {len(db)} identities: {list(db.keys())}")
        
        if "Wilson" not in db:
            print("✗ Target identity 'Wilson' not found in database")
            return False
        
        print("✓ Target identity 'Wilson' found in database")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_model():
    """Test model loading"""
    print("\nTesting model...")
    
    try:
        model_path = Path("models/embedder_arcface.onnx")
        if not model_path.exists():
            print(f"✗ Model file not found: {model_path}")
            print("Please download the ArcFace model as described in README.md")
            return False
        
        print(f"✓ Model file found: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_initialization():
    """Test system initialization"""
    print("\nTesting system initialization...")
    
    try:
        from src.face_locking import FaceLockingSystem
        
        # This will fail if database or model is missing
        system = FaceLockingSystem(target_identity="Wilson")
        print("✓ Face locking system initialized successfully")
        
        # Test action detector
        detector = system.action_detector
        print("✓ Advanced action detector initialized")
        
        # Test action classifier
        classifier = system.action_classifier
        print("✓ Action classifier initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False

def test_action_detection():
    """Test action detection algorithms"""
    print("\nTesting action detection...")
    
    try:
        from src.action_detection import AdvancedActionDetector
        import numpy as np
        
        detector = AdvancedActionDetector()
        
        # Test with dummy landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        landmarks = np.array([
            [100, 150],  # left eye
            [200, 150],  # right eye  
            [150, 180],  # nose
            [130, 220],  # left mouth
            [170, 220]   # right mouth
        ], dtype=np.float32)
        
        # Test blink detection
        blink_detected, eye_metrics = detector.detect_blink_advanced(landmarks, 0.0)
        print(f"✓ Blink detection test: detected={blink_detected}, EAR={eye_metrics.avg_eye_ratio:.3f}")
        
        # Test smile detection
        smile_detected, mouth_metrics = detector.detect_smile_advanced(landmarks, 0.0)
        print(f"✓ Smile detection test: detected={smile_detected}, ratio={mouth_metrics.mouth_ratio:.2f}")
        
        # Test movement detection
        movement = detector.detect_movement_advanced((150, 180), 0.0)
        print(f"✓ Movement detection test: {movement}")
        
        return True
        
    except Exception as e:
        print(f"✗ Action detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Face Locking System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_database, 
        test_model,
        test_initialization,
        test_action_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Face locking system is ready to use.")
        print("\nTo run the face locking system:")
        print("  python -m src.face_locking")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        
        if passed < 3:
            print("\nCommon issues:")
            print("- Make sure you've run: python init_project.py")
            print("- Download the ArcFace model as described in README.md")
            print("- Enroll at least one person using: python -m src.enroll")

if __name__ == "__main__":
    main()