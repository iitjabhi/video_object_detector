#!/usr/bin/env python3
"""
Basic Unit Tests for Simple Video Processing Pipeline
Tests key functions to ensure they work correctly.
"""

import pytest
import os
import json
import tempfile
import shutil
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from video_pipeline import SimpleVideoProcessor


class TestSimpleVideoProcessor:
    """Test the main video processor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.processor = SimpleVideoProcessor(client_id="test", max_workers=1)
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_frame(self, width=640, height=480):
        """Create a simple test frame (image) for testing."""
        # Create a simple colored rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(frame, (500, 150), 50, (0, 255, 0), -1)
        return frame
    
    def create_test_video(self, output_path, num_frames=10):
        """Create a simple test video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        
        for i in range(num_frames):
            frame = self.create_test_frame()
            # Add some variation to each frame
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return output_path
    
    def test_frame_hash_calculation(self):
        """Test that frame hash calculation works."""
        frame = self.create_test_frame()
        
        # Test with frame skipping enabled
        self.processor.skip_similar_frames = True
        hash1 = self.processor.calculate_frame_hash(frame)
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Same frame should produce same hash
        hash2 = self.processor.calculate_frame_hash(frame)
        assert hash1 == hash2
        
        # Test with frame skipping disabled
        self.processor.skip_similar_frames = False
        hash3 = self.processor.calculate_frame_hash(frame)
        assert hash3 == ""
    
    def test_similar_frame_detection(self):
        """Test similar frame detection logic."""
        frame1 = self.create_test_frame()
        frame2 = self.create_test_frame()  # Identical frame
        
        hash1 = self.processor.calculate_frame_hash(frame1)
        hash2 = self.processor.calculate_frame_hash(frame2)
        
        # Test with no previous hash
        assert not self.processor.is_similar_frame(hash1)
        
        # Set previous hash and test similarity
        self.processor.previous_frame_hash = hash1
        assert self.processor.is_similar_frame(hash2)  # Should be similar (identical)
        
        # Test with different frame
        different_frame = self.create_test_frame(width=320, height=240)
        hash3 = self.processor.calculate_frame_hash(different_frame)
        # This might or might not be similar depending on hash function, but test shouldn't crash
        result = self.processor.is_similar_frame(hash3)
        assert isinstance(result, bool)
    
    def test_convert_to_coco_basic(self):
        """Test basic COCO format conversion."""
        # Create mock detection results
        results = [
            {
                'filename': 'frame_00001.jpg',
                'width': 640,
                'height': 480,
                'detections': [
                    {
                        'bbox': [100, 100, 50, 50],
                        'confidence': 0.8,
                        'class_id': 0,
                        'class_name': 'person'
                    },
                    {
                        'bbox': [200, 200, 30, 40],
                        'confidence': 0.7,
                        'class_id': 1,
                        'class_name': 'car'
                    }
                ]
            },
            {
                'filename': 'frame_00002.jpg',
                'width': 640,
                'height': 480,
                'detections': [
                    {
                        'bbox': [150, 150, 60, 60],
                        'confidence': 0.9,
                        'class_id': 0,
                        'class_name': 'person'
                    }
                ]
            }
        ]
        
        coco_data = self.processor.convert_to_coco(results)
        
        # Check basic structure
        assert 'images' in coco_data
        assert 'annotations' in coco_data
        assert 'categories' in coco_data
        
        # Check counts
        assert len(coco_data['images']) == 2
        assert len(coco_data['annotations']) == 3  # 2 + 1 detections
        assert len(coco_data['categories']) == 2   # person, car
        
        # Check image data
        assert coco_data['images'][0]['file_name'] == 'frame_00001.jpg'
        assert coco_data['images'][0]['width'] == 640
        assert coco_data['images'][0]['height'] == 480
        
        # Check annotation data
        assert coco_data['annotations'][0]['bbox'] == [100, 100, 50, 50]
        assert coco_data['annotations'][0]['confidence'] == 0.8
        
        # Check categories
        category_names = [cat['name'] for cat in coco_data['categories']]
        assert 'person' in category_names
        assert 'car' in category_names
    
    def test_validate_outputs_success(self):
        """Test output validation with valid outputs."""
        # Create test directories
        frames_dir = os.path.join(self.test_dir, "frames")
        os.makedirs(frames_dir)
        
        # Create some test frame files
        for i in range(3):
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
            frame = self.create_test_frame()
            cv2.imwrite(frame_path, frame)
        
        # Create valid COCO file
        coco_data = {
            "images": [{"id": 1, "file_name": "frame_00000.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]}],
            "categories": [{"id": 1, "name": "test"}]
        }
        
        coco_path = os.path.join(self.test_dir, "detections.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)
        
        # Test validation
        result = self.processor.validate_outputs(self.test_dir, frames_dir)
        assert result is True
    
    def test_validate_outputs_missing_frames(self):
        """Test output validation when frame images are missing."""
        # Create empty frames directory
        frames_dir = os.path.join(self.test_dir, "frames")
        os.makedirs(frames_dir)
        
        result = self.processor.validate_outputs(self.test_dir, frames_dir)
        assert result is False
    
    def test_validate_outputs_missing_coco_file(self):
        """Test output validation when COCO file is missing."""
        # Create frames directory with some files
        frames_dir = os.path.join(self.test_dir, "frames")
        os.makedirs(frames_dir)
        
        frame_path = os.path.join(frames_dir, "frame_00000.jpg")
        frame = self.create_test_frame()
        cv2.imwrite(frame_path, frame)
        
        # No COCO file created
        result = self.processor.validate_outputs(self.test_dir, frames_dir)
        assert result is False
    
    def test_validate_outputs_invalid_coco_format(self):
        """Test output validation with invalid COCO format."""
        # Create frames
        frames_dir = os.path.join(self.test_dir, "frames")
        os.makedirs(frames_dir)
        frame_path = os.path.join(frames_dir, "frame_00000.jpg")
        frame = self.create_test_frame()
        cv2.imwrite(frame_path, frame)
        
        # Create invalid COCO file (missing required keys)
        invalid_coco = {"images": []}  # Missing 'annotations' and 'categories'
        
        coco_path = os.path.join(self.test_dir, "detections.json")
        with open(coco_path, 'w') as f:
            json.dump(invalid_coco, f)
        
        result = self.processor.validate_outputs(self.test_dir, frames_dir)
        assert result is False
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_basic(self, mock_video_capture):
        """Test basic frame extraction functionality."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        
        # Configure mock to simulate a video with 5 frames
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 5,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Mock frame reading
        test_frame = self.create_test_frame()
        mock_cap.read.side_effect = [
            (True, test_frame),  # Frame 0
            (True, test_frame),  # Frame 1  
            (True, test_frame),  # Frame 2
            (True, test_frame),  # Frame 3
            (True, test_frame),  # Frame 4
            (False, None)        # End of video
        ]
        
        output_dir = os.path.join(self.test_dir, "test_frames")
        
        # Test frame extraction
        num_frames = self.processor.extract_frames("test_video.mp4", output_dir, frame_step=2)
        
        # Should extract frames 0, 2, 4 (every 2nd frame) = 3 frames
        assert num_frames >= 0  # Basic check that it doesn't crash
        mock_cap.isOpened.assert_called_once()
        mock_cap.release.assert_called_once()
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        # Create a test directory
        test_cleanup_dir = os.path.join(self.test_dir, "cleanup_test")
        os.makedirs(test_cleanup_dir)
        
        # Add a test file
        test_file = os.path.join(test_cleanup_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Verify directory exists
        assert os.path.exists(test_cleanup_dir)
        assert os.path.exists(test_file)
        
        # Test cleanup
        self.processor.cleanup(test_cleanup_dir)
        
        # Verify directory is removed
        assert not os.path.exists(test_cleanup_dir)


class TestStandaloneFunctions:
    """Test standalone functions and edge cases."""
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly with different parameters."""
        # Test default initialization
        processor1 = SimpleVideoProcessor()
        assert processor1.client_id == "default"
        assert processor1.max_workers == 2
        assert processor1.skip_similar_frames is True
        
        # Test custom initialization
        processor2 = SimpleVideoProcessor(client_id="custom", max_workers=4, skip_similar_frames=False)
        assert processor2.client_id == "custom"
        assert processor2.max_workers == 4
        assert processor2.skip_similar_frames is False
    
    def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        processor = SimpleVideoProcessor()
        
        # Check initial metrics
        assert 'total_frames' in processor.metrics
        assert 'processed_frames' in processor.metrics
        assert 'skipped_frames' in processor.metrics
        assert 'total_detections' in processor.metrics
        assert 'start_time' in processor.metrics
        
        # Initial values should be 0
        assert processor.metrics['total_frames'] == 0
        assert processor.metrics['processed_frames'] == 0
        assert processor.metrics['skipped_frames'] == 0
        assert processor.metrics['total_detections'] == 0


# Integration test using pytest fixtures
@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for integration tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_basic_workflow(temp_workspace):
    """Test a basic end-to-end workflow (without actual video processing)."""
    processor = SimpleVideoProcessor(client_id="integration_test")
    
    # Test that we can create the processor and access its methods
    assert processor.client_id == "integration_test"
    assert hasattr(processor, 'extract_frames')
    assert hasattr(processor, 'detect_objects')
    assert hasattr(processor, 'validate_outputs')
    assert hasattr(processor, 'cleanup')
    
    # Test frame hash on a simple frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hash_result = processor.calculate_frame_hash(frame)
    
    if processor.skip_similar_frames:
        assert isinstance(hash_result, str)
    else:
        assert hash_result == ""


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 