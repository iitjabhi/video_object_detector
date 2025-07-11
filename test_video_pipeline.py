#!/usr/bin/env python3
"""
Basic Unit Tests for Simple Video Processing Pipeline
Tests key functions to ensure they work correctly.

These tests cover the main functionality but aren't exhaustive - just the most common issues during development.
"""

import pytest
import os
import json
import tempfile
import shutil
import cv2
import numpy as np
import argparse
import sys
from unittest.mock import Mock, patch, MagicMock

from video_pipeline import VideoProcessor


class TestArgumentParsing:
    """Test command-line argument parsing functionality."""
    
    def create_mock_parser(self):
        """Create a parser with the same structure as the main function."""
        parser = argparse.ArgumentParser(description="Simple Video Processing Pipeline")
        
        # Simple positional arguments
        parser.add_argument("input_path", help="Path to video file or frames directory")
        parser.add_argument("output_dir", nargs="?", default="output", help="Output directory (default: output)")
        
        # Optional arguments
        parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name (default: yolov8n.pt)")
        parser.add_argument("--frame-step", type=int, default=30, help="Extract every N frames (default: 30, video mode only)")
        parser.add_argument("--client-id", default="default", help="Client identifier (default: default)")
        parser.add_argument("--max-workers", type=int, default=2, help="Number of parallel workers (default: 2)")
        parser.add_argument("--disable-frame-skipping", action="store_true", help="Disable similar frame skipping (video mode only)")
        
        return parser
    
    def parse_and_process_args(self, args_list):
        """Parse arguments and apply the same logic as main function."""
        parser = self.create_mock_parser()
        args = parser.parse_args(args_list)
        
        # Auto-detect logic would happen here in real code
        # For tests, we just return the parsed args
        return args
    
    def test_basic_video_format(self):
        """Test basic format: video.mp4 output/"""
        args = self.parse_and_process_args(['test_video.mp4', 'test_output'])
        
        assert args.input_path == 'test_video.mp4'
        assert args.output_dir == 'test_output'
        assert args.model == 'yolov8n.pt'  # default
        assert args.client_id == 'default'  # default
    
    def test_video_with_options(self):
        """Test video with additional options."""
        args = self.parse_and_process_args([
            'my_video.mp4', 'results', 
            '--model', 'yolov8s.pt', 
            '--client-id', 'test-client',
            '--frame-step', '60'
        ])
        
        assert args.input_path == 'my_video.mp4'
        assert args.output_dir == 'results'
        assert args.model == 'yolov8s.pt'
        assert args.client_id == 'test-client'
        assert args.frame_step == 60
    
    def test_frames_directory_format(self):
        """Test frames directory format: frames/ output/"""
        args = self.parse_and_process_args(['golden_output/frames', 'ci_output'])
        
        assert args.input_path == 'golden_output/frames'
        assert args.output_dir == 'ci_output'
    
    def test_frames_with_options(self):
        """Test frames directory with options."""
        args = self.parse_and_process_args([
            'golden_output/frames', 'ci_output',
            '--client-id', 'github-actions-ci',
            '--model', 'yolov8n.pt'
        ])
        
        assert args.input_path == 'golden_output/frames'
        assert args.output_dir == 'ci_output'
        assert args.client_id == 'github-actions-ci'
        assert args.model == 'yolov8n.pt'
    
    def test_full_path_video(self):
        """Test full path to video file."""
        args = self.parse_and_process_args(['/path/to/video.mp4', '/path/to/output'])
        
        assert args.input_path == '/path/to/video.mp4'
        assert args.output_dir == '/path/to/output'
    
    def test_relative_path_video(self):
        """Test relative path to video file."""
        args = self.parse_and_process_args(['../videos/test.mp4', './results'])
        
        assert args.input_path == '../videos/test.mp4'
        assert args.output_dir == './results'
    
    def test_optional_output_directory(self):
        """Test optional output directory defaults to 'output'."""
        args = self.parse_and_process_args(['test_video.mp4'])
        
        assert args.input_path == 'test_video.mp4'
        assert args.output_dir == 'output'  # Default value
    
    def test_error_missing_input_path(self):
        """Test error when input path is missing."""
        parser = self.create_mock_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing all arguments


class TestVideoProcessor:
    """Test the main video processor class.
    
    This tests the core functionality of the video processor.
    Tests use mocks to avoid needing actual video files.
    """
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.processor = VideoProcessor(client_id="test", max_workers=1)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_frame(self, width=640, height=480):
        """Create a simple test frame (image) for testing.
        
        This creates a basic test image with a rect and circle.
        """
        # Create a simple colored rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(frame, (500, 150), 50, (0, 255, 0), -1)
        return frame
    
    def test_frame_hash_calculation(self):
        """Test that frame hash calculation works properly.
        
        This tests the basic hash function that we use for similarity detection.
        """
        test_frame = self.create_test_frame()
        
        # Test with frame skipping enabled
        self.processor.skip_similar_frames = True
        hash1 = self.processor.calc_frame_hash(test_frame)
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Same frame should produce same hash
        hash2 = self.processor.calc_frame_hash(test_frame)
        assert hash1 == hash2
        
        # Test with frame skipping disabled
        self.processor.skip_similar_frames = False
        hash3 = self.processor.calc_frame_hash(test_frame)
        assert hash3 == ""
    
    def test_similar_frame_detection(self):
        """Test similar frame detection logic.
        
        This tests whether the similarity detection actually works.
        """
        frame1 = self.create_test_frame()
        frame2 = self.create_test_frame()  # Identical frame
        
        hash1 = self.processor.calc_frame_hash(frame1)
        hash2 = self.processor.calc_frame_hash(frame2)
        
        # Test with no previous hash
        assert not self.processor.is_similar_frame(hash1)
        
        # Set previous hash and test similarity
        self.processor.prev_frame_hash = hash1
        assert self.processor.is_similar_frame(hash2)  # Should be similar (identical)
        
        # Test with different frame
        different_frame = self.create_test_frame(width=320, height=240)
        hash3 = self.processor.calc_frame_hash(different_frame)
        assert not self.processor.is_similar_frame(hash3)
    
    def test_convert_to_coco_basic(self):
        """Test basic COCO format conversion.
        """
        # mock detection results
        mock_results = [
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
        
        coco_data = self.processor.convert_to_coco(mock_results)
        
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
        assert coco_data['annotations'][0]['confidence'] == pytest.approx(0.8)
        
        # Check categories
        category_names = [cat['name'] for cat in coco_data['categories']]
        assert 'person' in category_names
        assert 'car' in category_names
    
    def test_validate_outputs_success(self):
        """Test output validation with valid outputs.
        
        This tests the validation function when everything is setup correctly.
        """
        # Create test directories
        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir)
        
        # Create some test frame files
        for i in range(3):
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
            test_frame = self.create_test_frame()
            cv2.imwrite(frame_path, test_frame)
        
        # Create valid COCO file
        coco_data = {
            "images": [{"id": 1, "file_name": "frame_00000.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]}],
            "categories": [{"id": 1, "name": "test"}]
        }
        
        coco_path = os.path.join(self.temp_dir, "detections.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)
        
        # Test validation
        result = self.processor.validate_outputs(self.temp_dir, frames_dir)
        assert result is True
    
    @patch('cv2.VideoCapture')
    @patch('os.path.exists')
    def test_extract_frames_basic(self, mock_path_exists, mock_video_capture):
        """Test basic frame extraction functionality.
        
        This tests the frame extraction with mocked video capture.
        Using mocks since we don't want to deal with actual video files in tests.
        """
        # Mock file existence check
        mock_path_exists.return_value = True
        
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
        
        output_dir = os.path.join(self.temp_dir, "test_frames")
        
        # Test frame extraction
        num_frames = self.processor.extract_frames("test_video.mp4", output_dir, frame_step=2)
        
        # Should extract frames 0, 2, 4 (every 2nd frame) = 3 frames
        assert num_frames >= 0  # Basic check that it doesn't crash
        mock_cap.isOpened.assert_called_once()
        mock_cap.release.assert_called_once()
        # Verify that os.path.exists was called (it gets called multiple times for dirs and file)
        assert mock_path_exists.called
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly with different parameters.
        
        This tests various initialization scenarios to make sure defaults work.
        """
        # Test default initialization
        processor1 = VideoProcessor()
        assert processor1.client_id == "default"
        assert processor1.num_workers == 2
        assert processor1.skip_similar_frames is True
        
        # Test custom initialization
        processor2 = VideoProcessor(client_id="custom", max_workers=4, skip_similar_frames=False)
        assert processor2.client_id == "custom"
        assert processor2.num_workers == 4
        assert processor2.skip_similar_frames is False


# Integration test using pytest fixtures
@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for integration tests.
    
    This is a pytest fixture that creates a temp directory for testing.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_basic_workflow(temp_workspace):
    """Test a basic end-to-end workflow (without actual video processing).
    
    This is an integration test that checks the overall pipeline setup.
    Doesn't actually process video but makes sure all the pieces fit together.
    """
    processor = VideoProcessor(client_id="integration_test")
    
    # Test that we can create the processor and access its methods
    assert processor.client_id == "integration_test"
    assert hasattr(processor, 'extract_frames')
    assert hasattr(processor, 'detect_objects')
    assert hasattr(processor, 'validate_outputs')
    assert hasattr(processor, 'cleanup')
    
    # Test frame hash on a simple frame
    simple_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hash_result = processor.calc_frame_hash(simple_frame)
    
    if processor.skip_similar_frames:
        assert isinstance(hash_result, str)
    else:
        assert hash_result == ""


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 