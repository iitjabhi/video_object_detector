#!/usr/bin/env python3
"""
Production Video Processing Pipeline - Object Detection with YOLO
Extracts frames from video and performs object detection, outputting COCO-format annotations.

Some parts could probably be optimized but works for most basic use cases.
"""

import cv2
import os
import json
import argparse
import sys
import time
import logging
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import shutil

try:
    from ultralytics import YOLO
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please run: pip install ultralytics tqdm")
    sys.exit(1)


class VideoProcessingError(Exception):
    """Custom exception for video processing validation errors."""
    pass


class VideoProcessor:
    """Video processing pipeline with basic features.
    
    This class handles the main video processing workflow - extracting frames,
    running object detection, and generating output files.
    """
    
    def __init__(self, client_id="default", max_workers=2, skip_similar_frames=True):
        self.client_id = client_id
        self.num_workers = max_workers
        self.skip_similar_frames = skip_similar_frames
        self.prev_frame_hash = None
        self.yolo_model = None
        
        # Setup simple logging
        self.setup_logging()
        
        # Track metrics - this got a bit complex but helps with debugging
        self.stats = {  # using 'stats' instead of 'metrics' for variety
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'extracted_images': 0,
            'total_detections': 0,
            'detections_per_frame': [],
            'class_distribution': {},
            'frame_drop_ratio': 0.0,
            'start_time': datetime.now(),
            'stage_times': {
                'frame_extraction': 0,
                'object_detection': 0,
                'output_validation': 0,
                'total_processing': 0
            }
        }
        
        self.logger.info(f"Initialized video processor for client: {client_id}")
    
    def setup_logging(self):
        """Setup basic logging that logs to file as well as console.
        """
        # Create logs directory if it doesn't exist
        log_dir = f"logs/{self.client_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/pipeline.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"video_pipeline_{self.client_id}")
        self.logger.info("Logging setup complete")
    
    def calc_frame_hash(self, frame):
        """Calculate simple hash for frame similarity detection.
        
        Approach - resize to small image and hash the bytes.
        TODO: Use more sophisticated methods (perceptual hashing).
        """
        if not self.skip_similar_frames:
            return ""
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray_frame, (64, 64))
        return hashlib.md5(small_frame.tobytes()).hexdigest()
    
    def is_similar_frame(self, frame_hash):
        """Check if frame is too similar to previous one.
        
        Cmpares hashes - catches duplicates.
        """
        if not self.skip_similar_frames or not self.prev_frame_hash:
            return False
        return frame_hash == self.prev_frame_hash
    
    def load_model(self, model_name):
        """Load YOLO model with basic retry logic to handle occasional failures.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Loading model: {model_name} (attempt {attempt + 1})")
                self.yolo_model = YOLO(model_name)
                self.logger.info("Model loaded successfully")
                return
            except Exception as e:
                self.logger.warning(f"Model loading failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    raise Exception(f"Failed to load model after {max_retries} attempts")

    def _validate_video_format(self, video_path):
        """Validate video file format and codec compatibility."""
        self.logger.info(f"Validating video format: {video_path}")
        
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in valid_extensions:
            raise VideoProcessingError(f"Unsupported video format: {file_ext}. Supported formats: {', '.join(valid_extensions)}")
        
        # Test video properties with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {video_path}")
        
        try:
            # Check if video has frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise VideoProcessingError(f"Video has no frames: {video_path}")
            
            # Check if video has reasonable frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise VideoProcessingError(f"Invalid frame rate: {fps}")
            
            # Check video dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise VideoProcessingError(f"Invalid video dimensions: {width}x{height}")
            
            # Test reading the first frame
            ret, frame = cap.read()
            if not ret or frame is None:
                raise VideoProcessingError(f"Cannot read first frame from video: {video_path}")
            
            self.logger.info(f"Video validation passed: {frame_count} frames, {fps:.1f} FPS, {width}x{height}")
            
        finally:
            cap.release()

    def _validate_frames_directory(self, frames_dir):
        """Validate frames directory contains valid image files."""
        self.logger.info(f"Validating frames directory: {frames_dir}")
        
        if not os.path.isdir(frames_dir):
            raise VideoProcessingError(f"Frames directory does not exist: {frames_dir}")
        
        # Check for supported image formats
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        all_files = os.listdir(frames_dir)
        image_files = [f for f in all_files 
                      if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if not image_files:
            raise VideoProcessingError(f"No valid image files found in: {frames_dir}. Supported formats: {', '.join(valid_extensions)}")
        
        # Test loading a sample image
        sample_image = os.path.join(frames_dir, image_files[0])
        try:
            img = cv2.imread(sample_image)
            if img is None:
                raise VideoProcessingError(f"Cannot read sample image: {sample_image}")
            
            height, width = img.shape[:2]
            if height <= 0 or width <= 0:
                raise VideoProcessingError(f"Invalid image dimensions: {width}x{height}")
            
        except Exception as e:
            raise VideoProcessingError(f"Invalid image format in directory: {e}")
        
        self.logger.info(f"Frames directory validation passed: {len(image_files)} images found")

    def _validate_output_directory(self, output_dir):
        """Validate output directory is writable and has sufficient space."""
        self.logger.info(f"Validating output directory: {output_dir}")
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            raise VideoProcessingError(f"Cannot create output directory: {output_dir}")
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            raise VideoProcessingError(f"Output directory not writable: {output_dir}")
        
        # Check available disk space (estimate needed space)
        required_space_mb = 100  # Minimum 100MB
        try:
            free_space = shutil.disk_usage(output_dir).free / 1024 / 1024
            if free_space < required_space_mb:
                raise VideoProcessingError(f"Insufficient disk space: {free_space:.1f}MB < {required_space_mb}MB")
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
        
        self.logger.info(f"Output directory validation passed: {output_dir}")

    def _validate_input_path(self, input_path):
        """Validate and determine input type (video file or frames directory)."""
        self.logger.info(f"Validating input path: {input_path}")
        
        if not os.path.exists(input_path):
            raise VideoProcessingError(f"Input path does not exist: {input_path}")
        
        if os.path.isfile(input_path):
            # It's a video file
            self._validate_video_format(input_path)
            return "video"
        elif os.path.isdir(input_path):
            # It's a frames directory
            self._validate_frames_directory(input_path)
            return "frames"
        else:
            raise VideoProcessingError(f"Input path is neither file nor directory: {input_path}")

    def _validate_model_name(self, model_name):
        """Validate YOLO model name."""
        self.logger.info(f"Validating model name: {model_name}")
        
        # Standard YOLO models
        standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        
        if model_name not in standard_models:
            # Check if it's a custom model file
            if os.path.exists(model_name):
                self.logger.info(f"Using custom model file: {model_name}")
            else:
                self.logger.warning(f"Model not in standard list: {model_name}. Will attempt download.")
        
        self.logger.info(f"Model name validation passed: {model_name}")

    def validate_all_inputs(self, input_path, output_dir, model_name):
        """Perform comprehensive Phase 1 validation."""
        self.logger.info("Starting comprehensive input validation...")
        
        try:
            # Validate input path and determine type
            input_type = self._validate_input_path(input_path)
            
            # Validate output directory
            self._validate_output_directory(output_dir)
            
            # Validate model name
            self._validate_model_name(model_name)
            
            self.logger.info("All input validation passed successfully!")
            return input_type
            
        except VideoProcessingError as e:
            self.logger.error(f"Input validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}")
            raise VideoProcessingError(f"Validation error: {e}")
    
    def extract_frames(self, video_path, output_dir, frame_step=30):
        """Extract frames from video with basic optimization.
        
        This extracts every nth frame from the video and saves as JPG files.
        Also invokes basic duplicate detection to avoid processing nearly identical frames.
        """
        self.logger.info(f"Starting frame extraction from: {video_path}")
        
        # Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Get video info
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        self.stats['total_frames'] = total_frames
        
        self.logger.info(f"Video info: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        saved_count = 0
        
        # Process frames - where the magic happens
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                if frame_count % frame_step == 0:
                    frame_hash = self.calc_frame_hash(frame)
                    
                    # Skip similar frames if enabled
                    if self.is_similar_frame(frame_hash):
                        self.stats['skipped_frames'] += 1
                        self.logger.debug(f"Skipped similar frame at {frame_count}")
                    else:
                        filename = f"frame_{saved_count:05d}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, frame)
                        saved_count += 1
                        self.stats['processed_frames'] += 1
                        self.stats['extracted_images'] += 1
                        self.prev_frame_hash = frame_hash
                
                frame_count += 1
                pbar.update(1)
        
        video_cap.release()
        
        # Calculate frame drop ratio
        total_candidate_frames = (total_frames // frame_step) + 1
        self.stats['frame_drop_ratio'] = self.stats['skipped_frames'] / max(total_candidate_frames, 1)
        
        self.logger.info(f"Extracted {saved_count} frames, skipped {self.stats['skipped_frames']} similar frames")
        self.logger.info(f"Frame drop ratio: {self.stats['frame_drop_ratio']:.2%}")
        return saved_count
    
    def process_frames_batch(self, img_files, frames_dir):
        """Process a batch of frames for object detection.
        
        This processes multiple images at once to make things more efficient.
        Each batch runs in parallel which speeds things up.
        """
        batch_results = []
        
        for img_file in img_files:
            try:
                image_path = os.path.join(frames_dir, img_file)
                
                detections = self.yolo_model(image_path)[0]
                height, width = detections.orig_shape
                
                frame_detections = []
                if detections.boxes is not None:
                    for box in detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf, class_id = box
                        if conf >= 0.5:  # Basic confidence threshold
                            class_name = self.yolo_model.names[int(class_id)]
                            frame_detections.append({
                                'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'confidence': conf,
                                'class_id': int(class_id),
                                'class_name': class_name
                            })
                            
                            # Track class distribution
                            if class_name not in self.stats['class_distribution']:
                                self.stats['class_distribution'][class_name] = 0
                            self.stats['class_distribution'][class_name] += 1
                
                # Track detections per frame
                self.stats['detections_per_frame'].append(len(frame_detections))
                
                batch_results.append({
                    'filename': img_file,
                    'width': width,
                    'height': height,
                    'detections': frame_detections
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {img_file}: {e}")
                continue
        
        return batch_results
    
    def detect_objects(self, frames_dir, output_file):
        """Detect objects in frames with simple parallel processing.
        
        This is where we actually run the AI model on all our extracted frames.
        Uses threading to speed things up - works pretty well on most hardware.
        """
        self.logger.info("Starting object detection")
        
        # Get list of image files
        image_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        image_files.sort()
        
        if not image_files:
            raise Exception(f"No images found in {frames_dir}")
        
        # Split into batches for parallel processing
        batch_size = 10
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        # Process batches in parallel (simplified approach)
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(self.process_frames_batch, batch, frames_dir)
                futures.append(future)
            
            # Collect results
            for future in tqdm(futures, desc="Processing batches"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
        
        # Convert to COCO format
        coco_data = self.convert_to_coco(all_results)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        self.logger.info(f"Detection complete. Results saved to: {output_file}")
        return len(coco_data['annotations'])
    
    def convert_to_coco(self, results):
        """Convert detection results to COCO format.
        """
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Track categories
        category_map = {}
        next_cat_id = 1
        next_img_id = 1
        next_anno_id = 1
        
        for result in results:
            # Add image info
            coco_data["images"].append({
                "id": next_img_id,
                "file_name": result['filename'],
                "width": result['width'],
                "height": result['height']
            })
            
            # Add detections
            for detection in result['detections']:
                class_name = detection['class_name']
                
                # Add category if new
                if class_name not in category_map:
                    category_map[class_name] = next_cat_id
                    coco_data["categories"].append({
                        "id": next_cat_id,
                        "name": class_name
                    })
                    next_cat_id += 1
                
                # Add annotation
                bbox = detection['bbox']
                coco_data["annotations"].append({
                    "id": next_anno_id,
                    "image_id": next_img_id,
                    "category_id": category_map[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "confidence": detection['confidence']
                })
                next_anno_id += 1
                self.stats['total_detections'] += 1
            
            next_img_id += 1
        
        return coco_data
    
    def cleanup(self, temp_dir):
        """Basic cleanup of temporary files; simply removes the temp directory
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def validate_outputs(self, output_dir, temp_frames_dir):
        """Simple validation - check that images were extracted and COCO file exists and is valid.
        
        This does basic sanity checks to make sure everything worked properly; catches the most common issues.
        """
        self.logger.info("Validating outputs...")
        
        # Check if any frame images were extracted
        if os.path.exists(temp_frames_dir):
            frame_files = [f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')]
            if not frame_files:
                self.logger.error("No frame images found - frame extraction may have failed!")
                return False
            self.logger.info(f"Found {len(frame_files)} extracted frames")
        else:
            self.logger.error("Frame directory not found!")
            return False
        
        # Check if detections.json exists
        detections_file = os.path.join(output_dir, "detections.json")
        if not os.path.exists(detections_file):
            self.logger.error("detections.json file not found!")
            return False
        
        try:
            # Check if JSON is valid and has expected structure
            with open(detections_file, 'r') as f:
                data = json.load(f)
            
            # Basic COCO format validation
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Missing required key '{key}' in COCO format")
                    return False
                elif not isinstance(data[key], list):
                    self.logger.error(f"Key '{key}' should be a list")
                    return False
            
            # Basic counts
            num_images = len(data['images'])
            num_annotations = len(data['annotations'])
            
            self.logger.info(f"Valid COCO file: {num_images} images, {num_annotations} detections")
            return True
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def log_stats(self):
        """Log comprehensive processing statistics.
        
        This prints out a nice summary of what happened during processing.
        Really helpful for debugging and understanding performance.
        """
        duration = datetime.now() - self.stats['start_time']
        
        self.logger.info("=== Processing Summary ===")
        self.logger.info(f"Total frames: {self.stats['total_frames']}")
        self.logger.info(f"Extracted images: {self.stats['extracted_images']}")
        self.logger.info(f"Skipped frames: {self.stats['skipped_frames']}")
        self.logger.info(f"Frame drop ratio: {self.stats['frame_drop_ratio']:.1%}")
        self.logger.info(f"Total detections: {self.stats['total_detections']}")
        
        if self.stats['detections_per_frame']:
            avg_detections = sum(self.stats['detections_per_frame']) / len(self.stats['detections_per_frame'])
            max_detections = max(self.stats['detections_per_frame'])
            self.logger.info(f"Avg detections per frame: {avg_detections:.1f}")
            self.logger.info(f"Max detections per frame: {max_detections}")
        
        self.logger.info(f"Processing time: {duration}")
        
        # Log class distribution
        if self.stats['class_distribution']:
            self.logger.info("Class distribution:")
            sorted_classes = sorted(self.stats['class_distribution'].items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:5]:  # Top 5 classes
                self.logger.info(f"  - {class_name}: {count}")
        
        self.logger.info("=== End Summary ===")
    
    def generate_report(self, output_dir, video_path):
        """Generate Markdown report with processing statistics.
        
        Creates a nice markdown report that you can view in any markdown reader.
        Makes it easy to share results with others or keep track of runs.
        """
        self.logger.info("📊 Generating processing report...")
        
        # Calculate processing time
        end_time = datetime.now()
        total_time = (end_time - self.stats['start_time']).total_seconds()
        self.stats['stage_times']['total_processing'] = total_time
        
        # Calculate statistics
        avg_detections = sum(self.stats['detections_per_frame']) / max(len(self.stats['detections_per_frame']), 1)
        max_detections = max(self.stats['detections_per_frame']) if self.stats['detections_per_frame'] else 0
        
        # Report data
        report_data = {
            'video_path': video_path,
            'client_id': self.client_id,
            'processing_time': f"{total_time:.1f}s",
            'total_frames': self.stats['total_frames'],
            'extracted_images': self.stats['extracted_images'],
            'skipped_frames': self.stats['skipped_frames'],
            'frame_drop_ratio': f"{self.stats['frame_drop_ratio']:.1%}",
            'total_detections': self.stats['total_detections'],
            'avg_detections_per_frame': f"{avg_detections:.1f}",
            'max_detections_per_frame': max_detections,
            'class_distribution': self.stats['class_distribution'],
            'stage_times': self.stats['stage_times']
        }
        
        # Generate Markdown report
        md_report = self._generate_markdown_report(report_data)
        md_path = os.path.join(output_dir, "processing_report.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        self.logger.info(f"📄 Report generated: {md_path}")
        return md_path
    

    def _generate_markdown_report(self, data):
        """Generate Markdown report.
        
        This creates the actual markdown content for the report.
        """
        md = f"""# 🎬 Video Processing Report

**Client:** {data['client_id']}  
**Video:** {data['video_path']}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Metrics

| Metric | Value |
|--------|-------|
| **Total Video Frames** | {data['total_frames']} |
| **Images Extracted** | {data['extracted_images']} |
| **Total Detections** | {data['total_detections']} |
| **Avg Detections/Frame** | {data['avg_detections_per_frame']} |
| **Frame Drop Ratio** | {data['frame_drop_ratio']} |
| **Processing Time** | {data['processing_time']} |

## Object Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|"""
        
        total_detections = sum(data['class_distribution'].values())
        for class_name, count in sorted(data['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            md += f"\n| {class_name} | {count} | {percentage:.1f}% |"
        
        md += f"""

## ⏱Processing Stage Breakdown

| Stage | Time | Percentage |
|-------|------|------------|
| Frame Extraction | {data['stage_times']['frame_extraction']:.1f}s | {(data['stage_times']['frame_extraction']/data['stage_times']['total_processing']*100):.1f}% |
| Object Detection | {data['stage_times']['object_detection']:.1f}s | {(data['stage_times']['object_detection']/data['stage_times']['total_processing']*100):.1f}% |
| Output Validation | {data['stage_times']['output_validation']:.1f}s | {(data['stage_times']['output_validation']/data['stage_times']['total_processing']*100):.1f}% |
| **Total** | **{data['stage_times']['total_processing']:.1f}s** | **100%** |

## Processing Details

- **Skipped Frames:** {data['skipped_frames']}
- **Max Detections in Single Frame:** {data['max_detections_per_frame']}
- **Frame Drop Ratio:** {data['frame_drop_ratio']} (lower is better)

---
*Generated by Video Processing Pipeline*
"""
        return md

    def process_frames_directory(self, frames_dir, output_dir, model_name="yolov8n.pt"):
        """Process pre-extracted frames from a directory.
        
        This is optimized for CI testing where frames are already extracted.
        Much faster than processing full videos.
        """
        try:
            self.logger.info(f"Starting frame processing for client: {self.client_id}")
            self.logger.info(f"Processing frames from: {frames_dir}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Load model
            self.load_model(model_name)
            
            # Step 2: Count frames (simulate frame extraction)
            frame_start = time.time()
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            num_frames = len(frame_files)
            
            # Update stats as if we extracted them
            self.stats['total_frames'] = num_frames  # In frames mode, this is the frame count
            self.stats['extracted_images'] = num_frames
            self.stats['skipped_frames'] = 0  # No skipping in frame mode
            self.stats['frame_drop_ratio'] = 0.0
            
            self.stats['stage_times']['frame_extraction'] = time.time() - frame_start
            
            if num_frames == 0:
                raise Exception("No frame files found in directory")
            
            self.logger.info(f"Found {num_frames} frames to process")
            
            # Step 3: Detect objects (use the existing frames directly)
            detection_start = time.time()
            output_file = os.path.join(output_dir, "detections.json")
            self.detect_objects(frames_dir, output_file)
            self.stats['stage_times']['object_detection'] = time.time() - detection_start
            
            # Step 4: Validate output
            validation_start = time.time()
            validation_passed = self.validate_outputs(output_dir, frames_dir)
            self.stats['stage_times']['output_validation'] = time.time() - validation_start
            
            # Step 5: Generate reports
            self.generate_report(output_dir, frames_dir)  # Pass frames_dir as "video_path"
            
            # Step 6: Log results
            self.log_stats()
            
            if validation_passed:
                self.logger.info("Frame processing completed successfully!")
            else:
                self.logger.warning("Processing completed but output validation failed")
                self.logger.info("Check the logs above for specific validation issues")
            
            return {
                'extracted_images': num_frames,
                'total_detections': self.stats['total_detections'],
                'processing_time': sum(self.stats['stage_times'].values()),
                'validation_passed': validation_passed
            }
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            raise

    def process_video(self, video_path, output_dir, model_name="yolov8n.pt", frame_step=30):
        """Main processing function - simplified workflow with comprehensive tracking.
        
        This is the main entry point that orchestrates the entire pipeline.
        It handles all the steps from loading the model to generating the final report.
        """
        try:
            self.logger.info(f"Starting video processing for client: {self.client_id}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create temporary directory for frames
            temp_frames_dir = os.path.join(output_dir, "temp_frames")
            
            # Step 1: Load model
            self.load_model(model_name)
            
            # Step 2: Extract frames
            frame_start = time.time()
            num_frames = self.extract_frames(video_path, temp_frames_dir, frame_step)
            self.stats['stage_times']['frame_extraction'] = time.time() - frame_start
            
            if num_frames == 0:
                raise Exception("No frames extracted from video")
            
            # Step 3: Detect objects
            detection_start = time.time()
            output_file = os.path.join(output_dir, "detections.json")
            self.detect_objects(temp_frames_dir, output_file)
            self.stats['stage_times']['object_detection'] = time.time() - detection_start
            
            # Step 4: Validate output
            validation_start = time.time()
            validation_passed = self.validate_outputs(output_dir, temp_frames_dir)
            self.stats['stage_times']['output_validation'] = time.time() - validation_start
            
            # Step 5: Generate reports
            self.generate_report(output_dir, video_path)
            
            # Step 6: Log results
            self.log_stats()
            
            if validation_passed:
                self.logger.info("Video processing completed successfully!")
            else:
                self.logger.warning("⚠️Processing completed but output validation failed")
                self.logger.info("Check the logs above for specific validation issues")
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

        finally:
            self.cleanup(temp_frames_dir)


def main():
    """Main entry point with simplified argument parsing.
    
    Auto-detects whether input is a video file or frames directory.
    """
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
    
    args = parser.parse_args()
    
    # Create processor
    processor = VideoProcessor(
        client_id=args.client_id,
        max_workers=args.max_workers,
        skip_similar_frames=not args.disable_frame_skipping
    )
    
    try:
        # Validate inputs and determine input type
        input_type = processor.validate_all_inputs(args.input_path, args.output_dir, args.model)
        
        if input_type == "video":
            print(f"Processing video: {args.input_path}")
            # Video-based processing
            processor.process_video(
                video_path=args.input_path,
                output_dir=args.output_dir,
                model_name=args.model,
                frame_step=args.frame_step
            )
        else:
            print(f"Processing pre-extracted frames from: {args.input_path}")
            # Frame-based processing (fast CI mode)
            results = processor.process_frames_directory(
                frames_dir=args.input_path,
                output_dir=args.output_dir,
                model_name=args.model
            )
            print(f"Results: {results['extracted_images']} frames, {results['total_detections']} detections")
        
        print("Processing completed successfully!")
        return 0
        
    except VideoProcessingError as e:
        print(f"Validation failed: {e}")
        return 1
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 