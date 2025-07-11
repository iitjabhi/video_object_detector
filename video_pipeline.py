#!/usr/bin/env python3
"""
Simple Video Processing Pipeline - Object Detection with YOLO
A learner-friendly version with simplified implementations.
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


class SimpleVideoProcessor:
    """Simple video processing pipeline with basic features."""
    
    def __init__(self, client_id="default", max_workers=2, skip_similar_frames=True):
        self.client_id = client_id
        self.max_workers = max_workers
        self.skip_similar_frames = skip_similar_frames
        self.previous_frame_hash = None
        self.model = None
        
        # Setup simple logging
        self.setup_logging()
        
        # Track comprehensive metrics
        self.metrics = {
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
        """Setup basic logging - much simpler than the expert version."""
        # Create logs directory if it doesn't exist
        log_dir = f"logs/{self.client_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup basic logging
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
    
    def calculate_frame_hash(self, frame):
        """Calculate simple hash for frame similarity detection."""
        if not self.skip_similar_frames:
            return ""
        
        # Simple approach: convert to grayscale and hash
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def is_similar_frame(self, frame_hash):
        """Check if frame is similar to previous one."""
        if not self.skip_similar_frames or not self.previous_frame_hash:
            return False
        return frame_hash == self.previous_frame_hash
    
    def load_model(self, model_name):
        """Load YOLO model with basic retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Loading model: {model_name} (attempt {attempt + 1})")
                self.model = YOLO(model_name)
                self.logger.info("Model loaded successfully")
                return
            except Exception as e:
                self.logger.warning(f"Model loading failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    raise Exception(f"Failed to load model after {max_retries} attempts")
    
    def extract_frames(self, video_path, output_dir, frame_step=30):
        """Extract frames from video with basic optimization."""
        self.logger.info(f"Starting frame extraction from: {video_path}")
        
        # Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.metrics['total_frames'] = total_frames
        
        self.logger.info(f"Video info: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        saved_count = 0
        
        # Process frames
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every nth frame
                if frame_count % frame_step == 0:
                    frame_hash = self.calculate_frame_hash(frame)
                    
                    # Skip similar frames if enabled
                    if self.is_similar_frame(frame_hash):
                        self.metrics['skipped_frames'] += 1
                    else:
                        # Save frame
                        filename = f"frame_{saved_count:05d}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, frame)
                        saved_count += 1
                        self.metrics['processed_frames'] += 1
                        self.metrics['extracted_images'] += 1
                        self.previous_frame_hash = frame_hash
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Calculate frame drop ratio
        total_candidate_frames = (total_frames // frame_step) + 1
        self.metrics['frame_drop_ratio'] = self.metrics['skipped_frames'] / max(total_candidate_frames, 1)
        
        self.logger.info(f"Extracted {saved_count} frames, skipped {self.metrics['skipped_frames']} similar frames")
        self.logger.info(f"Frame drop ratio: {self.metrics['frame_drop_ratio']:.2%}")
        return saved_count
    
    def process_frames_batch(self, image_files, frames_dir):
        """Process a batch of frames for object detection."""
        results = []
        
        for image_file in image_files:
            try:
                image_path = os.path.join(frames_dir, image_file)
                
                # Run detection
                detections = self.model(image_path)[0]
                height, width = detections.orig_shape
                
                # Process detections
                frame_detections = []
                if detections.boxes is not None:
                    for box in detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf, class_id = box
                        if conf >= 0.5:  # Basic confidence threshold
                            class_name = self.model.names[int(class_id)]
                            frame_detections.append({
                                'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'confidence': conf,
                                'class_id': int(class_id),
                                'class_name': class_name
                            })
                            
                            # Track class distribution
                            if class_name not in self.metrics['class_distribution']:
                                self.metrics['class_distribution'][class_name] = 0
                            self.metrics['class_distribution'][class_name] += 1
                
                # Track detections per frame
                self.metrics['detections_per_frame'].append(len(frame_detections))
                
                results.append({
                    'filename': image_file,
                    'width': width,
                    'height': height,
                    'detections': frame_detections
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return results
    
    def detect_objects(self, frames_dir, output_file):
        """Detect objects in frames with simple parallel processing."""
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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
        """Convert detection results to COCO format."""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Track categories
        category_map = {}
        next_category_id = 1
        next_image_id = 1
        next_annotation_id = 1
        
        for result in results:
            # Add image info
            coco_data["images"].append({
                "id": next_image_id,
                "file_name": result['filename'],
                "width": result['width'],
                "height": result['height']
            })
            
            # Add detections
            for detection in result['detections']:
                class_name = detection['class_name']
                
                # Add category if new
                if class_name not in category_map:
                    category_map[class_name] = next_category_id
                    coco_data["categories"].append({
                        "id": next_category_id,
                        "name": class_name
                    })
                    next_category_id += 1
                
                # Add annotation
                bbox = detection['bbox']
                coco_data["annotations"].append({
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": category_map[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "confidence": detection['confidence']
                })
                next_annotation_id += 1
                self.metrics['total_detections'] += 1
            
            next_image_id += 1
        
        return coco_data
    
    def cleanup(self, temp_dir):
        """Basic cleanup of temporary files."""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def validate_outputs(self, output_dir, temp_frames_dir):
        """Simple validation - check that images were extracted and COCO file exists and is valid."""
        self.logger.info("🔍 Validating outputs...")
        
        # Check if any frame images were extracted
        if os.path.exists(temp_frames_dir):
            frame_files = [f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')]
            if not frame_files:
                self.logger.error("❌ No frame images found - frame extraction may have failed!")
                return False
            self.logger.info(f"✅ Found {len(frame_files)} extracted frames")
        else:
            self.logger.error("❌ Frame directory not found!")
            return False
        
        # Check if detections.json exists
        detections_file = os.path.join(output_dir, "detections.json")
        if not os.path.exists(detections_file):
            self.logger.error("❌ detections.json file not found!")
            return False
        
        try:
            # Check if JSON is valid and has expected structure
            with open(detections_file, 'r') as f:
                data = json.load(f)
            
            # Basic COCO format validation
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"❌ Missing required key '{key}' in COCO format")
                    return False
                elif not isinstance(data[key], list):
                    self.logger.error(f"❌ Key '{key}' should be a list")
                    return False
            
            # Basic counts
            num_images = len(data['images'])
            num_annotations = len(data['annotations'])
            
            self.logger.info(f"✅ Valid COCO file: {num_images} images, {num_annotations} detections")
            return True
        
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Validation error: {e}")
            return False
    
    def log_metrics(self):
        """Log comprehensive processing metrics."""
        duration = datetime.now() - self.metrics['start_time']
        
        self.logger.info("=== Processing Summary ===")
        self.logger.info(f"Total frames: {self.metrics['total_frames']}")
        self.logger.info(f"Extracted images: {self.metrics['extracted_images']}")
        self.logger.info(f"Skipped frames: {self.metrics['skipped_frames']}")
        self.logger.info(f"Frame drop ratio: {self.metrics['frame_drop_ratio']:.1%}")
        self.logger.info(f"Total detections: {self.metrics['total_detections']}")
        
        if self.metrics['detections_per_frame']:
            avg_detections = sum(self.metrics['detections_per_frame']) / len(self.metrics['detections_per_frame'])
            max_detections = max(self.metrics['detections_per_frame'])
            self.logger.info(f"Avg detections per frame: {avg_detections:.1f}")
            self.logger.info(f"Max detections per frame: {max_detections}")
        
        self.logger.info(f"Processing time: {duration}")
        
        # Log class distribution
        if self.metrics['class_distribution']:
            self.logger.info("Class distribution:")
            sorted_classes = sorted(self.metrics['class_distribution'].items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:5]:  # Top 5 classes
                self.logger.info(f"  - {class_name}: {count}")
        
        self.logger.info("=== End Summary ===")
    
    def generate_report(self, output_dir, video_path):
        """Generate HTML and Markdown reports with processing statistics."""
        self.logger.info("📊 Generating processing report...")
        
        # Calculate processing time
        end_time = datetime.now()
        total_time = (end_time - self.metrics['start_time']).total_seconds()
        self.metrics['stage_times']['total_processing'] = total_time
        
        # Calculate statistics
        avg_detections = sum(self.metrics['detections_per_frame']) / max(len(self.metrics['detections_per_frame']), 1)
        max_detections = max(self.metrics['detections_per_frame']) if self.metrics['detections_per_frame'] else 0
        
        # Report data
        report_data = {
            'video_path': video_path,
            'client_id': self.client_id,
            'processing_time': f"{total_time:.1f}s",
            'total_frames': self.metrics['total_frames'],
            'extracted_images': self.metrics['extracted_images'],
            'skipped_frames': self.metrics['skipped_frames'],
            'frame_drop_ratio': f"{self.metrics['frame_drop_ratio']:.1%}",
            'total_detections': self.metrics['total_detections'],
            'avg_detections_per_frame': f"{avg_detections:.1f}",
            'max_detections_per_frame': max_detections,
            'class_distribution': self.metrics['class_distribution'],
            'stage_times': self.metrics['stage_times']
        }
        
        # Generate HTML report
        html_report = self._generate_html_report(report_data)
        html_path = os.path.join(output_dir, "processing_report.html")
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        # Generate Markdown report
        md_report = self._generate_markdown_report(report_data)
        md_path = os.path.join(output_dir, "processing_report.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        self.logger.info(f"📄 Reports generated: {html_path}, {md_path}")
        return html_path, md_path
    
    def _generate_html_report(self, data):
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Report - {data['client_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; color: #333; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Processing Report</h1>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{data['total_frames']}</div>
                <div class="metric-label">Total Video Frames</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['extracted_images']}</div>
                <div class="metric-label">Images Extracted</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['total_detections']}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['avg_detections_per_frame']}</div>
                <div class="metric-label">Avg Detections/Frame</div>
            </div>
        </div>
        
        <h2>📋 Processing Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Video Path</td><td>{data['video_path']}</td></tr>
            <tr><td>Client ID</td><td>{data['client_id']}</td></tr>
            <tr><td>Processing Time</td><td>{data['processing_time']}</td></tr>
            <tr><td>Frame Drop Ratio</td><td>{data['frame_drop_ratio']}</td></tr>
            <tr><td>Skipped Frames</td><td>{data['skipped_frames']}</td></tr>
            <tr><td>Max Detections/Frame</td><td>{data['max_detections_per_frame']}</td></tr>
        </table>
        
        <h2>🏷️ Class Distribution</h2>
        <table>
            <tr><th>Object Class</th><th>Detection Count</th><th>Percentage</th></tr>"""
        
        total_detections = sum(data['class_distribution'].values())
        for class_name, count in sorted(data['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            html += f"<tr><td>{class_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html += f"""
        </table>
        
        <h2>⏱️ Processing Stage Times</h2>
        <table>
            <tr><th>Stage</th><th>Time (seconds)</th><th>Percentage</th></tr>
            <tr><td>Frame Extraction</td><td>{data['stage_times']['frame_extraction']:.1f}s</td><td>{(data['stage_times']['frame_extraction']/data['stage_times']['total_processing']*100):.1f}%</td></tr>
            <tr><td>Object Detection</td><td>{data['stage_times']['object_detection']:.1f}s</td><td>{(data['stage_times']['object_detection']/data['stage_times']['total_processing']*100):.1f}%</td></tr>
            <tr><td>Output Validation</td><td>{data['stage_times']['output_validation']:.1f}s</td><td>{(data['stage_times']['output_validation']/data['stage_times']['total_processing']*100):.1f}%</td></tr>
            <tr><td><strong>Total Processing</strong></td><td><strong>{data['stage_times']['total_processing']:.1f}s</strong></td><td><strong>100%</strong></td></tr>
        </table>
        
        <p style="color: #666; font-size: 12px; margin-top: 30px;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Simple Video Processing Pipeline
        </p>
    </div>
</body>
</html>"""
        return html
    
    def _generate_markdown_report(self, data):
        """Generate Markdown report."""
        md = f"""# 🎬 Video Processing Report

**Client:** {data['client_id']}  
**Video:** {data['video_path']}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Total Video Frames** | {data['total_frames']} |
| **Images Extracted** | {data['extracted_images']} |
| **Total Detections** | {data['total_detections']} |
| **Avg Detections/Frame** | {data['avg_detections_per_frame']} |
| **Frame Drop Ratio** | {data['frame_drop_ratio']} |
| **Processing Time** | {data['processing_time']} |

## 🏷️ Object Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|"""
        
        total_detections = sum(data['class_distribution'].values())
        for class_name, count in sorted(data['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            md += f"\n| {class_name} | {count} | {percentage:.1f}% |"
        
        md += f"""

## ⏱️ Processing Stage Breakdown

| Stage | Time | Percentage |
|-------|------|------------|
| Frame Extraction | {data['stage_times']['frame_extraction']:.1f}s | {(data['stage_times']['frame_extraction']/data['stage_times']['total_processing']*100):.1f}% |
| Object Detection | {data['stage_times']['object_detection']:.1f}s | {(data['stage_times']['object_detection']/data['stage_times']['total_processing']*100):.1f}% |
| Output Validation | {data['stage_times']['output_validation']:.1f}s | {(data['stage_times']['output_validation']/data['stage_times']['total_processing']*100):.1f}% |
| **Total** | **{data['stage_times']['total_processing']:.1f}s** | **100%** |

## 📋 Processing Details

- **Skipped Frames:** {data['skipped_frames']}
- **Max Detections in Single Frame:** {data['max_detections_per_frame']}
- **Frame Drop Ratio:** {data['frame_drop_ratio']} (lower is better)

---
*Generated by Simple Video Processing Pipeline*
"""
        return md

    def process_video(self, video_path, output_dir, model_name="yolov8n.pt", frame_step=30):
        """Main processing function - simplified workflow with comprehensive tracking."""
        try:
            self.logger.info(f"Starting video processing for client: {self.client_id}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create temporary directory for frames
            temp_frames_dir = os.path.join(output_dir, "temp_frames")
            
            # Step 1: Load model
            self.load_model(model_name)
            
            # Step 2: Extract frames (with timing)
            frame_start = time.time()
            num_frames = self.extract_frames(video_path, temp_frames_dir, frame_step)
            self.metrics['stage_times']['frame_extraction'] = time.time() - frame_start
            
            if num_frames == 0:
                raise Exception("No frames extracted from video")
            
            # Step 3: Detect objects (with timing)
            detection_start = time.time()
            output_file = os.path.join(output_dir, "detections.json")
            self.detect_objects(temp_frames_dir, output_file)
            self.metrics['stage_times']['object_detection'] = time.time() - detection_start
            
            # Step 4: Validate outputs (with timing)
            validation_start = time.time()
            validation_passed = self.validate_outputs(output_dir, temp_frames_dir)
            self.metrics['stage_times']['output_validation'] = time.time() - validation_start
            
            # Step 5: Generate reports
            self.generate_report(output_dir, video_path)
            
            # Step 6: Log results
            self.log_metrics()
            
            if validation_passed:
                self.logger.info("🎉 Video processing completed successfully!")
            else:
                self.logger.warning("⚠️  Processing completed but output validation failed")
                self.logger.info("💡 Check the logs above for specific validation issues")
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

        finally:
            # Always Cleanup
            self.cleanup(temp_frames_dir)


def main():
    """Main entry point with simplified argument parsing."""
    parser = argparse.ArgumentParser(description="Simple Video Processing Pipeline")
    
    # Required arguments
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_dir", help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name (default: yolov8n.pt)")
    parser.add_argument("--frame-step", type=int, default=30, help="Extract every N frames (default: 30)")
    parser.add_argument("--client-id", default="default", help="Client identifier (default: default)")
    parser.add_argument("--max-workers", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--disable-frame-skipping", action="store_true", help="Disable similar frame skipping")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SimpleVideoProcessor(
        client_id=args.client_id,
        max_workers=args.max_workers,
        skip_similar_frames=not args.disable_frame_skipping
    )
    
    try:
        # Process video
        processor.process_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            model_name=args.model,
            frame_step=args.frame_step
        )
        
        print("✅ Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 