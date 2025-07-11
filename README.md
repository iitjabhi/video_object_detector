# YOLO Video Processing Pipeline

![CI](https://github.com/iitjabhi/video_object_detector/workflows/Video%20Processing%20Pipeline%20CI/badge.svg)

A video processing pipeline that extracts frames from videos and performs object detection using YOLO.

Includes CI/CD pipeline with automated testing, dependency management, and golden reference validation.

## Features

- **Simple Setup**: Easy to understand and modify
- **Basic Parallel Processing**: Uses 2 workers by default
- **Frame Similarity Detection**: Skips identical frames to save processing time
- **Docker Support**: Easy containerized deployment
- **COCO Format Output**: Standard object detection format
- **Data Observability**: Comprehensive metrics tracking and reporting
- **Markdown Reports**: Detailed reports with stats and breakdowns
- **CI/CD Pipeline**: Automated testing, security scans, and performance benchmarks
- **Dependency Management**: Pinned versions for reproducible builds

## Quick Start

### 1. Build the Container
```bash
chmod +x build.sh
./build.sh
```

### 2. Add Your Video
Put your video file in the `input/` directory:
```bash
cp your_video.mp4 input/
```

### 3. Process the Video
```bash
chmod +x run.sh
./run.sh your_video.mp4
```

Results will be saved in the `output/` directory as `detections.json`.

## Manual Usage

### Using Docker Compose
```bash
# Show help
docker-compose run video-processor --help

# Process a video
docker-compose run video-processor /app/input/your_video.mp4 /app/output
```

### Using Docker Directly
```bash
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output simple-video-pipeline:latest /app/input/your_video.mp4 /app/output
```

### Running Locally (without Docker)
```bash
pip install -r requirements.txt
python video_pipeline.py input/your_video.mp4 output/
```

### Running Tests
```bash
# Install test dependencies
pip install -r requirements.txt

# Run tests
pytest test_video_pipeline.py -v
```

## Command Line Options

```bash
python video_pipeline.py <video_path> <output_dir> [options]

Options:
  --model MODEL              YOLO model (default: yolov8n.pt)
  --frame-step N             Extract every N frames (default: 30)
  --client-id ID             Client identifier (default: default)
  --max-workers N            Number of parallel workers (default: 2)
  --disable-frame-skipping   Don't skip similar frames
```

## Examples

```bash
# Basic usage
./run.sh my_video.mp4

# Use larger model for better accuracy
./run.sh my_video.mp4 --model yolov8s.pt

# Extract more frames (every 15 frames instead of 30)
./run.sh my_video.mp4 --frame-step 15

# Use more workers for faster processing
./run.sh my_video.mp4 --max-workers 4

# Process for specific client
./run.sh my_video.mp4 --client-id client123
```

## Output

The pipeline creates:
- `output/detections.json` - Object detection results in COCO format
- `output/processing_report.md` - Comprehensive report with stats and analysis
- `logs/<client_id>/pipeline.log` - Processing logs

### Output Validation

The pipeline automatically validates outputs after processing. You can also run validation manually:

```bash
# Validate output directory
python validate_output.py output/

# Validate specific JSON file
python validate_output.py output/detections.json

# Run from Docker
docker run -v $(pwd)/output:/app/output simple-video-pipeline:latest python validate_output.py /app/output
```

The validator checks:
- ✅ COCO format structure
- ✅ Required keys and data types
- ✅ File existence and size
- ✅ Detection statistics
- ⚠️  Common issues and warnings

## Data Observability & Reporting

The pipeline automatically tracks comprehensive metrics and generates detailed reports:

### 📊 Metrics Tracked
- **Images extracted** vs total video frames
- **Detection counts** per frame and overall
- **Frame drop ratio** (skipped similar/duplicate frames)
- **Class distribution** of detected objects
- **Processing time breakdown** by pipeline stage
- **Performance statistics** (avg/max detections per frame)

### 📄 Generated Report
After processing, find this report in your output directory:

**Markdown Report (`processing_report.md`)**
- Comprehensive text-based report with all statistics
- Easy to read and include in project documentation
- Version control friendly
- Contains all metrics, class distribution, and timing analysis

### 📈 Report Contents
- **Key metrics overview** with processing statistics
- **Dataset statistics** (frame counts, detection totals)  
- **Object class distribution** with percentages
- **Processing stage timing** breakdown
- **Performance analysis** and frame drop statistics

## Project Structure

```
simple_video_pipeline/
├── video_pipeline.py      # Main processing script
├── validate_output.py     # Output validation script
├── test_video_pipeline.py # Unit tests
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker compose configuration
├── build.sh             # Build script
├── run.sh               # Easy run script
├── input/               # Put video files here
├── output/              # Results saved here
└── logs/                # Log files
```

## How It Works

1. **Frame Extraction**: Extracts every Nth frame from the video
2. **Similarity Check**: Skips frames that are identical to previous ones
3. **Object Detection**: Runs YOLO model on each frame
4. **Parallel Processing**: Processes frames in batches using multiple workers
5. **Output Generation**: Saves results in COCO format

## CI/CD Pipeline

This project includes a comprehensive GitHub Actions workflow that provides:

### 🧪 **Automated Testing**
- **Unit Tests**: 7 comprehensive tests covering all functionality
- **Integration Tests**: Full pipeline testing with output validation
- **Environment Validation**: Dependency compatibility checks
- **Docker Testing**: Container build and execution verification

### 📋 **CI Documentation**
- **[CI Guide](FRAME_BASED_CI.md)**: Complete CI/CD workflow documentation
- **Golden Reference Data**: Automated output validation against known good results

### 🚀 **Getting Started with CI**
```bash
# Validate your environment locally
python validate_environment.py

# Run all tests
pytest test_video_pipeline.py -v

# Check CI requirements
pip install -r requirements.txt
```

## Learner-Friendly Features

This simplified version includes:

- **Basic Logging**: Uses simple `logging.basicConfig()` instead of complex handlers
- **Simple Classes**: One main class with clear methods
- **Basic Error Handling**: Try/catch blocks with simple error messages
- **Straightforward Parallel Processing**: Uses `ThreadPoolExecutor` with basic batching
- **Clear Code Structure**: Easy to read and modify
- **Minimal Dependencies**: Only essential packages
- **Unit Tests**: Basic tests to verify functionality

## Differences from Expert Version

Compared to the expert production version, this simplified version:

- ✅ **Simpler logging** (no custom formatters or handlers)
- ✅ **Basic parallel processing** (2 workers vs 4+, simpler batching)
- ✅ **Simple error handling** (basic try/catch vs sophisticated retry mechanisms)
- ✅ **Basic metrics** (dictionary vs dataclass with comprehensive tracking)
- ✅ **Minimal configuration** (direct parameters vs configuration classes)
- ✅ **Essential features only** (core functionality without advanced optimizations)

## Troubleshooting

### Video Not Found
Make sure your video file is in the `input/` directory.

### Docker Build Issues
Make sure Docker is running and you have sufficient permissions.

### Processing Errors
Check the log files in the `logs/` directory for detailed error information.

### Model Download
The first run may take longer as YOLO models are downloaded automatically.

### Validation Failures
If output validation fails, check for:
- Empty or corrupted video files
- Videos without detectable objects (try lowering `--confidence` if available)
- Very short videos (may not generate enough frames)
- Network issues during model download

## Testing

The project includes basic unit tests to verify core functionality:

### What's Tested
- **Frame hash calculation** - Ensures similarity detection works
- **COCO format conversion** - Validates output format is correct  
- **Output validation** - Tests file existence and format checks
- **Basic processing functions** - Core functionality verification

### Test Structure
- `TestSimpleVideoProcessor` - Main class functionality tests
- `TestStandaloneFunctions` - Individual function tests  
- Integration test - Basic end-to-end workflow test

### Running Tests in Docker
```bash
# Build the container (includes test file)
./build.sh

# Run tests in container
docker run --rm simple-video-pipeline:latest pytest test_video_pipeline.py -v
```

## Requirements

- Python 3.10+
- Docker (for containerized usage)
- 2GB+ RAM
- GPU recommended but not required

