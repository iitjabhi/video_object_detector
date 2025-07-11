#!/bin/bash

# Video Processing Pipeline Runner
# This script handles building and running the video processing container

set -e

# Function to show usage
show_usage() {
    echo "Video Processing Pipeline Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] <input_path> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  input_path           Path to input video file OR frames directory"
    echo "  output_dir           Output directory for results (default: output)"
    echo ""
    echo "Options:"
    echo "  --build              Build the Docker image first"
    echo "  --model <model>      YOLO model to use (default: yolov8n.pt)"
    echo "  --frame-step <n>     Process every nth frame (default: 30)"
    echo "  --client-id <id>     Client identifier for output organization"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 my_video.mp4                           # Video file, default output"
    echo "  $0 frames_directory/                      # Frames directory, default output"
    echo "  $0 my_video.mp4 output                    # Video file, custom output"
    echo "  $0 /path/to/video.mp4 /path/to/output     # Full paths"
    echo "  $0 --build ~/video.mp4 ./results         # With build option"
    echo "  $0 --model yolov8s.pt video.mp4 output   # Custom model"
    echo "  $0 --client-id client123 /path/to/video.mp4 /path/to/output"
}

# Function to build Docker image
build_image() {
    echo "Building Docker image..."
    docker build -t simple-video-pipeline:latest .
    echo "Docker image built successfully"
}

# Function to setup directories
setup_directories() {
    mkdir -p input logs
}

# Function to check if Docker image exists
check_image() {
    if ! docker images simple-video-pipeline:latest -q | grep -q .; then
        echo "Docker image not found. Building..."
        build_image
    fi
}

# Parse command line arguments
BUILD_FLAG=false
DOCKER_ARGS=()
VIDEO_FILE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG=true
            shift
            ;;
        --model)
            DOCKER_ARGS+=("--model" "$2")
            shift 2
            ;;
        --frame-step)
            DOCKER_ARGS+=("--frame-step" "$2")
            shift 2
            ;;
        --client-id)
            DOCKER_ARGS+=("--client-id" "$2")
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$VIDEO_FILE" ]]; then
                VIDEO_FILE="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            else
                echo "ERROR: Too many arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if both video file and output directory are provided
if [[ -z "$VIDEO_FILE" ]]; then
    echo "ERROR: No video file specified"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="output"
    echo "Using default output directory: $OUTPUT_DIR"
fi

# Setup directories
setup_directories

# Build image if requested or if it doesn't exist
if [[ "$BUILD_FLAG" == true ]]; then
    build_image
else
    check_image
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Handle full path vs filename for input video
if [[ "$VIDEO_FILE" == /* ]] || [[ "$VIDEO_FILE" == ~* ]]; then
    # Full path provided
    # Expand ~ to home directory if needed
    VIDEO_FILE=$(eval echo "$VIDEO_FILE")
    
    if [[ ! -f "$VIDEO_FILE" ]]; then
        echo "ERROR: Video file '$VIDEO_FILE' not found"
        exit 1
    fi
    
    # Get the directory and filename
    VIDEO_DIR=$(dirname "$VIDEO_FILE")
    VIDEO_FILENAME=$(basename "$VIDEO_FILE")
    
    echo "Processing video: $VIDEO_FILE"
    echo "Output directory: $OUTPUT_DIR"
    
    # Run with custom volume mount
    docker run --rm \
        -v "$VIDEO_DIR:/app/input" \
        -v "$(realpath "$OUTPUT_DIR"):/app/output" \
        -v "$(pwd)/logs:/app/logs" \
        simple-video-pipeline:latest \
        "/app/input/$VIDEO_FILENAME" \
        "/app/output" \
        "${DOCKER_ARGS[@]}"
else
    # Handle relative path or filename
    if [[ -f "$VIDEO_FILE" ]]; then
        # File exists in current directory
        VIDEO_DIR=$(dirname "$(realpath "$VIDEO_FILE")")
        VIDEO_FILENAME=$(basename "$VIDEO_FILE")
        
        echo "Processing video: $VIDEO_FILE (found in current directory)"
        echo "Output directory: $OUTPUT_DIR"
        
        # Run with custom volume mount
        docker run --rm \
            -v "$VIDEO_DIR:/app/input" \
            -v "$(realpath "$OUTPUT_DIR"):/app/output" \
            -v "$(pwd)/logs:/app/logs" \
            simple-video-pipeline:latest \
            "/app/input/$VIDEO_FILENAME" \
            "/app/output" \
            "${DOCKER_ARGS[@]}"
    elif [[ -f "input/$VIDEO_FILE" ]]; then
        # File exists in input directory
        echo "Processing video: $VIDEO_FILE (found in input/ directory)"
        echo "Output directory: $OUTPUT_DIR"
        
        # Run with default volume mount
        docker run --rm \
            -v "$(pwd)/input:/app/input" \
            -v "$(realpath "$OUTPUT_DIR"):/app/output" \
            -v "$(pwd)/logs:/app/logs" \
            simple-video-pipeline:latest \
            "/app/input/$VIDEO_FILE" \
            "/app/output" \
            "${DOCKER_ARGS[@]}"
    else
        echo "ERROR: Video file '$VIDEO_FILE' not found"
        echo "Searched in:"
        echo "  - Current directory: $(pwd)/$VIDEO_FILE"
        echo "  - Input directory: $(pwd)/input/$VIDEO_FILE"
        echo ""
        echo "Please ensure the video file exists in one of these locations, or provide a full path."
        exit 1
    fi
fi

if [[ $? -eq 0 ]]; then
    echo "Video processing completed successfully"
    echo "Results saved in: $OUTPUT_DIR"
else
    echo "ERROR: Video processing failed"
    echo "Check logs/ directory for details"
    exit 1
fi 