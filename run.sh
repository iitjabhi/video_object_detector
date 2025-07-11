#!/bin/bash

# Simple script to run video processing

if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_file> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 my_video.mp4"
    echo "  $0 my_video.mp4 --model yolov8s.pt --frame-step 60"
    echo "  $0 my_video.mp4 --client-id client123"
    echo ""
    echo "The video file should be in the 'input/' directory"
    echo "Results will be saved in the 'output/' directory"
    exit 1
fi

VIDEO_FILE=$1
shift  # Remove first argument, keep the rest as options

# Check if video file exists
if [ ! -f "input/$VIDEO_FILE" ]; then
    echo "❌ Video file 'input/$VIDEO_FILE' not found!"
    echo "Please put your video file in the 'input/' directory"
    exit 1
fi

echo "🎬 Processing video: $VIDEO_FILE"
echo "📁 Input directory: ./input"
echo "📁 Output directory: ./output"

# Run the Docker container
docker run --rm \
    -v "$(pwd)/input:/app/input" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/logs:/app/logs" \
    simple-video-pipeline:latest \
    "/app/input/$VIDEO_FILE" \
    "/app/output" \
    "$@"

if [ $? -eq 0 ]; then
    echo "✅ Video processing completed successfully!"
    echo "📄 Check the output directory for results:"
    echo "   - detections.json (COCO format detections)"
    echo "   - processing_report.md (comprehensive analysis report)"
    echo "📋 Check the logs directory for detailed logs"
else
    echo "❌ Video processing failed!"
    echo "📋 Check the logs directory for error details"
fi 