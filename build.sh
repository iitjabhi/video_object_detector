#!/bin/bash

echo "🛠️  Building Simple Video Processing Pipeline..."

# Build the Docker image
docker build -t simple-video-pipeline:latest .

echo "✅ Docker image built successfully!"

# Create necessary directories
mkdir -p input output logs

echo "📁 Created directories:"
echo "  - input/    (put your video files here)"
echo "  - output/   (results will be saved here)"
echo "  - logs/     (log files will be saved here)"

echo ""
echo "🎯 Ready to process videos!"
echo ""
echo "Usage examples:"
echo "  # Process a video using docker-compose:"
echo "  docker-compose run video-processor /app/input/your_video.mp4 /app/output"
echo ""
echo "  # Process a video using docker directly:"
echo "  docker run -v \$(pwd)/input:/app/input -v \$(pwd)/output:/app/output simple-video-pipeline:latest /app/input/your_video.mp4 /app/output"
echo ""
echo "  # Show help:"
echo "  docker run simple-video-pipeline:latest --help" 