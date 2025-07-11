#!/usr/bin/env python3
"""
CI Output Validation Script
Validates that the video processing pipeline produces expected outputs.
"""

import json
import os
import sys
import re
from pathlib import Path


def load_json(file_path):
    """Load JSON file safely."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def validate_coco_structure(coco_data):
    """Validate COCO format structure."""
    required_keys = ['images', 'annotations', 'categories']
    
    for key in required_keys:
        if key not in coco_data:
            print(f"Missing required key: {key}")
            return False
        if not isinstance(coco_data[key], list):
            print(f"Key '{key}' should be a list")
            return False
    
    print("COCO structure is valid")
    return True


def validate_output_directory(output_dir, expected_structure):
    """Validate output directory against expected structure."""
    print(f"🔍 Validating output directory: {output_dir}")
    
    for required_file in expected_structure['required_files']:
        file_path = os.path.join(output_dir, required_file)
        if not os.path.exists(file_path):
            print(f"Missing required file: {required_file}")
            return False
        print(f"Found required file: {required_file}")
    
    coco_path = os.path.join(output_dir, 'detections.json')
    coco_data = load_json(coco_path)
    if not coco_data:
        return False
    
    if not validate_coco_structure(coco_data):
        return False
    
    print("All validation checks passed!")
    return True


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_ci_output.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Load expected structure
    expected_structure_path = os.path.join(
        os.path.dirname(__file__), 
        'golden_output', 
        'expected_structure.json'
    )
    
    expected_structure = load_json(expected_structure_path)
    if not expected_structure:
        print("Could not load expected structure")
        sys.exit(1)
    
    # Validate output
    if validate_output_directory(output_dir, expected_structure):
        print("CI validation successful!")
        sys.exit(0)
    else:
        print("CI validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 