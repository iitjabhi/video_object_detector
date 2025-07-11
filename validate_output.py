#!/usr/bin/env python3
"""
Simple Output Validator
Validates video processing pipeline outputs to help users verify their results.
"""

import os
import json
import argparse
import sys


def validate_coco_file(file_path):
    """Validate a COCO format JSON file."""
    print(f"🔍 Validating COCO file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check required keys
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing required key: '{key}'")
                return False
            if not isinstance(data[key], list):
                print(f"❌ Key '{key}' should be a list, got {type(data[key])}")
                return False
        
        # Get counts
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        num_categories = len(data['categories'])
        
        print("✅ COCO format is valid!")
        print(f"   📸 Images: {num_images}")
        print(f"   🏷️  Annotations: {num_annotations}")
        print(f"   📂 Categories: {num_categories}")
        
        # File size
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   📄 File size: {file_size:.1f} KB")
        
        # Detailed validation
        print("\n📋 Detailed validation:")
        
        # Check images structure
        if num_images > 0:
            sample_image = data['images'][0]
            required_image_keys = ['id', 'file_name', 'width', 'height']
            missing_keys = [key for key in required_image_keys if key not in sample_image]
            if missing_keys:
                print(f"⚠️  Sample image missing keys: {missing_keys}")
            else:
                print("✅ Image entries have correct structure")
        
        # Check annotations structure
        if num_annotations > 0:
            sample_annotation = data['annotations'][0]
            required_ann_keys = ['id', 'image_id', 'category_id', 'bbox']
            missing_keys = [key for key in required_ann_keys if key not in sample_annotation]
            if missing_keys:
                print(f"⚠️  Sample annotation missing keys: {missing_keys}")
            else:
                print("✅ Annotation entries have correct structure")
            
            # Check bbox format
            if 'bbox' in sample_annotation:
                bbox = sample_annotation['bbox']
                if isinstance(bbox, list) and len(bbox) == 4:
                    print("✅ Bounding box format is correct [x, y, width, height]")
                else:
                    print(f"⚠️  Incorrect bbox format: {bbox}")
        
        # Check categories structure
        if num_categories > 0:
            sample_category = data['categories'][0]
            required_cat_keys = ['id', 'name']
            missing_keys = [key for key in required_cat_keys if key not in sample_category]
            if missing_keys:
                print(f"⚠️  Sample category missing keys: {missing_keys}")
            else:
                print("✅ Category entries have correct structure")
        
        # Summary statistics
        print(f"\n📊 Statistics:")
        if num_images > 0:
            avg_detections_per_image = num_annotations / num_images
            print(f"   🎯 Average detections per image: {avg_detections_per_image:.1f}")
        
        # Category breakdown
        if num_categories > 0 and num_annotations > 0:
            category_counts = {}
            for ann in data['annotations']:
                cat_id = ann.get('category_id')
                if cat_id:
                    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            
            print("   📈 Detections by category:")
            for cat in data['categories']:
                cat_id = cat['id']
                cat_name = cat['name']
                count = category_counts.get(cat_id, 0)
                print(f"      {cat_name}: {count}")
        
        # Warnings for common issues
        print(f"\n⚠️  Warnings:")
        if num_images == 0:
            print("   📸 No images found - check frame extraction")
        if num_annotations == 0:
            print("   🏷️  No objects detected - try lowering confidence threshold or different model")
        if num_categories == 0:
            print("   📂 No categories found - check YOLO model")
        if file_size < 1:
            print("   📄 Very small file size - processing may have failed")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def validate_output_directory(output_dir):
    """Validate an entire output directory."""
    print(f"🔍 Validating output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return False
    
    # Check for detections.json
    detections_file = os.path.join(output_dir, "detections.json")
    
    validation_passed = validate_coco_file(detections_file)
    
    # Check for other files
    files_in_dir = os.listdir(output_dir)
    print(f"\n📁 Files in output directory: {len(files_in_dir)}")
    for file in sorted(files_in_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"   📄 {file} ({size_kb:.1f} KB)")
    
    return validation_passed


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate video processing pipeline outputs")
    parser.add_argument("path", help="Path to output directory or detections.json file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed validation info")
    
    args = parser.parse_args()
    
    print("🔍 Simple Video Pipeline Output Validator")
    print("=" * 50)
    
    if os.path.isfile(args.path):
        # Validate single file
        if args.path.endswith('.json'):
            success = validate_coco_file(args.path)
        else:
            print(f"❌ Unsupported file type: {args.path}")
            success = False
    elif os.path.isdir(args.path):
        # Validate directory
        success = validate_output_directory(args.path)
    else:
        print(f"❌ Path not found: {args.path}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Validation completed successfully!")
        print("💡 Your video processing output looks good!")
        return 0
    else:
        print("❌ Validation failed!")
        print("💡 Check the error messages above and review your processing")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 