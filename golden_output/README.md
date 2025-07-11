# Golden Reference Data for CI/CD

This directory contains reference data generated from processing `timelapse_test.MP4` with the video processing pipeline. This data serves as the baseline for continuous integration testing.

## 📊 Test Video Details
- **File:** `timelapse_test.MP4`
- **Total Frames:** 4,077
- **Duration:** 127.4 seconds
- **FPS:** 32.0
- **Resolution:** 1080x608

## 🎯 Processing Configuration
- **Frame Step:** 407 (extracts every 407th frame)
- **Expected Frames Extracted:** ~10-11
- **Client ID:** ci-test

## 📁 Generated Files

### `detections.json`
- **Format:** COCO standard format
- **Images:** 10 processed frames
- **Detections:** 3 total detections
- **Classes:** car (2), train (1)
- **Confidence Range:** 0.57 - 0.60

### `processing_report.md`
- **Processing Time:** 18.6 seconds
- **Stage Breakdown:** Frame extraction (69.3%), Detection (28.7%)
- **Performance Metrics:** All timing and accuracy data

### `expected_structure.json`
- **Validation Criteria:** Defines minimum requirements for CI
- **Thresholds:** Min/max values for detection counts, processing time
- **Required Files:** List of expected output files

### `sample_detections.json`
- **Example Output:** Shows ideal COCO format structure
- **Reference:** For comparison when model works optimally

## 🔍 CI Validation

Use the validation script:
```bash
python validate_ci_output.py golden_output
```

### Validation Checks
- ✅ Required files exist (detections.json, processing_report.md)
- ✅ Valid COCO format structure
- ✅ Minimum detection counts met
- ✅ Processing metrics extracted
- ✅ No critical errors

## 🚀 CI/CD Usage

1. **Environment Setup:** Install dependencies, fix NumPy version
2. **Pipeline Execution:** Run video processing on test video
3. **Output Validation:** Compare against this golden reference
4. **Pass/Fail Criteria:** All validation checks must pass

## 🛠 Environment Requirements

### Critical Version Compatibility
**These exact versions are tested and compatible:**

```bash
# Install all dependencies (recommended)
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy==1.26.4` (MUST be <2.0 for ultralytics compatibility)
- `opencv-python==4.8.1.78` (compatible with numpy 1.x) 
- `torch==2.2.2`
- `ultralytics==8.3.164`

### Dependency Conflicts to Avoid
- ❌ `numpy>=2.0` - Breaks ultralytics/torch compatibility
- ❌ `opencv-python>=4.10.0` - Requires numpy>=2.0
- ❌ Mixing `opencv-python` and `opencv-contrib-python`

### Environment Setup Commands
```bash
# Install all dependencies (recommended)
pip install -r requirements.txt

# Manual setup with compatibility fix (if needed)
pip install "numpy<2"
pip install opencv-python==4.8.1.78
pip install torch ultralytics tqdm pytest
```

## 📋 Expected Results

A successful CI run should produce:
- **10+ extracted frames** from 4,077 total
- **1-10 object detections** (cars, trains, etc.)
- **Valid COCO format** with proper structure
- **Processing time < 60 seconds**
- **No critical errors** in the pipeline

## 🐛 Troubleshooting

### Common CI Failures

**NumPy Compatibility Error:**
```
'Conv' object has no attribute 'bn'
Numpy is not available
```
**Solution:** Ensure `numpy<2.0` with `pip install "numpy<2"`

**Dependency Conflicts:**
```
opencv-python requires numpy>=2
ultralytics requires numpy<2
```
**Solution:** Use `opencv-python==4.8.1.78` from `requirements.txt` 