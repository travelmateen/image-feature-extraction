---
title: Feature Search
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: mit
tags:
  - computer-vision
  - image-search
  - feature-matching
  - similarity-search
  - opencv
---

# Feature Search - AI Image Matcher

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mateenahmed/feature-search)

An advanced image feature matching application that uses computer vision techniques (ORB/SIFT) to find similar images based on visual features. Draw a region on an image or select the entire image to search for matches in your image database.

## Features

- 🎯 **Region-based Search**: Draw custom bounding boxes to search specific regions
- 🔍 **Advanced Feature Matching**: Uses ORB (fast) and SIFT (high accuracy) algorithms
- 🎥 **Real-time Visualization**: Watch the matching process in real-time
- 📊 **Match Scoring**: Get detailed match scores and visualizations
- 🖼️ **Interactive UI**: Modern, responsive interface with drag-and-drop support

## How It Works

1. **Upload an Image**: Drag and drop or click to upload an image
2. **Select Region**: Click anywhere to select the entire image, or draw a bounding box around a specific region
3. **Process**: The app extracts visual features (keypoints and descriptors) from your selection
4. **Match**: Compares features against all images in the database
5. **Results**: Displays the best matches with visualization and scores

## Technology Stack

- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV (ORB/SIFT feature detection)
- **Frontend**: Vanilla JavaScript, HTML5 Canvas
- **Deployment**: Docker, Hugging Face Spaces

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/mateenahmed/feature-search
cd feature-search

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will be available at `http://localhost:5000`

## Docker Deployment

```bash
# Build the Docker image
docker build -t feature-search .

# Run the container
docker run -p 7860:7860 feature-search
```

The app will be available at `http://localhost:7860`

## Usage Tips

- **Fast Mode**: Use default ORB detection for quick results
- **High Accuracy Mode**: Enable SIFT for more accurate matching (slower)
- **Disable Visualization**: Turn off real-time visualization for faster processing
- **Threshold**: Matches with scores below 8 are filtered out

## How Feature Matching Works

The app uses feature-based image matching:

1. **Feature Detection**: Identifies keypoints (corners, edges, blobs) in images
2. **Descriptor Extraction**: Creates unique numerical descriptions for each keypoint
3. **Matching**: Compares descriptors between query and database images
4. **Filtering**: Uses Lowe's ratio test to filter out poor matches
5. **Scoring**: Counts the number of good matches to rank results

## Configuration

- **Max Upload Size**: 16MB
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Match Threshold**: 8 feature matches minimum
- **Port**: 7860 (Hugging Face Spaces default)

## Credits

Powered by [Techtics AI](https://techtics.ai)

## License

MIT License
