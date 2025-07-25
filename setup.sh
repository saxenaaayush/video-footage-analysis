#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up folders..."
mkdir -p data/raw data/frames data/detection_results
mkdir -p outputs/visualizations outputs/cropped_boxes

echo "Downloading models..."
python3 scripts/download_models.py