#!/bin/bash

echo "========================================="
echo "Building Breast Cancer Prediction System"
echo "========================================="

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create model directory if it doesn't exist
echo "Creating model directory..."
mkdir -p model

# Train the model
echo "Training the machine learning model..."
python model/model_building.py

# Verify model files were created
echo "Verifying model files..."
if [ -f "model/breast_cancer_model.pkl" ]; then
    echo "✓ breast_cancer_model.pkl created successfully"
else
    echo "✗ ERROR: breast_cancer_model.pkl not found"
    exit 1
fi

if [ -f "model/scaler.pkl" ]; then
    echo "✓ scaler.pkl created successfully"
else
    echo "✗ ERROR: scaler.pkl not found"
    exit 1
fi

if [ -f "model/feature_names.pkl" ]; then
    echo "✓ feature_names.pkl created successfully"
else
    echo "✗ ERROR: feature_names.pkl not found"
    exit 1
fi

echo "========================================="
echo "Build completed successfully!"
echo "========================================="
