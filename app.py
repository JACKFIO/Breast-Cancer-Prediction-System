"""
Breast Cancer Prediction System - Flask Backend
This is the main application logic that loads the model and handles predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = None
scaler = None
feature_names = None

def load_model_and_scaler():
    global model, scaler, feature_names
    try:
        model = keras.models.load_model('model.h5')
        print("Model loaded successfully!")
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print("Feature names loaded successfully!")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model_and_scaler()

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return the list of features needed for prediction"""
    if feature_names is None:
        return jsonify({'error': 'Feature names not loaded'}), 500
    
    return jsonify({
        'features': list(feature_names),
        'count': len(feature_names)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features"""
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = data['features']
        
        # Validate input
        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(features)}'}), 400
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        probability = float(prediction[0][0])
        
        # Interpret the result
        is_benign = probability > 0.5
        confidence = probability if is_benign else (1 - probability)
        
        result = {
            'prediction': 'Benign' if is_benign else 'Malignant',
            'probability': probability,
            'confidence': confidence,
            'benign_probability': probability,
            'malignant_probability': 1 - probability
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-sample', methods=['POST'])
def predict_sample():
    """Predict using a predefined sample (for testing)"""
    try:
        data = request.get_json()
        sample_type = data.get('sample_type', 'benign')
        
        # Example samples (using mean values from the dataset)
        if sample_type == 'benign':
            # Typical benign tumor characteristics
            sample = [13.0, 17.0, 85.0, 500.0, 0.08, 0.05, 0.03, 0.02, 0.16, 0.06,
                     0.3, 1.0, 2.0, 25.0, 0.005, 0.015, 0.02, 0.008, 0.015, 0.002,
                     14.5, 22.0, 95.0, 650.0, 0.11, 0.15, 0.15, 0.06, 0.25, 0.08]
        else:
            # Typical malignant tumor characteristics
            sample = [18.0, 25.0, 120.0, 1000.0, 0.12, 0.15, 0.18, 0.12, 0.22, 0.08,
                     0.6, 1.5, 4.0, 60.0, 0.008, 0.04, 0.05, 0.025, 0.03, 0.005,
                     22.0, 32.0, 145.0, 1500.0, 0.15, 0.35, 0.4, 0.2, 0.35, 0.12]
        
        input_scaled = scaler.transform(np.array(sample).reshape(1, -1))
        prediction = model.predict(input_scaled, verbose=0)
        probability = float(prediction[0][0])
        
        is_benign = probability > 0.5
        confidence = probability if is_benign else (1 - probability)
        
        result = {
            'prediction': 'Benign' if is_benign else 'Malignant',
            'probability': probability,
            'confidence': confidence,
            'sample_used': sample,
            'sample_type': sample_type
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running and model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not loaded. Please run model_training.py first!")
    else:
        print("Server starting...")
        print(f"Model input shape: {model.input_shape}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)