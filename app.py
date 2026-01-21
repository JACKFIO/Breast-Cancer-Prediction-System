"""
Breast Cancer Prediction System - Flask Web Application
Python Version: 3.13.7
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)

# Model directory
MODEL_DIR = 'model'

# Check if model files exist
def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        'breast_cancer_model.pkl',
        'scaler.pkl',
        'feature_names.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, file)):
            missing_files.append(file)
    
    return missing_files

# Check for model files
missing_files = check_model_files()

# If running locally and files are missing, show helpful error
if missing_files and not os.environ.get('RENDER'):
    print("\n" + "="*60)
    print("ERROR: Required model files not found!")
    print("="*60)
    print("\nMissing files:")
    for file in missing_files:
        print(f"  - {MODEL_DIR}/{file}")
    print("\n" + "-"*60)
    print("SOLUTION: Train the model first by running:")
    print("  python model/model_building.py")
    print("  OR use: 1_train_model.bat (Windows) / ./1_train_model.sh (Linux/Mac)")
    print("-"*60 + "\n")
    sys.exit(1)

# If on Render and files are missing, train the model
if missing_files and os.environ.get('RENDER'):
    print("\n" + "="*60)
    print("RENDER DEPLOYMENT: Training model...")
    print("="*60)
    import subprocess
    result = subprocess.run(['python', 'model/model_building.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR during training:")
        print(result.stderr)
        sys.exit(1)

# Load the trained model, scaler, and feature names
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'breast_cancer_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ Model files loaded successfully!")
except Exception as e:
    print(f"\n❌ Error loading model files: {e}")
    sys.exit(1)

# Feature information for the form
FEATURE_INFO = {
    'mean radius': {
        'min': 6.0,
        'max': 30.0,
        'step': 0.1,
        'default': 14.0,
        'description': 'Mean of distances from center to points on the perimeter'
    },
    'mean texture': {
        'min': 9.0,
        'max': 40.0,
        'step': 0.1,
        'default': 19.0,
        'description': 'Standard deviation of gray-scale values'
    },
    'mean perimeter': {
        'min': 40.0,
        'max': 200.0,
        'step': 0.1,
        'default': 92.0,
        'description': 'Mean size of the core tumor'
    },
    'mean area': {
        'min': 140.0,
        'max': 2500.0,
        'step': 1.0,
        'default': 654.0,
        'description': 'Mean area of the tumor'
    },
    'mean smoothness': {
        'min': 0.05,
        'max': 0.17,
        'step': 0.001,
        'default': 0.096,
        'description': 'Local variation in radius lengths'
    }
}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                         feature_names=feature_names,
                         feature_info=FEATURE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        # Get input values from form
        features = []
        input_values = {}
        
        for feature in feature_names:
            value = float(request.form.get(feature))
            features.append(value)
            input_values[feature] = value
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'prediction_class': int(prediction),
            'probability_malignant': float(probability[0]),
            'probability_benign': float(probability[1]),
            'confidence': float(max(probability)),
            'input_values': input_values
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/info')
def api_info():
    """Return API information"""
    return jsonify({
        'model': 'Logistic Regression',
        'features': feature_names,
        'feature_count': len(feature_names),
        'classes': ['Malignant (0)', 'Benign (1)'],
        'disclaimer': 'This system is for educational purposes only. Not for medical diagnosis.'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'features_count': len(feature_names)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎗️  BREAST CANCER PREDICTION SYSTEM")
    print("="*60)
    print(f"Model: Logistic Regression")
    print(f"Features: {len(feature_names)}")
    print(f"Server: Flask Development Server")
    print("="*60)
    print("\n🌐 Starting web application...")
    print("📍 Access the app at: http://localhost:5000")
    print("⚠️  For educational purposes only - not for medical diagnosis\n")
    
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # For production on Render.com (uncomment and comment the line above)
    # app.run(host='0.0.0.0', port=5000)
