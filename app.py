"""
Breast Cancer Prediction System - Flask Web Application
Python Version: 3.13.7
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and feature names
MODEL_DIR = 'model'
model = joblib.load(os.path.join(MODEL_DIR, 'breast_cancer_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

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

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # For production (uncomment the line below and comment the line above)
    # app.run(host='0.0.0.0', port=5000)