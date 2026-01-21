"""
Breast Cancer Prediction System - Model Development
Python Version: 3.13.7
Algorithm: Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os

def load_and_prepare_data():
    """Load the Breast Cancer Wisconsin dataset and prepare it"""
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())
    
    return df, data

def preprocess_data(df):
    """Preprocess the data: handle missing values, feature selection, encoding"""
    
    # Check for missing values
    print("\n" + "="*50)
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values[missing_values > 0]}")
    if missing_values.sum() == 0:
        print("No missing values found!")
    
    # Feature Selection - Select 5 features from the recommended list
    # Selected features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
    selected_features = [
        'mean radius',      # radius_mean
        'mean texture',     # texture_mean
        'mean perimeter',   # perimeter_mean
        'mean area',        # area_mean
        'mean smoothness'   # smoothness_mean
    ]
    
    print("\n" + "="*50)
    print("Selected Features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Extract features and target
    X = df[selected_features]
    y = df['diagnosis']
    
    # Target encoding (0 = malignant, 1 = benign in original dataset)
    # We'll keep it as is: 0 = Malignant, 1 = Benign
    print("\n" + "="*50)
    print("Target Variable Encoding:")
    print("0 = Malignant")
    print("1 = Benign")
    print(f"\nClass distribution:")
    print(y.value_counts())
    
    return X, y, selected_features

def train_model(X_train, y_train):
    """Train the Logistic Regression model"""
    print("\n" + "="*50)
    print("Training Logistic Regression model...")
    
    # Initialize the model
    model = LogisticRegression(
        random_state=42,
        max_iter=10000,
        solver='lbfgs'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n" + "-"*50)
    print("Classification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    
    print("\n" + "-"*50)
    print("Confusion Matrix:")
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              Mal    Ben")
    print(f"Actual Mal  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"       Ben  [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    
    return accuracy, precision, recall, f1

def save_model(model, scaler, feature_names, model_dir='model'):
    """Save the trained model, scaler, and feature names to disk"""
    print("\n" + "="*50)
    print("Saving model to disk...")
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Save model
    model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to: {features_path}")

def test_saved_model(X_test, y_test, model_dir='model'):
    """Reload and test the saved model"""
    print("\n" + "="*50)
    print("TESTING SAVED MODEL (Without Retraining)")
    print("="*50)
    
    # Load the saved model
    model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    
    print("Model and scaler loaded successfully!")
    
    # Scale the test data
    X_test_scaled = loaded_scaler.transform(X_test)
    
    # Make predictions
    y_pred = loaded_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nReloaded Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Test with a single sample
    print("\n" + "-"*50)
    print("Testing with a single sample:")
    print("-"*50)
    sample = X_test.iloc[0:1]
    sample_scaled = loaded_scaler.transform(sample)
    prediction = loaded_model.predict(sample_scaled)
    probability = loaded_model.predict_proba(sample_scaled)
    
    print(f"Sample features:\n{sample.iloc[0]}")
    print(f"\nPrediction: {'Benign' if prediction[0] == 1 else 'Malignant'}")
    print(f"Probability - Malignant: {probability[0][0]:.2%}, Benign: {probability[0][1]:.2%}")
    print(f"Actual: {'Benign' if y_test.iloc[0] == 1 else 'Malignant'}")

def main():
    """Main function to execute the entire workflow"""
    print("="*50)
    print("BREAST CANCER PREDICTION SYSTEM")
    print("Model Development Phase")
    print("="*50)
    
    # Load and prepare data
    df, data = load_and_prepare_data()
    
    # Preprocess data
    X, y, selected_features = preprocess_data(df)
    
    # Split the data
    print("\n" + "="*50)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Feature Scaling (Mandatory for Logistic Regression)
    print("\n" + "="*50)
    print("Applying Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed!")
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Save the model
    save_model(model, scaler, selected_features)
    
    # Test the saved model
    test_saved_model(X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nNote: This system is for educational purposes only.")
    print("It must not be used as a medical diagnostic tool.")

if __name__ == "__main__":
    main()