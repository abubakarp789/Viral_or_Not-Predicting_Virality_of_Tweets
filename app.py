from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import re
from textblob import TextBlob
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load models and preprocessing objects
MODELS_DIR = 'models'
rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'))
feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.joblib'))

# Initialize and fit imputer with correct number of features
n_features = len(feature_columns)
sample_data = np.zeros((10, n_features))  # Create array with correct number of features
imputer = SimpleImputer(strategy='constant', fill_value=0)
imputer.fit(sample_data)

def extract_features(tweet_text, tweet_time):
    """Extract features from tweet text and time."""
    # Initialize features dictionary with default values
    features = {
        'word_count': 0,
        'hashtag_count': 0,
        'url_count': 0,
        'sentiment': 0,
        'hour': 0,
        'day_of_week': 0
    }
    
    try:
        # Text features
        if tweet_text:
            # Word count
            features['word_count'] = len(tweet_text.split())
            
            # Hashtag count
            features['hashtag_count'] = len(re.findall(r'#\w+', tweet_text))
            
            # URL count
            features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_text))
            
            # Sentiment analysis
            blob = TextBlob(tweet_text)
            features['sentiment'] = blob.sentiment.polarity
        
        # Time features
        if tweet_time:
            features['hour'] = tweet_time.hour
            features['day_of_week'] = tweet_time.weekday()
            
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Keep default values if extraction fails
    
    return features

def predict_virality(tweet_text, tweet_time):
    """Predict tweet virality using both models."""
    try:
        # Extract features
        features = extract_features(tweet_text, tweet_time)
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all required columns exist
        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Reorder columns to match training data
        feature_df = feature_df[feature_columns]
        
        # Scale features
        scaled_features = feature_scaler.transform(feature_df)
        
        # Handle any remaining NaN values
        scaled_features = imputer.transform(scaled_features)
        
        # Get predictions from both models
        rf_prob = rf_model.predict_proba(scaled_features)[0][1]
        lr_prob = lr_model.predict_proba(scaled_features)[0][1]
        
        # Average the probabilities
        avg_prob = (rf_prob + lr_prob) / 2
        
        # Determine if viral (threshold = 0.5)
        is_viral = 'Viral' if avg_prob >= 0.5 else 'Not Viral'
        
        return {
            'prediction': is_viral,
            'probability': avg_prob,
            'rf_probability': rf_prob,
            'lr_probability': lr_prob,
            'features': features
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'prediction': 'Error',
            'probability': 0,
            'rf_probability': 0,
            'lr_probability': 0,
            'features': {},
            'error': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            tweet_text = request.form.get('tweet', '')
            hour = int(request.form.get('hour', 0))
            day = int(request.form.get('day', 0))
            
            # Create datetime object for time features
            tweet_time = datetime(2023, 1, 1, hour, 0)  # Using arbitrary date
            
            # Get prediction
            result = predict_virality(tweet_text, tweet_time)
            
            if 'error' in result:
                return render_template('predict.html', 
                                    error=result['error'],
                                    prediction=None)
            
            return render_template('predict.html',
                                prediction=result['prediction'],
                                probability=result['probability'],
                                rf_probability=result['rf_probability'],
                                lr_probability=result['lr_probability'],
                                features=result['features'])
            
        except Exception as e:
            return render_template('predict.html', 
                                error=str(e),
                                prediction=None)
    
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)