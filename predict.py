import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from datetime import datetime
import joblib
from typing import Dict, List, Union
import os

class TweetViralityPredictor:
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize the predictor with a trained model."""
        self.model_type = model_type
        self.models_dir = 'models'
        
        # Load the appropriate model
        if model_type == 'random_forest':
            self.model = joblib.load(os.path.join(self.models_dir, 'random_forest_model.joblib'))
        else:
            self.model = joblib.load(os.path.join(self.models_dir, 'logistic_regression_model.joblib'))
            
        # Load scaler and feature columns
        self.scaler = joblib.load(os.path.join(self.models_dir, 'feature_scaler.joblib'))
        self.feature_columns = joblib.load(os.path.join(self.models_dir, 'feature_columns.joblib'))
        
        self.keywords = ['delay', 'cancel', 'thanks', 'help', 'great', 'service', 'sorry', 'bad', 'good', 'love']
        
    def _extract_features(self, tweet: str, tweet_time: datetime = None) -> Dict[str, Union[int, float]]:
        """Extract features from a single tweet."""
        if tweet_time is None:
            tweet_time = datetime.now()
            
        features = {
            'word_count': len(str(tweet).split()),
            'hashtag_count': len([word for word in str(tweet).split() if word.startswith('#')]),
            'has_url': int('http' in str(tweet)),
            'tweet_length': len(str(tweet)),
            'is_reply': int('@' in str(tweet)),
            'sentiment_score': TextBlob(str(tweet)).sentiment.polarity,
            'subjectivity': TextBlob(str(tweet)).sentiment.subjectivity,
            'tweet_hour': tweet_time.hour,
            'tweet_day_of_week': tweet_time.weekday(),
            'is_weekend': int(tweet_time.weekday() >= 5)
        }
        
        # Time of day categories
        hour = tweet_time.hour
        time_categories = {
            'time_morning': int(5 <= hour < 12),
            'time_afternoon': int(12 <= hour < 17),
            'time_evening': int(17 <= hour < 22),
            'time_night': int(hour < 5 or hour >= 22)
        }
        features.update(time_categories)
        
        # Keyword features
        for keyword in self.keywords:
            features[f'has_{keyword}'] = int(keyword.lower() in str(tweet).lower())
            
        return features
    
    def predict_virality(self, tweet: str, tweet_time: datetime = None) -> Dict[str, Union[float, str]]:
        """Predict the virality of a tweet."""
        features = self._extract_features(tweet, tweet_time)
        features_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in features_df.columns:
                features_df[feature] = 0
                
        # Reorder columns to match training data
        features_df = features_df[self.feature_columns]
        
        # Ensure all features are numeric
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Fill any NaN values with 0
        features_df = features_df.fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        probability = self.model.predict_proba(features_scaled)[0][1]
        prediction = probability > 0.5
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = 'High'
        elif probability > 0.6 or probability < 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'viral': bool(prediction),
            'probability': float(probability),
            'confidence': confidence,
            'model_type': self.model_type
        }

def predict_tweet_virality(tweet: str, tweet_time: datetime = None, model_type: str = 'random_forest') -> Dict[str, Union[float, str]]:
    """Main function to predict tweet virality."""
    predictor = TweetViralityPredictor(model_type)
    return predictor.predict_virality(tweet, tweet_time) 