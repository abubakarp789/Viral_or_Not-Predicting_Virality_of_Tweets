import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
from textblob import TextBlob
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import plotly.express as px
import plotly.graph_objects as go
from predict import predict_tweet_virality

# Set page configuration
st.set_page_config(
    page_title="Tweet Virality Predictor",
    page_icon="��",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .viral {
        background-color: #e6ffe6;
        border: 2px solid #00cc00;
    }
    .not-viral {
        background-color: #ffe6e6;
        border: 2px solid #ff0000;
    }
    .feature-importance {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Define functions for feature extraction
def get_sentiment(text):
    """Extract sentiment score from text using TextBlob"""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

def get_subjectivity(text):
    """Extract subjectivity score from text using TextBlob"""
    try:
        return TextBlob(str(text)).sentiment.subjectivity
    except:
        return 0

def extract_hashtags(text):
    """Extract hashtags from text"""
    return re.findall(r'#(\w+)', str(text))

def categorize_time(hour):
    """Categorize hour into time of day"""
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

@st.cache_resource
def load_model():
    """Load the pre-trained model or train a new one if not available"""
    try:
        # Try to load the saved model
        model = joblib.load('twitter_virality_model.joblib')
        st.sidebar.success("Loaded pre-trained model")
        return model
    except:
        st.sidebar.warning("Pre-trained model not found. Please upload a dataset for training.")
        return None

def train_model(data):
    """Train a new model using uploaded data"""
    st.sidebar.info("Training model...")
    
    # Perform feature engineering
    data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))
    data['hashtag_count'] = data['text'].apply(lambda x: len([word for word in str(x).split() if word.startswith('#')]))
    data['has_url'] = data['text'].apply(lambda x: int('http' in str(x)))
    data['tweet_length'] = data['text'].apply(lambda x: len(str(x)))
    data['is_reply'] = data['text'].apply(lambda x: int('@' in str(x)))
    data['sentiment_score'] = data['text'].apply(get_sentiment)
    data['subjectivity'] = data['text'].apply(get_subjectivity)
    
    # Check if tweet_hour and tweet_day_of_week columns exist, if not create them
    if 'tweet_hour' not in data.columns or 'tweet_day_of_week' not in data.columns:
        # If tweet_created exists, extract hour and day of week
        if 'tweet_created' in data.columns:
            data['tweet_created'] = pd.to_datetime(data['tweet_created'])
            data['tweet_hour'] = data['tweet_created'].dt.hour
            data['tweet_day_of_week'] = data['tweet_created'].dt.dayofweek
        else:
            # If not, use defaults
            data['tweet_hour'] = 12  # Noon as default
            data['tweet_day_of_week'] = 2  # Wednesday as default
    
    data['is_weekend'] = data['tweet_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['time_of_day'] = data['tweet_hour'].apply(categorize_time)
    
    # Create time of day dummies
    time_dummies = pd.get_dummies(data['time_of_day'], prefix='time')
    data = pd.concat([data, time_dummies], axis=1)
    
    # Add keyword features
    keywords = ['delay', 'cancel', 'thanks', 'help', 'great', 'service', 'sorry', 'bad', 'good', 'love']
    for keyword in keywords:
        data[f'has_{keyword}'] = data['text'].apply(lambda x: int(keyword.lower() in str(x).lower()))
    
    # Ensure viral column exists
    if 'viral' not in data.columns:
        # If engagement exists, create viral based on that
        if 'engagement' in data.columns:
            viral_threshold = data['engagement'].quantile(0.9)
            data['viral'] = (data['engagement'] > viral_threshold).astype(int)
        elif 'retweet_count' in data.columns:
            # Otherwise use retweet_count
            viral_threshold = data['retweet_count'].quantile(0.9)
            data['viral'] = (data['retweet_count'] > viral_threshold).astype(int)
        else:
            st.error("Dataset doesn't contain engagement or retweet metrics for determining virality.")
            return None
    
    # Select features for modeling
    feature_columns = [
        'word_count', 'hashtag_count', 'has_url', 'tweet_length', 'is_reply',
        'sentiment_score', 'subjectivity', 'tweet_hour', 'tweet_day_of_week', 'is_weekend'
    ]
    
    # Add time dummies
    time_dummy_columns = [col for col in data.columns if col.startswith('time_')]
    feature_columns.extend(time_dummy_columns)
    
    # Add keyword features
    keyword_columns = [col for col in data.columns if col.startswith('has_') and col != 'has_url']
    feature_columns.extend(keyword_columns)
    
    # Ensure all features are numeric
    X = data[feature_columns].select_dtypes(include=['number'])
    y = data['viral']
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'twitter_virality_model.joblib')
    
    # Return feature list and model
    st.sidebar.success("Model trained successfully!")
    return model, X.columns.tolist()

def process_tweet(tweet_text, hour, day):
    """Process a single tweet to extract features"""
    features = {}
    
    # Basic text features
    features['word_count'] = len(tweet_text.split())
    features['hashtag_count'] = len([word for word in tweet_text.split() if word.startswith('#')])
    features['has_url'] = int('http' in tweet_text)
    features['tweet_length'] = len(tweet_text)
    features['is_reply'] = int('@' in tweet_text)
    
    # Sentiment features
    features['sentiment_score'] = get_sentiment(tweet_text)
    features['subjectivity'] = get_subjectivity(tweet_text)
    
    # Time features
    features['tweet_hour'] = hour
    features['tweet_day_of_week'] = day
    features['is_weekend'] = int(day >= 5)
    
    # Time of day
    time_of_day = categorize_time(hour)
    features['time_morning'] = int(time_of_day == 'morning')
    features['time_afternoon'] = int(time_of_day == 'afternoon')
    features['time_evening'] = int(time_of_day == 'evening')
    features['time_night'] = int(time_of_day == 'night')
    
    # Keyword features
    keywords = ['delay', 'cancel', 'thanks', 'help', 'great', 'service', 'sorry', 'bad', 'good', 'love']
    for keyword in keywords:
        features[f'has_{keyword}'] = int(keyword.lower() in tweet_text.lower())
    
    return features

def predict_tweet_virality(tweet_text, hour, day, model, feature_names):
    """Predict whether a tweet will go viral"""
    features = process_tweet(tweet_text, hour, day)
    
    # Create a DataFrame with the features in the correct order
    feature_df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in feature_df.columns:
            feature_df[feature] = 0
    
    # Select only the features used by the model, in the correct order
    X = feature_df[feature_names]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability

def display_tweet_statistics(tweet_text, hour, day, features):
    """Display statistics about the tweet"""
    col1, col2 = st.columns(2)
    
    # Basic stats
    with col1:
        st.metric("Word Count", features['word_count'])
        st.metric("Character Count", features['tweet_length'])
        st.metric("Hashtag Count", features['hashtag_count'])
        st.metric("Contains URL", "Yes" if features['has_url'] == 1 else "No")
        st.metric("Is Reply", "Yes" if features['is_reply'] == 1 else "No")
    
    # Sentiment and time stats
    with col2:
        st.metric("Sentiment Score", f"{features['sentiment_score']:.2f}")
        st.metric("Subjectivity", f"{features['subjectivity']:.2f}")
        st.metric("Time of Day", categorize_time(hour))
        st.metric("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day])
        st.metric("Is Weekend", "Yes" if features['is_weekend'] == 1 else "No")
    
    # Keyword presence
    st.subheader("Keyword Analysis")
    keyword_cols = [col for col in features.keys() if col.startswith('has_') and col != 'has_url']
    
    # Filter to only keywords present in the tweet
    present_keywords = {k.replace('has_', ''): features[k] for k in keyword_cols if features[k] == 1}
    
    if present_keywords:
        st.write("Keywords found in tweet:")
        for keyword, _ in present_keywords.items():
            st.write(f"- {keyword}")
    else:
        st.write("No monitored keywords found in tweet")

def display_feature_importance(model, feature_names):
    """Display feature importance from the model"""
    if model is not None:
        st.subheader("Feature Importance")
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display top 10 features
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
        plt.title('Top 10 Features for Predicting Viral Tweets')
        plt.tight_layout()
        st.pyplot(fig)

def main():
    """Main function to run the Streamlit app"""
    st.title("🐦 Tweet Virality Predictor")
    st.write("""
    This app predicts whether a tweet is likely to go viral based on various features.
    Enter your tweet and other information below to get a prediction.
    """)
    
    # Sidebar for model management
    st.sidebar.title("Model Management")
    model = load_model()
    feature_names = None
    
    # Option to upload a dataset for training
    uploaded_file = st.sidebar.file_uploader("Upload dataset for training (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if st.sidebar.button("Train Model with Uploaded Data"):
            with st.spinner("Training model... This may take a while"):
                model_results = train_model(data)
                if model_results:
                    model, feature_names = model_results
    
    # Main interface
    tweet_text = st.text_area("Enter your tweet text:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of day (24-hour format):", 0, 23, 12)
    with col2:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = st.selectbox("Day of week:", range(len(days)), format_func=lambda x: days[x])
    
    # Predict button
    if st.button("Predict Virality"):
        if not model:
            st.error("Please upload a dataset and train a model first!")
        elif not tweet_text:
            st.error("Please enter a tweet to analyze.")
        else:
            # If feature_names is None, use the model's feature names
            if feature_names is None:
                try:
                    # Attempt to extract feature names from the model
                    feature_names = model.feature_names_in_.tolist()
                except:
                    # Default feature list if model doesn't store feature names
                    feature_names = [
                        'word_count', 'hashtag_count', 'has_url', 'tweet_length', 'is_reply',
                        'sentiment_score', 'subjectivity', 'tweet_hour', 'tweet_day_of_week', 'is_weekend',
                        'time_morning', 'time_afternoon', 'time_evening', 'time_night',
                        'has_delay', 'has_cancel', 'has_thanks', 'has_help', 'has_great',
                        'has_service', 'has_sorry', 'has_bad', 'has_good', 'has_love'
                    ]
            
            # Get features and prediction
            features = process_tweet(tweet_text, hour, day)
            prediction, probability = predict_tweet_virality(tweet_text, hour, day, model, feature_names)
            
            # Display prediction result
            st.write("## Prediction Result")
            if prediction == 1:
                st.success(f"This tweet is likely to go viral! (Probability: {probability:.2f})")
            else:
                st.warning(f"This tweet is not likely to go viral. (Probability: {probability:.2f})")
            
            # Display tweet statistics
            st.write("## Tweet Statistics")
            display_tweet_statistics(tweet_text, hour, day, features)
            
            # Display feature importance
            display_feature_importance(model, feature_names)
    
    # Information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Model")
    st.sidebar.write("""
    The tweet virality prediction model uses features including:
    - Text statistics (word count, length, hashtags)
    - Sentiment analysis
    - Time of posting
    - Presence of specific keywords
    
    The model considers a tweet "viral" if it falls in the top 10% of engagement.
    """)
    
    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️ using Streamlit")

if __name__ == "__main__":
    main()