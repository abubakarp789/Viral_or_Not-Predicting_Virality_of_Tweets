# Tweet Virality Predictor

A machine learning application that analyzes tweet content and metadata to predict the likelihood of a tweet going viral.

![Tweet Virality Predictor](https://github.com/abubakarp789/Viral_or_Not-Predicting_Virality_of_Tweets/blob/main/Images/app_screenshot.png)

## Overview

This project implements a machine learning model to predict whether a tweet will go viral based on various features extracted from the tweet's content and metadata. The prediction system is wrapped in a user-friendly Streamlit web application that allows users to input tweet text and related metadata to get virality predictions.

## Features

- **Tweet Content Analysis**: Extracts key features from tweet text including word count, hashtag usage, URL presence, and mentions
- **Sentiment Analysis**: Uses TextBlob to analyze sentiment and subjectivity of tweets
- **Time-Based Features**: Considers posting time (hour of day, day of week) in virality predictions
- **Keyword Detection**: Identifies presence of specific keywords that may impact engagement
- **Interactive UI**: User-friendly interface for entering tweets and viewing predictions
- **Custom Model Training**: Allows users to upload their own datasets to train custom prediction models
- **Visualization**: Displays feature importance and tweet statistics for better understanding of predictions

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/abubakarp789/Viral_or_Not-Predicting_Virality_of_Tweets.git
   cd Viral_or_Not–Predicting_Virality_of_Tweets
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter a tweet in the text area, select posting time parameters, and click "Predict Virality"

4. View the prediction results and tweet analysis

### Custom Model Training

To train a custom model:

1. Prepare a CSV dataset with at minimum the following columns:
   - `text`: The tweet content
   - Either `engagement` (retweet_count + favorite_count) or `retweet_count`

2. Upload the dataset using the file uploader in the sidebar

3. Click "Train Model with Uploaded Data"

4. Once training is complete, the new model will be used for predictions

## Dataset

The default model is trained on an airline tweets dataset containing:
- Tweet text
- Sentiment information
- Engagement metrics (retweet counts)
- Timestamps

For optimal results when training custom models, ensure your dataset includes similar fields.

## Model Details

The prediction system uses a Random Forest Classifier with the following features:
- Text-based features (word count, hashtag count, tweet length, etc.)
- Sentiment and subjectivity scores
- Temporal features (hour, day, time period)
- Presence of specific keywords

The model considers a tweet "viral" if it falls within the top 10% of engagement metrics in the training dataset.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

See `requirements.txt` for detailed version information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original dataset used for default model training was sourced from [Kaggle]('https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment')
- Thanks to the Streamlit team for making interactive data applications easier to build