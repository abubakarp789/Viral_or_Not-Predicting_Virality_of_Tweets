# Tweet Virality Predictor 

A machine learning application that predicts the virality of tweets using advanced classification techniques. Developed during the 3 Days Data Analysis Bootcamp at GDGoC UMT.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project analyzes tweets to predict their potential virality using various features including:
- Content analysis (word count, hashtags, URLs)
- Sentiment analysis
- Timing features (hour, day of week)
- Engagement patterns

The application provides an interactive interface for users to:
- Predict virality of new tweets
- Analyze tweet patterns
- Visualize key insights
- Understand model performance


## 🚀 Features

- **Dual Model Approach**: Utilizes both Random Forest and Logistic Regression for robust predictions
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Comprehensive Analysis**: 
  - Time-based tweet patterns
  - Sentiment analysis
  - Feature importance visualization
  - Correlation analysis
- **Real-time Predictions**: Instant virality predictions for new tweets

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/abubakarp789/Viral_or_Not-Predicting_Virality_of_Tweets.git
cd Viral_or_Not-Predicting_Virality_of_Tweets
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your dataset in the `Dataset` folder:
- Ensure it contains the required columns

5. Train the models in Notebook:
```bash
Predicting_Virality_of_Tweets.ipynb
```

6. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Model Performance

The models were evaluated using standard metrics:
- Accuracy: 85%
- Precision (Viral): 78%
- Recall (Viral): 82%
- F1-Score: 80%

## 🔍 Key Features Analysis

### 1. Content Analysis
- Word count and complexity
- Hashtag usage patterns
- URL presence and impact
- Sentiment analysis

### 2. Timing Analysis
- Optimal posting times
- Day of week patterns
- Hourly engagement trends

### 3. Engagement Patterns
- Retweet prediction
- Like count estimation
- Reply probability

## 🧩 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn
- plotly
- joblib
- textblob

## 📚 Development Context

This project was developed during the 3 Days Data Analysis Bootcamp at GDGoC UMT, focusing on:
- Machine learning model development
- Data preprocessing and feature engineering
- Interactive web application development
- Data visualization and analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abu Bakar**
- Email: abubakarp789@gmail.com
- GitHub: [Abu Bakar](https://github.com/abubakarp789)

## 🙏 Acknowledgments

- GDGoC UMT for organizing the Data Analysis Bootcamp
- Mentors and instructors for their guidance
- Open-source community for their valuable tools and libraries

---

<div align="center">
  <sub>Built with ❤️ during the GDGoC UMT Data Analysis Bootcamp</sub>
</div>