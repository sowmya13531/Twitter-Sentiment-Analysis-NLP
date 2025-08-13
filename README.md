# 🐦 Twitter Sentiment Analysis NLP

## 📌 Overview

This project is a Twitter Sentiment Analysis tool built with Natural Language Processing (NLP) and Machine Learning.
It predicts whether a given tweet expresses Positive or Negative sentiment.
The app is deployed using Streamlit for a simple and interactive user interface.

## 🚀 Features

- Real-time sentiment prediction from user-input tweets.

- Text preprocessing: tokenization, stopword removal, punctuation removal, and lowercasing.

- Machine Learning model trained on a labeled Twitter dataset.(Sentiment 140 Dataset)

- Interactive web app built with Streamlit.

## 📂 Project Structure

- twitter-sentiment-analysis/
- │
- ├── app.py               # Main - Streamlit application
- ├── model.pkl            #Trained Machine Learning model
- ├── vectorizer.pkl       # TFIDF Vectorizer for feature extraction
- ├── requirements.txt     # Python dependencies
- ├── README.md            # Project documentation
- └── dataset.csv          # Twitter sentiment dataset (if included)

## 🧠 How It Works

- 1. Data Collection

Dataset contains tweets labeled as Positive, Negative.
Public datasets from Kaggle/Twitter sentiment datasets were used.

- 2. Data Preprocessing
Steps applied to each tweet:
1. Lowercasing – Convert text to lowercase.
2. Removing URLs, mentions, and hashtags – Clean unnecessary elements.
3. Removing punctuation and numbers.
4. Removing stopwords – Words like "is", "the", "and" are removed.
5. (Optional) Stemming/Lemmatization – Reduce words to root form.
  
- 3. Feature Extraction
TF-IDF Vectorizer converts cleaned text into numerical form for the ML model.

- 4. Model Training
Logistic Regression model trained on the vectorized dataset.
Model saved as model (2).pkl using Pickle.

- 5. Deployment with Streamlit
Users can enter a tweet in the web app.
The model predicts and displays sentiment instantly.


## 💻 Installation & Usage

1. Clone the Repository
```
git clone https://github.com/your-username/Twitter-Sentiment-Analysis-NLP-.git
cd Twitter-Sentiment-Analysis-NLP-
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run the App
bash
streamlit run app.py
## Present my app locally deployed in streamlit 
http://localhost:8903

## 📊 Example Output
### Tweet	Predicted Sentiment

- "I love this new phone!"	Positive
- "This is the worst service ever!"	Negative
- "The weather is okay today."	Neutral

## 🛠️ Technologies Used

- Python 3.x
- Pandas – Data manipulation
- Scikit-learn – Machine learning models
- NLTK – Text preprocessing
- Streamlit – Web app deployment

## 📌 Future Improvements

- Add support for real-time Twitter scraping using Tweepy or snscrape.
- Integrate deep learning models like BERT for better accuracy.
- Display word clouds and sentiment distribution charts in the app.
