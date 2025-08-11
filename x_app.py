import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download and load stopwords
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    for word in ['not', 'no', 'nor', 'never']:
        stop_words.discard(word)
    return stop_words

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("model (2).pkl", "rb") as mf, open("vectorizer.pkl", "rb") as vf:
        model = pickle.load(mf)
        vectorizer = pickle.load(vf)
    return model, vectorizer

# Preprocessing function
def preprocess_text(text, stop_words):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower().strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# Predict sentiment
def predict_sentiment(text, model, vectorizer, stop_words):
    processed = preprocess_text(text, stop_words)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

# Create sentiment card
def create_card(tweet_text, sentiment):
    color = "#27ae60" if sentiment == "Positive" else "#c0392b"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main function
def main():
    # Page Title
    st.set_page_config(page_title="Twitter Sentiment Analysis NLP", layout="centered")

    st.markdown(
        """
        <style>
            /* Blue background for full app */
            .stApp {
                background-color: #001f3f;
                color: white;
            }

            /* Make text boxes blue */
            div[role="textbox"] {
                background-color: #003366;
                color: white;
                border: none;
            }

            /* Blue theme for selectbox and buttons */
            .css-1cpxqw2, .st-bb, .st-c6 {
                background-color: #004080 !important;
                color: white !important;
            }

            /* Button hover effect */
            .stButton>button:hover {
                background-color: #0059b3 !important;
                color: white !important;
            }

            /* Default button color */
            .stButton>button {
                background-color: #0074D9;
                color: white;
                border-radius: 5px;
                padding: 0.5em 1em;
                border: none;
            }

            /* Info box styling */
            .stAlert {
                background-color: #003366 !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔵Twitter Sentiment Analysis NLP")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose an option", ["Input text", "Demo tweets"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")

    else:
        st.info("Using Demo tweets.")
        sample_tweets = [
            "I'm feeling so blessed today. What a wonderful morning!",   # Positive
            "This phone broke after one day. Absolutely terrible.",     # Negative
            "Amazing service at the restaurant today!",                 # Positive
            "Worst customer experience I’ve ever had.",                 # Negative
            "I just got promoted! I'm thrilled!",                       # Positive
            "The food was cold and tasteless. Never again.",           # Negative
            "Such a fun night with my best friends!",                  # Positive
            "Nothing worked as advertised. Waste of money.",           # Negative
            "Everything about this trip was perfect.",                 # Positive
            "This app crashes constantly. Super frustrating.",         # Negative
        ]

        if st.button("Show Demo Tweets Sentiment"):
            for tweet_text in sample_tweets:
                sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                card_html = create_card(tweet_text, sentiment)
                st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
