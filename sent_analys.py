from transformers import pipeline


import json
import re
import string
import emoji
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

# Download necessary NLTK data

# Load spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Load the pre-trained sentiment analysis model
sentiment_pipeline = pipeline("text-classification", model="MarieAngeA13/Sentiment-Analysis-BERT")

def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Function to replace emoji with its meaning
def demojize_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))  # Replace emoji with description

# Function to apply lemmatization using spaCy
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct])
    return lemmatized_text

# Complete preprocessing pipeline
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = remove_url(text)
    # Remove punctuation
    text = remove_punctuation(text)
    # Replace emoji with text
    text = demojize_text(text)
    # Remove stopwords
    text = remove_stopwords(text)
    # Lemmatization
    text = lemmatize_text(text)
   
    return text

def analyze_sentiment(text):
    """
    Analyze sentiment using the pre-trained Hugging Face model.
    """

    download('punkt_tab')
    download('stopwords')
    text =  preprocess_text(text)
    result = sentiment_pipeline(text)[0]
    
    sentiment = result['label'].lower()
    confidence = result['score']
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }
