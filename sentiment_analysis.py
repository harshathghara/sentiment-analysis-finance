import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to saved models
finbert_model_path = "models/finbert_sentiment"
lr_model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

# Load FinBERT Model and Tokenizer
finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model_path)
finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_path)
finbert_model.eval()  # Set model to evaluation mode

# Load Linear Regression Model and Vectorizer
with open(lr_model_path, "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open(vectorizer_path, "rb") as vec_file:
    vectorizer = pickle.load(vec_file)


def preprocess_text(text):
    """
    Preprocess the input text: Lowercasing and stripping extra spaces.
    """
    return text.lower().strip()


def predict_with_finbert(text):
    """
    Predict sentiment using the Fine-tuned FinBERT model.
    """
    inputs = finbert_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[predicted_class]


# Define sentiment mapping
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

def predict_with_lr(text):
    """ Predict sentiment using Linear Regression Model """
    text_tfidf = vectorizer.transform([text])  # Convert text to TF-IDF format
    predicted_class = lr_model.predict(text_tfidf)[0]  # Get predicted sentiment

    print(f"Predicted Sentiment from LR: {predicted_class}")  # Debugging output

    # Ensure key exists before accessing
    if predicted_class not in sentiment_map:
        return f"Unknown Sentiment: {predicted_class}"
    
    return predicted_class  # Directly return the sentiment label



if __name__ == "__main__":
    while True:
        text = input("\nEnter a financial news headline (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break

        text = preprocess_text(text)

        finbert_prediction = predict_with_finbert(text)
        lr_prediction = predict_with_lr(text)

        print("\n--- Sentiment Analysis Results ---")
        print(f"FinBERT Prediction: {finbert_prediction}")
        print(f"Linear Regression Prediction: {lr_prediction}")


# Take the help of this input while running the code
'''Positive Sentiment:
✅ "Tech stocks soar as market hits record highs."
✅ "Company X reports better-than-expected earnings, stock surges 10%."

Neutral Sentiment:
➖ "Federal Reserve maintains interest rates at current levels."
➖ "Oil prices remain stable as supply and demand balance out."

Negative Sentiment:
❌ "Stock market plunges amid recession fears."
❌ "Company Y faces lawsuit over misleading financial statements."  '''
