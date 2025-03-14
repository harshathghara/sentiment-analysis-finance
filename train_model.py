import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Download NLTK stopwords and tokenizer if not already available
nltk.download("stopwords")
nltk.download("punkt")

# Define file path for cleaned dataset
file_path = r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\data\processed\cleaned_financial_news.csv"

# Attempt to load CSV file with proper encoding
try:
    df = pd.read_csv(file_path, encoding="utf-8")  # Try UTF-8
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="ISO-8859-1")  # Fallback encoding

# Display first few rows for verification
print(df.head())

# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Drop any rows with missing values (if needed)
df.dropna(inplace=True)

# Re-check missing values after handling
print("Missing values after handling:")
print(df.isnull().sum())

# Ensure correct column names
df.columns = ["Sentiment", "Headline", "Cleaned_Headline"]

# Text Preprocessing Function
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Apply text preprocessing
df["Cleaned_Headline"] = df["Cleaned_Headline"].apply(clean_text)

# Define features (X) and target labels (y)
X = df["Cleaned_Headline"]
y = df["Sentiment"]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer for future use
with open(r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\models\sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\models\tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")


#this was using linear regression