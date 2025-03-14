import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources for text processing
nltk.download("punkt")  # Tokenizer (splits text into words)
nltk.download("stopwords")  # Stopwords (e.g., "the", "is", "and", etc.)
nltk.download("wordnet")  # WordNet Lemmatizer (converts words to base form)

# Define file paths
file_path = r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\data\raw\financial_news.csv"  # Path to raw dataset (update if needed)
output_path = r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\data/processed/cleaned_financial_news.csv"  # Path to save cleaned data

# Try loading the dataset with UTF-8 encoding first, fall back to ISO-8859-1 if it fails
try:
    df = pd.read_csv(file_path, encoding="utf-8", header=None, names=["Sentiment", "Headline"])
except UnicodeDecodeError:
    print("Failed with encoding: utf-8")
    df = pd.read_csv(file_path, encoding="ISO-8859-1", header=None, names=["Sentiment", "Headline"])
    print("Successfully loaded with encoding: ISO-8859-1")

# Display dataset information
print(f"Dataset Loaded Successfully. Shape: {df.shape}")
print(df.head())  # Show first few rows to verify data structure

# Function to preprocess text data
def preprocess_text(text):
    """
    Function to clean and preprocess text:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenize words
    4. Remove stopwords
    5. Apply lemmatization
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters, punctuation, etc.
    words = word_tokenize(text)  # Split text into words (tokenization)

    # Remove stopwords (common words that don't add meaning)
    words = [word for word in words if word not in stopwords.words("english")]

    # Initialize lemmatizer (reduces words to their root form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization

    return " ".join(words)  # Join words back into a single string

# Apply preprocessing function to the 'Headline' column
df["Cleaned_Headline"] = df["Headline"].astype(str).apply(preprocess_text)

# Save the cleaned dataset as a new CSV file
df.to_csv(output_path, index=False)

# Confirm completion
print(f"Preprocessed data saved at: {output_path}")


