# Sentiment Analysis on Financial News

This project aims to perform **sentiment analysis** on financial news headlines using **machine learning** and **deep learning models**. It compares a **Linear Regression model** and **FinBERT**, a financial domain-specific BERT model, to classify headlines into **positive, neutral, or negative sentiments**.

## 📁 Project Structure

SENTIMENT_ANALYSIS_FINANCE/ 
│── data/  
|  ├── raw/ # Original financial news dataset 
|  |   |── financial_news.csv
|  ├── processed/ # Cleaned and preprocessed data 
|  |   ├── cleaned_financial_news.csv 
│── logs/ # Training logs 
│── models/ 
|  │ ├── finbert_results/ # FinBERT training outputs 
|  │ ├── finbert_sentiment/ # Trained FinBERT model 
│  ├── sentiment_model.pkl # Trained Linear Regression model 
|  │ ├── tfidf_vectorizer.pkl # TF-IDF vectorizer 
│── preprocessing/ 
|  ├── preprocess_data.py # Data cleaning and processing 
│── venv/ # Virtual environment 
│── train_model.py # Training script for Linear Regression 
│── train_model_bert.py # Training script for FinBERT 
│── sentiment_analysis.py # Inference script 
│── requirements.txt # Dependencies 
│── README.md # Project documentation


# Models & Performance

1️⃣ Linear Regression Model
Accuracy: 74%

Classification Report:

Sentiment	      Precision	Recall	F1-Score 	Support
Negative	            0.79	      0.41	   0.54	   113
Neutral	            0.73	      0.95	   0.82	   567
Positive	            0.76	      0.45	   0.57	   289
Overall Accuracy	   0.74	      -	      -	      969

2️⃣ FinBERT Model
Accuracy: 81.2%

Evaluation Results:

{
  "eval_loss": 1.013,
  "eval_accuracy": 0.812,
  "eval_precision": 0.8119,
  "eval_recall": 0.8121,
  "eval_f1": 0.8120
}

# Why FinBERT?
FinBERT is a specialized BERT model trained on financial text, making it more effective for contextual sentiment analysis than traditional machine learning models.

# Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/harshathghara/sentiment-analysis-finance.git
cd sentiment-analysis-finance

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Train the Models

   :-Train the Linear Regression Model

      python train_model.py

   :-Train the FinBERT Model
      python train_model_bert.py

4️⃣ Run Sentiment Analysis on New Data

python sentiment_analysis.py

# Future Enhancements

✔ Improve FinBERT model with hyperparameter tuning
✔ Experiment with LSTM, CNN, or other transformer models
✔ Add real-time financial news scraping and analysis
✔ Deploy the model using Flask/FastAPI

# Contributors

Harsh Kumar – Machine Learning Engineer

# License

This project is licensed under the MIT License.

