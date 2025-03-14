import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# ==============================
# 1. Load and Preprocess Dataset
# ==============================

# Define the file path for the cleaned dataset
file_path = r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\data\processed\cleaned_financial_news.csv"

# Load the dataset using Pandas
df = pd.read_csv(file_path)

# Keep only the required columns and drop any missing values
df = df[['Sentiment', 'Cleaned_Headline']].dropna()

# Convert sentiment labels to numerical values
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}  
df['Sentiment'] = df['Sentiment'].map(sentiment_map)

# Split data into training (80%) and validation (20%) sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Cleaned_Headline'].tolist(),  # List of headlines
    df['Sentiment'].tolist(),  # Corresponding sentiment labels
    test_size=0.2,  # 20% data for validation
    random_state=42  # Ensures reproducibility
)

# ============================
# 2. Tokenization using FinBERT
# ============================

# Load the tokenizer for FinBERT
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Define a custom dataset class for tokenized text
class FinancialNewsDataset(Dataset):
    def __init__(self, texts, labels):
        """
        Initializes the dataset by tokenizing the input texts and storing labels as tensors.
        """
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512)
        self.labels = torch.tensor(labels)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized input IDs, attention masks, and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # Tokenized text
        item["labels"] = self.labels[idx]  # Corresponding sentiment label
        return item

# Create dataset objects for training and validation
train_dataset = FinancialNewsDataset(train_texts, train_labels)
val_dataset = FinancialNewsDataset(val_texts, val_labels)

# =======================================
# 3. Load Pretrained FinBERT for Training
# =======================================

# Load FinBERT model with a classification head (3 labels: negative, neutral, positive)
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)

# ==================================
# 4. Define Evaluation Metrics
# ==================================

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics including accuracy, precision, recall, and F1-score.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert logits to label indices
    acc = accuracy_score(labels, predictions)  # Compute accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")  # Compute other metrics
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ==========================================
# 5. Define Training Arguments & Fine-Tuning
# ==========================================

# Specify training parameters
training_args = TrainingArguments(
    output_dir=r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\models\finbert_results",  # Output directory
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps
    weight_decay=0.01,  # Weight decay (for regularization)
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    logging_dir="../logs",  # Directory for logging training progress
    logging_steps=10,  # Log every 10 steps
    save_strategy="epoch"  # Save the model at the end of each epoch
)

# Initialize the Trainer API for fine-tuning
trainer = Trainer(
    model=model,  # Pretrained FinBERT model
    args=training_args,  # Training configuration
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=val_dataset,  # Validation dataset
    compute_metrics=compute_metrics  # Function to compute evaluation metrics
)

# ============================
# 6. Train the Model
# ============================

# Start model training
trainer.train()

# ============================
# 7. Evaluate Model Performance
# ============================

# Evaluate the model on validation data
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# ============================
# 8. Save Model and Tokenizer
# ============================

# Save the trained model for future use
model.save_pretrained(r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\models\finbert_sentiment")

# Save the tokenizer as well
tokenizer.save_pretrained(r"C:\Users\Harsh Kumar\OneDrive\Documents\Extion Infotech Projects\sentiment_analysis_finance\models\finbert_sentiment")

print("FinBERT training and evaluation complete! Model saved.")

# Trained using FinBERT model for sentiment analysis on financial news headlines. The model is fine-tuned on a dataset of financial news headlines with sentiment labels (negative, neutral, positive) and evaluated using accuracy, precision, recall, and F1-score metrics. The trained model and tokenizer are saved for future use.