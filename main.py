import re
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------
# Step 1: Load the CSV
# ------------------------------
df = pd.read_csv("comments.csv")

# Quick check
print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nDataset shape (rows, columns):")
print(df.shape)

# Check for nulls and deleted comments
print("\nNumber of null body rows:", df['body'].isnull().sum())
print("Number of [deleted] comments:", (df['body'] == "[deleted]").sum())
print("Number of [removed] comments:", (df['body'] == "[removed]").sum())

# ------------------------------
# Step 2: Clean the comments
# ------------------------------


def clean_text(text):
    if pd.isnull(text):
        return None
    text = text.strip()
    if text.lower() in ["[deleted]", "[removed]"]:
        return None
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text


df['clean_body'] = df['body'].apply(clean_text)

# Drop any rows where cleaning returned None
df = df.dropna(subset=['clean_body'])

print("\nFirst 5 cleaned comments:")
print(df['clean_body'].head())
print("\nNumber of usable comments:", len(df))

# ------------------------------
# Step 3: Extract stock tickers
# ------------------------------


def extract_tickers(text):
    """
    Extract stock tickers in the format $XXX (1-5 uppercase letters)
    Returns a list of tickers found in the comment.
    """
    return re.findall(r'\$[A-Z]{1,5}', text)


df['tickers'] = df['clean_body'].apply(extract_tickers)

print("\nFirst 5 comments with tickers:")
print(df[['clean_body', 'tickers']].head())
print("\nTotal number of comments (including ones with no tickers):", len(df))

# ------------------------------
# Step 4: FinBERT Sentiment Analysis
# ------------------------------
print("\nLoading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone")
label_map = {0: "neutral", 1: "positive", 2: "negative"}


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].numpy()
    probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
    label_id = int(np.argmax(probs))
    return label_map[label_id]


# Apply FinBERT to all comments
df['sentiment'] = df['clean_body'].apply(get_sentiment)

# ------------------------------
# Step 5: Check results
# ------------------------------
print("\nFirst 5 comments with tickers and sentiment:")
print(df[['clean_body', 'tickers', 'sentiment']].head())
