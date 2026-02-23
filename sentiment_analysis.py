import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load your cleaned comments
df = pd.read_csv("comments_cleaned.csv")  # Make sure this file exists

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to handle sentiment analysis safely


def analyze_sentiment(text):
    if pd.isnull(text):
        text = ""  # Avoid errors on empty/NaN
    return sia.polarity_scores(str(text))


# Apply sentiment analysis to all cleaned comments
sentiment_scores = df['clean_body'].apply(analyze_sentiment)

# Expand the dictionary of scores into separate columns
sentiment_df = pd.DataFrame(list(sentiment_scores))

# Combine original data with sentiment scores
df_sentiment = pd.concat([df, sentiment_df], axis=1)

# Save to a new CSV
df_sentiment.to_csv("comments_sentiment.csv", index=False)

print("Sentiment analysis complete! Saved as comments_sentiment.csv")
