import re

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    return " ".join(text.split())


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk

        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = _get_vader_analyzer()
    scores = df["content_clean"].apply(lambda text: analyzer.polarity_scores(text))
    df = df.copy()
    df["sent_compound"] = scores.apply(lambda item: item["compound"])
    df["sent_pos"] = scores.apply(lambda item: item["pos"])
    df["sent_neg"] = scores.apply(lambda item: item["neg"])
    df["sent_neu"] = scores.apply(lambda item: item["neu"])
    return df
