import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd


def clean_text(text: str) -> str:
    """Normalize, remove punctuation, remove stopwords, and tokenize simple text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def tokenize_text(text: str) -> List[str]:
    """Tokenize a cleaned text string into words."""
    if not isinstance(text, str):
        return []
    return [token for token in text.split() if token]


def combine_title_and_text(df: pd.DataFrame) -> pd.Series:
    """Combine title and text columns into a single feature column."""
    combined = df["title"].fillna("") + " " + df["text"].fillna("")
    return combined


def prepare_text_series(series: pd.Series) -> pd.Series:
    """Apply cleaning and tokenization to a pandas Series of text."""
    return series.fillna("").astype(str).apply(clean_text)
