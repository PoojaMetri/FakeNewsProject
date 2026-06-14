import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from utils.preprocessing import clean_text


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset with title, text, and label columns."""
    data = pd.read_csv(path)
    expected = {"title", "text", "label"}
    if not expected.issubset(set(data.columns)):
        raise ValueError(f"Dataset must contain columns: {expected}")
    return data


def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with sensible defaults."""
    return TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1, 2), max_features=20000)


def train_models(X_train, y_train):
    """Train a set of candidate models and return them."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, solver="liblinear"),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def save_pickle(obj, path: str) -> None:
    """Serialize a model or vectorizer to disk."""
    import pickle

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str):
    """Load a pickle file from disk."""
    import pickle

    with open(path, "rb") as file:
        return pickle.load(file)


def save_json(data: dict, path: str) -> None:
    """Save metrics or model metadata as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_prediction_scores(model: BaseEstimator, X):
    """Return probability scores for the positive class when available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        probs = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
        return probs
    raise AttributeError("Model does not support probability or decision scoring")


def explain_prediction(model: BaseEstimator, vectorizer: TfidfVectorizer, raw_text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    """Return the top positive and negative words influencing a prediction."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    cleaned = clean_text(raw_text)
    X = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = X.toarray()[0]
    nonzero_indices = np.where(tfidf_values > 0)[0]
    if len(nonzero_indices) == 0:
        return []

    word_scores = {}
    if hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 1 or coefs.shape[0] == 1:
            # Binary linear models store coefficients for the positive class only.
            coefs = coefs.flatten()
            predicted_class_idx = np.argmax(model.predict_proba(X), axis=1)[0]
            if hasattr(model, "classes_") and len(model.classes_) == 2 and predicted_class_idx == 0:
                coefs = -coefs
        else:
            class_idx = np.argmax(model.predict_proba(X), axis=1)[0]
            coefs = coefs[class_idx]

        for idx in nonzero_indices:
            word_scores[feature_names[idx]] = float(coefs[idx] * tfidf_values[idx])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for idx in nonzero_indices:
            word_scores[feature_names[idx]] = float(importances[idx] * tfidf_values[idx])
    else:
        for idx in nonzero_indices:
            word_scores[feature_names[idx]] = float(tfidf_values[idx])

    sorted_words = sorted(word_scores.items(), key=lambda item: abs(item[1]), reverse=True)
    return sorted_words[:top_n]


def prepare_batch_predictions(model: BaseEstimator, vectorizer: TfidfVectorizer, df: "pd.DataFrame") -> "pd.DataFrame":
    """Generate batch predictions for an uploaded CSV file."""
    text_column = "text" if "text" in df.columns else None
    if text_column is None:
        raise ValueError("Batch CSV must contain a 'text' column")

    input_text = df[text_column].astype(str).apply(clean_text)
    X = vectorizer.transform(input_text)
    predicted = model.predict(X)
    predicted_probs = get_prediction_scores(model, X)
    class_index = np.argmax(predicted_probs, axis=1)
    confidence = predicted_probs[np.arange(len(predicted)), class_index] * 100

    output = df.copy()
    output["prediction"] = predicted
    output["confidence"] = confidence.round(2)
    return output
