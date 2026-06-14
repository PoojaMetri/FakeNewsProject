import io
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so top-level imports work from Streamlit.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.evaluation import plot_confusion_matrix
from utils.model_utils import load_pickle, explain_prediction, prepare_batch_predictions
from utils.preprocessing import clean_text


MODEL_PATH = Path("models/best_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
METRICS_PATH = Path("models/model_metrics.json")
CONFUSION_PATH = Path("models/confusion_matrix.png")


def load_resources():
    model = load_pickle(MODEL_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
    return model, vectorizer, metrics


def get_prediction(model, vectorizer, text: str):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    predicted = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = round(100 * max(probabilities), 2)
    return predicted, confidence, probabilities


def format_highlights(highlights: List[tuple]) -> str:
    if not highlights:
        return "No highlighted keywords detected."
    lines = []
    for word, score in highlights:
        color = "#16a34a" if score > 0 else "#dc2626"
        lines.append(f"<span style='padding:2px 6px; margin:2px; border-radius:4px; background:{color}; color:white;'>{word}</span>")
    return "".join(lines)


def get_batch_dataframe(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        return None


def run_app():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide",
    )

    st.title("Fake News Detector")
    st.markdown(
        "Use the model to classify news articles as **REAL** or **FAKE**. Upload a CSV for batch predictions or paste a text snippet to evaluate one article at a time."
    )

    model, vectorizer, metrics = load_resources()

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Predict a single news article")
        user_text = st.text_area(
            "Paste the news title + body text here", height=240, placeholder="Enter news text to classify..."
        )
        if st.button("Classify article") and user_text.strip():
            label, confidence, probabilities = get_prediction(model, vectorizer, user_text)
            color = "green" if label == "REAL" else "red"
            st.markdown(f"### Result: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence}%")
            probability_labels = model.classes_
            prob_table = pd.DataFrame(
                {"label": probability_labels, "probability": [round(v * 100, 2) for v in probabilities]}
            )
            st.table(prob_table)

            highlights = explain_prediction(model, vectorizer, clean_text(user_text), top_n=10)
            st.markdown("**Important words influencing prediction:**")
            st.markdown(format_highlights(highlights), unsafe_allow_html=True)

    with col2:
        st.header("Model summary")
        if metrics:
            st.metric("Best model", metrics.get("best_model", "Unknown"))
            summary = metrics.get("metrics", {})
            if summary:
                summary_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "model"})
                st.dataframe(summary_df.style.format({"accuracy": "{:.2f}", "precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}))
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), caption="Confusion Matrix", use_column_width=True)

    st.markdown("---")
    st.header("Batch prediction")
    uploaded_file = st.file_uploader("Upload a CSV file containing a `text` column", type=["csv"])
    if uploaded_file:
        df = get_batch_dataframe(uploaded_file)
        if df is None:
            st.error("Unable to read the uploaded CSV. Please upload a valid CSV file.")
        elif "text" not in df.columns:
            st.error("CSV must contain a 'text' column for batch prediction.")
        else:
            batch_results = prepare_batch_predictions(model, vectorizer, df)
            st.success("Batch predictions generated.")
            st.dataframe(batch_results.head(20))
            csv_bytes = batch_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.caption("This app uses TF-IDF and classic supervised models to detect fake news. For best results, train with a larger dataset.")


if __name__ == "__main__":
    run_app()
