from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.evaluation import compare_models, evaluate_model, plot_confusion_matrix
from utils.model_utils import (
    build_vectorizer,
    load_dataset,
    save_pickle,
    save_json,
    train_models,
)
from utils.preprocessing import combine_title_and_text, prepare_text_series


DATA_PATH = Path("data/fake_or_real_news.csv")
MODEL_PATH = Path("models/best_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
METRICS_PATH = Path("models/model_metrics.json")
CONFUSION_PNG = Path("models/confusion_matrix.png")


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    df["content"] = combine_title_and_text(df)
    df["cleaned"] = prepare_text_series(df["content"])

    print("Preparing training and test sets...")
    X = df["cleaned"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Building TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training models...")
    models = train_models(X_train_tfidf, y_train)

    print("Evaluating models...")
    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate_model(model, X_test_tfidf, y_test)

    comparison = compare_models(metrics)
    print("Model comparison:")
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["model"]
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")

    print(f"Saving best model to {MODEL_PATH}")
    save_pickle(best_model, MODEL_PATH)
    save_pickle(vectorizer, VECTORIZER_PATH)

    summary = {
        "best_model": best_model_name,
        "metrics": {
            model_name: {
                "accuracy": round(data["accuracy"], 4),
                "precision": round(data["precision"], 4),
                "recall": round(data["recall"], 4),
                "f1": round(data["f1"], 4),
            }
            for model_name, data in metrics.items()
        },
    }
    save_json(summary, METRICS_PATH)

    print(f"Saving confusion matrix to {CONFUSION_PNG}")
    cm = metrics[best_model_name]["confusion_matrix"]
    plot_confusion_matrix(cm, ["FAKE", "REAL"], str(CONFUSION_PNG))

    print("Training complete.")
    print(f"Best model saved: {best_model_name}")


if __name__ == "__main__":
    main()
