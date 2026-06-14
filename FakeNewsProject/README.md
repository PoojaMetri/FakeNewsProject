# Fake News Detection System

This project implements a complete fake news detection pipeline using Python and Streamlit.

## Features
- Real dataset support with `title`, `text`, `label` columns
- Text preprocessing: lowercase, punctuation removal, stopword removal, tokenization
- TF-IDF vectorization
- Model training for:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Model comparison by accuracy, precision, recall, F1 score
- Best model persistence with `pickle`
- Streamlit app for single and batch prediction
- Confidence score and important word highlighting
- Confusion matrix visualization

## Project Structure
- `data/` - sample dataset and dataset downloader
- `models/` - stored model, vectorizer, and metrics files
- `utils/` - preprocessing, training, and evaluation helpers
- `app/` - Streamlit application
- `train.py` - training pipeline

## Setup
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the full dataset (optional, or use the sample dataset):

```bash
python data/download_dataset.py
```

3. Train the models:

```bash
python train.py
```

4. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Notes
- The app supports CSV upload for batch predictions.
- Color-coded results display green for `REAL` and red for `FAKE`.
- The code is organized for easy deployment to Streamlit Cloud.
