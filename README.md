# Fake News Detection System

## Overview

This project is a Machine Learning-based Fake News Detection System that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) techniques. It provides real-time predictions through an interactive web application.


## Objective

To develop a system that can identify misinformation by analyzing news content and predicting its authenticity with high accuracy.

## Features

* Real-time prediction of news as REAL or FAKE
* Confidence score for predictions
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Multiple machine learning models comparison
* Interactive web interface using Streamlit
* Batch prediction support (optional)


## Tech Stack

* Programming Language: Python
* Libraries:

  * pandas
  * numpy
  * scikit-learn
  * nltk
  * streamlit



## Project Structure

```
FakeNewsDetection/
│── app.py
│── model.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
│── data/
│     └── news.csv
```


## Machine Learning Workflow

1. Data Collection (Kaggle dataset)
2. Data Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training (Logistic Regression, Naive Bayes, Random Forest)
5. Model Evaluation (Accuracy, Precision, Recall, F1 Score)
6. Deployment using Streamlit

## Installation and Setup

### Step 1: Clone Repository

```
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the Application

```
streamlit run app.py
```

## Live Demo

Add your deployed Streamlit link here
Example: https://your-app-name.streamlit.app


## Model Performance

* Accuracy: approximately 95 percent using Logistic Regression
* Evaluation Metrics: Precision, Recall, F1 Score


## Screenshots

Add screenshots of the application interface here

## Demo Video

Add your demo video link (YouTube or Google Drive)

## Future Enhancements

* Integration of deep learning models such as BERT
* Multilingual fake news detection
* Explainable AI for highlighting influential words
* Browser extension for real-time verification



## Applications

* Social media content verification
* News validation platforms
* Journalism and media analysis
* Cybersecurity and misinformation control


## Acknowledgements

* Kaggle dataset
* Open-source Python libraries



## Contact

For any queries or suggestions, feel free to reach out.



## License

This project is for educational and research purposes.
