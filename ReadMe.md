# Emotion Classification using Machine Learning

A Natural Language Processing (NLP) project that predicts emotions from text using traditional Machine Learning algorithms.

## Overview

This project classifies text into different emotion categories using text vectorization and supervised machine learning.

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- CountVectorizer
- Joblib

## Models Implemented

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | Evaluated |
| Logistic Regression | Evaluated |
| Linear SVM | **88%** |

Linear SVM achieved the best accuracy among the three tested models.

## Project Structure

```text
Emotion-Classifier/
├── train.py
├── predict.py
├── svm_model.pkl
├── vectorizer.pkl
├── emotion_mapping.pkl
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Emotion-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the prediction script:

```bash
python predict.py
```

Enter a sentence when prompted.

Example:

```text
Enter a text to predict the emotion:
I am so happy today!

The predicted emotion is: joy
```

## Workflow

1. Load and preprocess the text dataset.
2. Encode emotion labels.
3. Convert text into numerical features using CountVectorizer.
4. Split the dataset into training and testing sets.
5. Train and evaluate three ML models.
6. Save the best-performing model and supporting objects using Joblib.
7. Load the saved model to predict emotions from new text.

## Future Improvements

- Experiment with TF-IDF.
- Improve classification performance.
- Implement LSTM-based emotion classification.
- Explore Transformer-based NLP models.

---

**Project Status:** Completed — Traditional ML Emotion Classification.
