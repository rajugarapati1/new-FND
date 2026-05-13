"""
TruthLens — ml/train.py
Trains and evaluates the ML classifier pipeline.

Pipeline:
  Raw text → TF-IDF vectorizer → Logistic Regression classifier

Dataset expected at: data/dataset.csv
  Columns: text (str), label (0 = REAL, 1 = FAKE)

Outputs:
  ml/models/classifier.pkl   — Trained sklearn pipeline
  ml/models/metrics.json     — Evaluation metrics

Usage:
  python ml/train.py
  python ml/train.py --model nb         # Use Naive Bayes instead
  python ml/train.py --data data/custom.csv
"""

import argparse
import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "ml/models"
MODEL_PATH  = os.path.join(MODEL_DIR, "classifier.pkl")
METRICS_PATH= os.path.join(MODEL_DIR, "metrics.json")
DEFAULT_DATA= "data/dataset.csv"
RANDOM_SEED = 42


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove excess whitespace.
    NLTK-based stemming/stop-word removal is optional (see README).
    """
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)       # keep alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Pipeline factory ───────────────────────────────────────────────────────────

def build_pipeline(model_type: str = "lr") -> Pipeline:
    """
    Build the sklearn Pipeline.

    Args:
        model_type: "lr" (Logistic Regression) | "nb" (Naive Bayes)

    Returns:
        Untrained sklearn Pipeline
    """
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # apply log normalization
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r'\w{2,}',  # skip single-char tokens
        min_df=2,                 # ignore very rare terms
    )

    if model_type == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
            class_weight="balanced",  # handle label imbalance
        )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ── Main training routine ──────────────────────────────────────────────────────

def train(data_path: str = DEFAULT_DATA, model_type: str = "lr") -> dict:
    """
    Full training run: load data → preprocess → split → train → evaluate → save.

    Args:
        data_path  : Path to CSV dataset
        model_type : "lr" or "nb"

    Returns:
        metrics dict
    """
    logger.info("Loading dataset from '%s'…", data_path)
    df = pd.read_csv(data_path)

    assert "text" in df.columns and "label" in df.columns, \
        "Dataset must have 'text' and 'label' columns"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(preprocess_text)
    df["label"] = df["label"].astype(int)

    logger.info("Dataset: %d samples | REAL: %d | FAKE: %d",
                len(df), (df["label"] == 0).sum(), (df["label"] == 1).sum())

    X = df["text"].values
    y = df["label"].values

    # Train / test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # 5-fold cross-validation on training data
    logger.info("Running 5-fold cross-validation…")
    pipeline = build_pipeline(model_type)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores= cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info("CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Final fit on full training set
    logger.info("Training final model on full training set…")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    logger.info("Training complete in %.2fs", time.time() - t0)

    # Evaluation
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type":    model_type,
        "dataset":       data_path,
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "accuracy":      round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc":       round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_mean":       round(float(cv_scores.mean()), 4),
        "cv_std":        round(float(cv_scores.std()), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["REAL", "FAKE"], output_dict=True
        ),
    }

    logger.info("Test accuracy : %.4f", metrics["accuracy"])
    logger.info("ROC AUC       : %.4f", metrics["roc_auc"])
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # Save model and metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to '%s'", MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to '%s'", METRICS_PATH)

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TruthLens ML classifier")
    parser.add_argument("--data",  default=DEFAULT_DATA, help="Path to dataset CSV")
    parser.add_argument("--model", default="lr", choices=["lr", "nb"], help="Classifier type")
    args = parser.parse_args()
    train(data_path=args.data, model_type=args.model)
