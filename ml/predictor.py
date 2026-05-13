"""
TruthLens — ml/predictor.py
Wraps the trained scikit-learn classifier for inference.

Usage:
    predictor = MLPredictor(model_path="ml/models/classifier.pkl")
    result    = predictor.predict("Some news article text...")
    # => {"label": "FAKE", "confidence": 0.87, "probabilities": {"REAL": 0.13, "FAKE": 0.87}}
"""

import os
import pickle
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Loads a trained TruthLens scikit-learn pipeline and exposes a predict() method.

    The model file is produced by ml/train.py and includes:
      - TF-IDF vectorizer
      - Logistic Regression (or Naive Bayes) classifier
      - Label encoder
    """

    LABELS = ["REAL", "FAKE"]

    def __init__(self, model_path: str = "ml/models/classifier.pkl"):
        self.model_path = model_path
        self.pipeline: Optional[object] = None
        self._load_model()

    def _load_model(self):
        """Load the serialized sklearn pipeline from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(
                "Model file not found at '%s'. "
                "Run ml/train.py to generate it. Predictions will use random fallback.",
                self.model_path
            )
            return

        try:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            logger.info("ML model loaded from '%s'", self.model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self.pipeline = None

    def predict(self, text: str) -> dict:
        """
        Run inference on a single article text.

        Args:
            text: Raw article string

        Returns:
            {
              "label":         "REAL" | "FAKE",
              "confidence":    float (0.0 – 1.0),
              "probabilities": {"REAL": float, "FAKE": float}
            }
        """
        if self.pipeline is None:
            return self._fallback(text)

        try:
            proba = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_

            prob_map  = {str(cls).upper(): float(p) for cls, p in zip(classes, proba)}
            label     = max(prob_map, key=prob_map.get)
            confidence = prob_map[label]

            return {
                "label":         label,
                "confidence":    confidence,
                "probabilities": prob_map
            }
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return self._fallback(text)

    def _fallback(self, text: str) -> dict:
        """
        Heuristic fallback when model is unavailable.
        Uses simple keyword scoring as a rough proxy.
        """
        import re
        fake_keywords = [
            "SHOCKING", "!!!", "BREAKING", "MIRACLE", "THEY DON'T WANT",
            "SUPPRESSED", "WHISTLEBLOWER", "BIG PHARMA", "SHARE BEFORE",
            "UNDENIABLE", "PROVEN", "SECRET"
        ]
        upper = text.upper()
        hits  = sum(1 for kw in fake_keywords if kw in upper)
        fake_prob = min(0.5 + hits * 0.08, 0.95)
        real_prob = round(1.0 - fake_prob, 4)

        label = "FAKE" if fake_prob > 0.5 else "REAL"
        return {
            "label":         label,
            "confidence":    fake_prob if label == "FAKE" else real_prob,
            "probabilities": {"REAL": real_prob, "FAKE": fake_prob}
        }

    def reload(self):
        """Hot-reload the model (useful in long-running servers after retraining)."""
        self._load_model()
