"""
TruthLens — ml/nlp_signals.py
Extracts interpretable NLP signals from news text before passing to Claude.

Signals returned (all 0-100):
  - emotional_language  : High = lots of fear/anger/sensationalist words
  - source_credibility  : High = references to experts, institutions, data
  - factual_consistency : High = presence of specific, verifiable facts
  - clickbait_score     : High = exaggerated headline patterns
  - logical_coherence   : High = structured argumentation
  - bias_indicators     : High = partisan / one-sided framing

Usage:
    extractor = NLPSignalExtractor()
    signals   = extractor.extract("Some news article text...")
"""

import re
import math
from typing import Dict


class NLPSignalExtractor:
    """
    Rule-based + statistical NLP signal extractor.
    Designed to be fast and interpretable (no model loading required).
    """

    # ── Word lists ─────────────────────────────────────────────────────────────

    EMOTIONAL_WORDS = {
        "shocking", "outrageous", "terrifying", "horrifying", "disgusting",
        "unbelievable", "incredible", "explosive", "bombshell", "scandal",
        "catastrophic", "devastating", "alarming", "explosive", "secret",
        "cover-up", "dangerous", "threat", "crisis", "emergency", "disaster",
        "evil", "corrupt", "criminal", "liar", "fraud", "hoax", "conspiracy"
    }

    SOURCE_INDICATORS = {
        "according to", "study", "research", "scientists", "researchers",
        "experts", "officials", "report", "journal", "university", "institute",
        "professor", "doctor", "dr.", "phd", "published", "data", "evidence",
        "spokesperson", "statement", "confirmed", "announced", "survey"
    }

    FACT_INDICATORS = {
        "percent", "%", "billion", "million", "thousand",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "said", "stated", "reported", "announced", "confirmed"
    }

    CLICKBAIT_PATTERNS = [
        r"you won'?t believe",
        r"doctors? hate",
        r"one weird trick",
        r"this (will|is going to) (shock|amaze|blow your mind)",
        r"SHARE before (it['s]* )?deleted",
        r"they don'?t want you to know",
        r"miracle (cure|remedy|solution)",
        r"big (pharma|media|tech|government)",
        r"\b(!!+|\?\?+)",          # multiple exclamation/question marks
        r"\bALL CAPS\b",
        r"BREAKING[:\s]",
        r"[A-Z]{4,}",              # long all-caps sequences
    ]

    COHERENCE_INDICATORS = {
        "however", "therefore", "because", "although", "since", "while",
        "furthermore", "consequently", "nevertheless", "in contrast",
        "on the other hand", "as a result", "for example", "specifically",
        "in addition", "moreover", "despite", "regardless", "thus"
    }

    BIAS_WORDS = {
        "radical", "extreme", "far-left", "far-right", "liberal agenda",
        "conservative agenda", "elite", "globalist", "deep state",
        "mainstream media", "fake news", "corrupt media", "regime",
        "communist", "fascist", "socialist", "propaganda"
    }

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, int]:
        """
        Extract all NLP signals from text.

        Args:
            text: Raw article string

        Returns:
            Dict mapping signal name → integer score (0-100)
        """
        text_lower = text.lower()
        words      = re.findall(r'\b\w+\b', text_lower)
        word_count = max(len(words), 1)

        return {
            "emotional_language":  self._emotional_language(text_lower, words, word_count),
            "source_credibility":  self._source_credibility(text_lower, word_count),
            "factual_consistency": self._factual_consistency(text, text_lower, word_count),
            "clickbait_score":     self._clickbait_score(text, text_lower),
            "logical_coherence":   self._logical_coherence(text_lower, word_count),
            "bias_indicators":     self._bias_indicators(text_lower, word_count),
        }

    # ── Signal methods ─────────────────────────────────────────────────────────

    def _emotional_language(self, text_lower: str, words: list, word_count: int) -> int:
        hits        = sum(1 for w in words if w in self.EMOTIONAL_WORDS)
        exclamation = text_lower.count("!")
        caps_ratio  = sum(1 for c in text_lower if c.isupper()) / max(len(text_lower), 1)

        raw = (hits / word_count) * 300 + (exclamation * 5) + (caps_ratio * 100)
        return self._clamp(int(raw))

    def _source_credibility(self, text_lower: str, word_count: int) -> int:
        hits = sum(1 for phrase in self.SOURCE_INDICATORS if phrase in text_lower)
        raw  = (hits / max(word_count / 100, 1)) * 60
        return self._clamp(int(raw))

    def _factual_consistency(self, text: str, text_lower: str, word_count: int) -> int:
        # Count numeric data points
        numbers = re.findall(r'\b\d+[\.,]?\d*\s*(%|billion|million|thousand)?\b', text)
        hits    = sum(1 for phrase in self.FACT_INDICATORS if phrase in text_lower)

        raw = min(len(numbers) * 8, 40) + min(hits * 4, 40) + (10 if word_count > 100 else 0)
        return self._clamp(int(raw))

    def _clickbait_score(self, text: str, text_lower: str) -> int:
        pattern_hits = sum(
            1 for p in self.CLICKBAIT_PATTERNS
            if re.search(p, text, re.IGNORECASE)
        )
        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        raw        = pattern_hits * 20 + caps_words * 5
        return self._clamp(int(raw))

    def _logical_coherence(self, text_lower: str, word_count: int) -> int:
        hits = sum(1 for phrase in self.COHERENCE_INDICATORS if phrase in text_lower)
        # Longer texts with logical connectives score higher
        length_bonus = min(word_count / 10, 20)
        raw          = hits * 12 + length_bonus
        return self._clamp(int(raw))

    def _bias_indicators(self, text_lower: str, word_count: int) -> int:
        hits = sum(1 for phrase in self.BIAS_WORDS if phrase in text_lower)
        raw  = (hits / max(word_count / 50, 1)) * 80
        return self._clamp(int(raw))

    # ── Utils ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
        return max(lo, min(hi, value))
