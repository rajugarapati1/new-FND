"""
TruthLens — backend/app.py
Flask REST API server.

Endpoints:
  POST /api/analyze   — Run ML pipeline + Claude AI analysis
  GET  /api/health    — Health check
  GET  /api/stats     — Usage statistics

Run:
  pip install flask flask-cors anthropic
  python app.py
"""

import os
import json
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

from .predictor import MLPredictor
from nlp_signals import NLPSignalExtractor

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from frontend dev server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
client        = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
ml_predictor = MLPredictor(model_path="models/classifier.pkl")
nlp_extractor = NLPSignalExtractor()
total_analyzed = 0

# ── Depth instructions ────────────────────────────────────────────────────────
DEPTH_INSTRUCTIONS = {
    "quick":    "Provide a concise 1-2 sentence summary.",
    "standard": "Provide a thorough 3-sentence analysis with clear reasoning.",
    "deep":     "Provide a comprehensive analysis covering linguistic patterns, specific evidence, and a critical credibility breakdown."
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return usage statistics."""
    return jsonify({"total_analyzed": total_analyzed})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.

    Request JSON:
      {
        "text": "<article text>",
        "mode": "quick" | "standard" | "deep"
      }

    Response JSON: See build_claude_prompt() for full schema.
    """
    global total_analyzed

    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "Missing 'text' field"}), 400

    text  = data["text"].strip()
    mode  = data.get("mode", "standard")

    if len(text) < 15:
        return jsonify({"error": "Text too short (minimum 15 characters)"}), 400
    if mode not in DEPTH_INSTRUCTIONS:
        return jsonify({"error": f"Invalid mode '{mode}'. Use: quick, standard, deep"}), 400

    try:
        # Step 1: ML pre-classification (fast, used to enrich the prompt)
        ml_result = ml_predictor.predict(text)
        logger.info("ML pre-classification: %s (%.2f)", ml_result["label"], ml_result["confidence"])

        # Step 2: NLP signal extraction
        signals = nlp_extractor.extract(text)
        logger.info("NLP signals extracted: %s", signals)

        # Step 3: Claude AI full analysis
        prompt   = build_claude_prompt(text, mode, ml_result, signals)
        response = call_claude(prompt)

        total_analyzed += 1
        return jsonify(response)

    except json.JSONDecodeError as e:
        logger.error("JSON parse error from Claude: %s", e)
        return jsonify({"error": "Claude returned malformed JSON"}), 502
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        return jsonify({"error": f"Claude API error: {e}"}), 502
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_claude_prompt(text: str, mode: str, ml_result: dict, signals: dict) -> str:
    """
    Construct the structured prompt for Claude, enriched with ML + NLP context.

    Args:
        text       : Raw article text
        mode       : Analysis depth mode
        ml_result  : {'label': str, 'confidence': float}
        signals    : Dict of pre-computed NLP signal scores (0-100)

    Returns:
        Formatted prompt string
    """
    return f"""You are TruthLens, an expert fake news detection AI trained in NLP, media literacy, and computational journalism.

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

PRE-COMPUTED CONTEXT (from ML pipeline — use as hints, not ground truth):
- ML classifier label: {ml_result["label"]} (confidence: {ml_result["confidence"]:.0%})
- NLP signals (0-100): {json.dumps(signals)}

ANALYSIS DEPTH: {mode.upper()}
{DEPTH_INSTRUCTIONS[mode]}

Respond ONLY with valid JSON — no markdown, no backticks, no preamble:
{{
  "verdict": "REAL" | "FAKE" | "UNCERTAIN",
  "confidence": <integer 0-100>,
  "summary": "<plain-language explanation>",
  "signals": {{
    "emotional_language": <0-100>,
    "source_credibility": <0-100>,
    "factual_consistency": <0-100>,
    "clickbait_score": <0-100>,
    "logical_coherence": <0-100>,
    "bias_indicators": <0-100>
  }},
  "red_flags": ["<flag1>", "<flag2>"],
  "positive_signals": ["<signal1>", "<signal2>"],
  "recommendation": "<one actionable sentence>"
}}"""


def call_claude(prompt: str) -> dict:
    """
    Send prompt to Claude and parse the JSON response.

    Args:
        prompt: Structured analysis prompt

    Returns:
        Parsed dict matching the TruthLens result schema
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = "".join(b.text for b in message.content if hasattr(b, "text"))
    cleaned  = raw_text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("TruthLens backend starting on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
