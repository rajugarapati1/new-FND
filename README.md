# TruthLens — AI Fake News Detector

> Detect fake news using NLP signals, a scikit-learn ML classifier, and Claude AI.

---

## Project structure

```
truthlens/
├── frontend/
│   ├── index.html       ← HTML markup
│   ├── styles.css       ← All CSS (variables, layout, components)
│   └── app.js           ← UI logic, Claude API calls, result rendering
│
├── backend/
│   └── app.py           ← Flask REST API (POST /api/analyze)
│
├── ml/
│   ├── train.py         ← Train & evaluate the sklearn pipeline
│   ├── predictor.py     ← Inference wrapper (MLPredictor class)
│   └── nlp_signals.py   ← Rule-based NLP signal extractor
│
├── data/
│   └── dataset_utils.py ← Dataset loaders, validator, sample generator
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or create a .env file:
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### 3. Prepare the dataset

```bash
# Option A — Generate a tiny synthetic dataset for smoke-testing
python data/dataset_utils.py generate --out data/dataset.csv --real 200 --fake 200

# Option B — Use WELFake (recommended, 72k articles)
# Download from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
# Place WELFake_Dataset.csv in data/, then:
python -c "
from data.dataset_utils import load_welfake
df = load_welfake('data/WELFake_Dataset.csv')
df.to_csv('data/dataset.csv', index=False)
"
```

### 4. Train the ML classifier

```bash
python ml/train.py                        # Logistic Regression (default)
python ml/train.py --model nb             # Naive Bayes
python ml/train.py --data data/custom.csv # Custom dataset
```

Outputs → `ml/models/classifier.pkl` and `ml/models/metrics.json`

### 5. Start the backend

```bash
python backend/app.py
# Server runs on http://localhost:5000
```

### 6. Open the frontend

Open `frontend/index.html` in your browser.  
The frontend calls the Anthropic API directly (no backend required for the Claude step).  
The backend adds ML pre-classification and NLP signals.

---

## API reference

### `POST /api/analyze`

**Request:**
```json
{
  "text": "Article text here…",
  "mode": "quick" | "standard" | "deep"
}
```

**Response:**
```json
{
  "verdict": "REAL" | "FAKE" | "UNCERTAIN",
  "confidence": 87,
  "summary": "This article…",
  "signals": {
    "emotional_language": 72,
    "source_credibility": 30,
    "factual_consistency": 25,
    "clickbait_score": 80,
    "logical_coherence": 20,
    "bias_indicators": 65
  },
  "red_flags": ["Excessive caps", "No sources cited"],
  "positive_signals": [],
  "recommendation": "Cross-check with a verified fact-checking site."
}
```

### `GET /api/health`
Returns `{"status": "ok", "timestamp": ...}`

### `GET /api/stats`
Returns `{"total_analyzed": ...}`

---

## Recommended datasets

| Dataset    | Size       | Source |
|------------|------------|--------|
| WELFake    | 72,134 articles | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| LIAR       | 12,836 statements | [UCSB](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |
| FakeNewsNet| ~23,000 articles | [GitHub](https://github.com/KaiDMML/FakeNewsNet) |

---

## Tech stack

- **Frontend**: Vanilla HTML/CSS/JS, DM Serif Display font, Tabler Icons
- **Backend**: Python 3.11, Flask, Flask-CORS
- **ML**: scikit-learn (TF-IDF + Logistic Regression / Naive Bayes)
- **NLP**: Rule-based signal extraction, NLTK (optional)
- **AI**: Anthropic Claude Sonnet 4
