"""
TruthLens — data/dataset_utils.py
Dataset loading, validation, augmentation, and sample generation utilities.

Supported dataset formats:
  1. WELFake     — kaggle dataset (text, label)  label: 0=REAL, 1=FAKE
  2. LIAR        — tsv format (statement, label)
  3. FakeNewsNet — json format
  4. Custom CSV  — (text, label) columns

Usage:
  # Load from Kaggle WELFake CSV
  df = load_welfake("data/WELFake_Dataset.csv")

  # Create a small sample dataset for testing
  df = generate_sample_dataset()
  df.to_csv("data/dataset.csv", index=False)

  # Validate any dataset
  report = validate_dataset("data/dataset.csv")
  print(report)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_REAL = 0
LABEL_FAKE = 1


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_welfake(path: str) -> pd.DataFrame:
    """
    Load the WELFake dataset (72,134 articles, 0=REAL / 1=FAKE).
    Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

    Args:
        path: Path to WELFake_Dataset.csv

    Returns:
        Cleaned DataFrame with columns: text, label
    """
    df = pd.read_csv(path)
    # Combine title + text for richer features
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    logger.info("WELFake loaded: %d rows", len(df))
    return df


def load_liar(path: str) -> pd.DataFrame:
    """
    Load the LIAR dataset (TSV format).
    Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

    Maps 6-way labels → binary: pants-fire/false/barely-true → FAKE, rest → REAL

    Args:
        path: Path to train.tsv, valid.tsv, or test.tsv

    Returns:
        DataFrame with columns: text, label
    """
    FAKE_LABELS = {"pants-fire", "false", "barely-true"}

    df = pd.read_csv(path, sep="\t", header=None,
                     names=["id","label","statement","subjects","speaker",
                             "job","state","party","barely_true","false",
                             "half_true","mostly_true","pants_fire","context"])
    df["text"]  = df["statement"].fillna("")
    df["label"] = df["label"].apply(lambda x: LABEL_FAKE if x in FAKE_LABELS else LABEL_REAL)
    logger.info("LIAR loaded: %d rows", len(df))
    return df[["text", "label"]]


def load_fakenewsnet(data_dir: str) -> pd.DataFrame:
    """
    Load FakeNewsNet dataset from local directory structure.
    Expects: data_dir/{gossipcop,politifact}/{fake,real}/news content.json

    Args:
        data_dir: Root directory of FakeNewsNet data

    Returns:
        DataFrame with columns: text, label
    """
    records = []
    root    = Path(data_dir)

    for source in ["gossipcop", "politifact"]:
        for verdict in ["fake", "real"]:
            label_val = LABEL_FAKE if verdict == "fake" else LABEL_REAL
            folder    = root / source / verdict

            if not folder.exists():
                continue

            for json_file in folder.glob("*/news content.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    text = (data.get("title", "") + " " + data.get("text", "")).strip()
                    if text:
                        records.append({"text": text, "label": label_val})
                except Exception as e:
                    logger.debug("Skipping %s: %s", json_file, e)

    df = pd.DataFrame(records)
    logger.info("FakeNewsNet loaded: %d rows", len(df))
    return df


def load_custom_csv(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load any CSV with configurable column names.

    Args:
        path      : CSV file path
        text_col  : Column name for article text
        label_col : Column name for binary label (0=REAL, 1=FAKE)

    Returns:
        Standardized DataFrame with columns: text, label
    """
    df = pd.read_csv(path)
    assert text_col in df.columns, f"Column '{text_col}' not found in {path}"
    assert label_col in df.columns, f"Column '{label_col}' not found in {path}"
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


# ── Sample dataset generator ──────────────────────────────────────────────────

def generate_sample_dataset(n_real: int = 50, n_fake: int = 50) -> pd.DataFrame:
    """
    Generate a minimal synthetic dataset for smoke-testing the pipeline.
    NOT suitable for production training — use a real dataset.

    Args:
        n_real: Number of "real" samples to generate
        n_fake: Number of "fake" samples to generate

    Returns:
        DataFrame with columns: text, label
    """
    real_templates = [
        "The Federal Reserve raised interest rates by {n} basis points on Wednesday. "
        "Fed Chair {name} said officials will monitor inflation data before deciding on future adjustments.",
        "Scientists at {uni} published research in {journal} suggesting {finding}. "
        "The study, which analyzed {n} participants, found statistically significant results (p < 0.05).",
        "The {country} government announced a new policy on {topic}. "
        "A spokesperson confirmed the measure would take effect in {month}.",
        "Shares of {company} rose {n}% on Thursday after the company reported quarterly earnings "
        "that exceeded analyst expectations.",
    ]
    fake_templates = [
        "SHOCKING: The government DOESN'T WANT you to know that {claim}!!! "
        "Thousands of whistleblowers are coming forward. SHARE before this gets deleted!!!",
        "BREAKING: Scientists PROVEN that {claim}. Big Pharma is desperately trying to suppress this. "
        "Doctors HATE this one weird trick!!!",
        "MIRACLE CURE discovered in the Amazon rainforest can {claim} in just 3 days. "
        "The deep state has been hiding this for DECADES.",
        "The radical agenda is destroying {thing}. The mainstream media refuses to report the TRUTH. "
        "Wake up sheeple!!!"
    ]

    real_fills = dict(
        n=["25", "50", "100", "200", "500", "1000"],
        name=["Jerome Powell", "Dr. Sarah Chen", "Prof. James Okafor"],
        uni=["MIT", "Stanford University", "University of Cambridge"],
        journal=["Nature", "Science", "The Lancet", "JAMA"],
        finding=["a correlation between diet and longevity", "new exoplanet candidates"],
        country=["The US", "Germany", "Japan", "Canada"],
        topic=["climate change", "infrastructure", "education funding"],
        month=["January", "March", "June"],
        company=["Apple", "Tesla", "Microsoft"],
    )
    fake_fills = dict(
        claim=["chemtrails contain mind-control chemicals",
               "vaccines cause autism",
               "5G towers spread disease",
               "the moon landing was faked"],
        thing=["our children", "the economy", "free speech", "the family"],
    )

    def fill(template, fills):
        import re
        result = template
        for key, options in fills.items():
            result = result.replace("{" + key + "}", np.random.choice(options))
        return result

    real_rows = [{"text": fill(np.random.choice(real_templates), real_fills), "label": LABEL_REAL}
                 for _ in range(n_real)]
    fake_rows = [{"text": fill(np.random.choice(fake_templates), fake_fills), "label": LABEL_FAKE}
                 for _ in range(n_fake)]

    df = pd.DataFrame(real_rows + fake_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Generated sample dataset: %d rows (%d real, %d fake)", len(df), n_real, n_fake)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataset(path: str) -> dict:
    """
    Validate a dataset CSV and return a health report.

    Args:
        path: CSV file path

    Returns:
        {
          "valid": bool,
          "rows": int,
          "label_distribution": {"REAL": int, "FAKE": int},
          "missing_text": int,
          "avg_text_length": float,
          "issues": [str]
        }
    """
    issues = []
    df     = pd.read_csv(path)

    if "text" not in df.columns:
        issues.append("Missing 'text' column")
    if "label" not in df.columns:
        issues.append("Missing 'label' column")

    if issues:
        return {"valid": False, "issues": issues}

    missing_text = df["text"].isna().sum()
    if missing_text > 0:
        issues.append(f"{missing_text} rows have missing text")

    invalid_labels = (~df["label"].isin([0, 1])).sum()
    if invalid_labels > 0:
        issues.append(f"{invalid_labels} rows have invalid labels (expected 0 or 1)")

    label_dist = {
        "REAL": int((df["label"] == 0).sum()),
        "FAKE": int((df["label"] == 1).sum()),
    }
    imbalance = max(label_dist.values()) / max(min(label_dist.values()), 1)
    if imbalance > 5:
        issues.append(f"Severe class imbalance ({imbalance:.1f}:1) — consider oversampling")

    return {
        "valid":              len(issues) == 0,
        "rows":               len(df),
        "label_distribution": label_dist,
        "missing_text":       int(missing_text),
        "avg_text_length":    round(float(df["text"].dropna().str.len().mean()), 1),
        "issues":             issues,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens dataset utilities")
    sub    = parser.add_subparsers(dest="cmd")

    p_gen = sub.add_parser("generate", help="Generate a sample dataset for testing")
    p_gen.add_argument("--out",  default="data/dataset.csv", help="Output CSV path")
    p_gen.add_argument("--real", type=int, default=100, help="Number of real samples")
    p_gen.add_argument("--fake", type=int, default=100, help="Number of fake samples")

    p_val = sub.add_parser("validate", help="Validate a dataset CSV")
    p_val.add_argument("path", help="Path to dataset CSV")

    args = parser.parse_args()

    if args.cmd == "generate":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df = generate_sample_dataset(n_real=args.real, n_fake=args.fake)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} rows to {args.out}")

    elif args.cmd == "validate":
        report = validate_dataset(args.path)
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()
