/**
 * TruthLens — app.js
 * Complete Working Version (Groq API Updated)
 */

// ── Example articles ───────────────────────────────────────────────

const EXAMPLES = [
  {
    label: "Clickbait",
    text: "SHOCKING SECRET the government doesn't want you to know: Scientists have PROVEN that chemtrails contain mind-control chemicals!!"
  },
  {
    label: "Health misinformation",
    text: "BREAKING: A miracle cure discovered deep in the Amazon rainforest can cure ALL cancers in just 3 days."
  },
  {
    label: "Science news",
    text: "NASA's Perseverance rover has collected its 23rd rock sample on Mars."
  },
  {
    label: "Political claim",
    text: "The president signed an executive order yesterday requiring all Americans to surrender vehicles by 2026."
  },
  {
    label: "Credible reporting",
    text: "The Federal Reserve raised interest rates by 25 basis points on Wednesday."
  }
];

// ── State ──────────────────────────────────────────────────────────

let analysisMode = "standard";
let historyData = [];
let totalAnalyzed = 0;

// ── Utilities ──────────────────────────────────────────────────────

function loadExample(idx) {

  const ta = document.getElementById("news-input");

  ta.value = EXAMPLES[idx].text;

  ta.classList.remove("error");

  ta.focus();
}

function setMode(el) {

  document
    .querySelectorAll(".mode-btn")
    .forEach(btn => btn.classList.remove("active"));

  el.classList.add("active");

  analysisMode = el.dataset.mode;
}

function handleUrlNote() {

  const url = document
    .getElementById("url-input")
    .value
    .trim();

  if (url) {

    document.getElementById("news-input").value =
      `[Article URL: ${url}]

Please analyze the credibility of this article.`;

    document.getElementById("url-input").value = "";
  }
}

// ── Main Analysis ──────────────────────────────────────────────────

async function analyzeNews() {

  const input = document.getElementById("news-input");

  const text = input.value.trim();

  if (!text || text.length < 15) {

    input.classList.add("error");

    input.focus();

    return;
  }

  const btn = document.getElementById("analyze-btn");

  btn.disabled = true;

  btn.innerHTML = "Analyzing...";

  const depthInstructions = {
    quick: "Give a short analysis.",
    standard: "Give a detailed 3 sentence analysis.",
    deep: "Give a deep detailed analysis."
  };

  const prompt = buildPrompt(
    text,
    depthInstructions[analysisMode]
  );

  try {

    const result = await callClaudeAPI(prompt);

    totalAnalyzed++;

    const liveCount =
      document.getElementById("live-count");

    if (liveCount) {
      liveCount.textContent = totalAnalyzed;
    }

    addToHistory(text, result);

    renderResult(result);

  } catch (err) {

    console.error(err);

    renderError(err.message);

  } finally {

    btn.disabled = false;

    btn.innerHTML = "Analyze credibility";
  }
}

// ── Prompt Builder ─────────────────────────────────────────────────

function buildPrompt(text, depthInstruction) {

  return `
You are TruthLens AI.

Analyze this news text carefully.

TEXT:
${text}

${depthInstruction}

Respond ONLY with valid JSON.

{
  "verdict":"REAL",
  "confidence":90,
  "summary":"text",
  "signals":{
    "emotional_language":20,
    "source_credibility":80,
    "factual_consistency":85,
    "clickbait_score":15,
    "logical_coherence":90,
    "bias_indicators":25
  },
  "red_flags":["flag"],
  "positive_signals":["signal"],
  "recommendation":"recommendation"
}
`;
}

// ── API CALL ───────────────────────────────────────────────────────

async function callClaudeAPI(prompt) {

  // Paste your Groq API key here
  const API_KEY = "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";

  const resp = await fetch(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
      },

      body: JSON.stringify({

        // UPDATED MODEL
        model: "llama-3.3-70b-versatile",

        messages: [
          {
            role: "user",
            content: prompt
          }
        ],

        temperature: 0.3,

        max_tokens: 1000
      })
    }
  );

  if (!resp.ok) {

    const errText = await resp.text();

    throw new Error(errText);
  }

  const data = await resp.json();

  const rawText =
    data.choices[0].message.content;

  const cleaned = rawText
    .replace(/```json/g, "")
    .replace(/```/g, "")
    .trim();

  return JSON.parse(cleaned);
}

// ── Themes ─────────────────────────────────────────────────────────

const THEMES = {

  REAL: {
    color: "#1D9E75",
    title: "Credible News"
  },

  FAKE: {
    color: "#E24B4A",
    title: "Likely Fake News"
  },

  UNCERTAIN: {
    color: "#BA7517",
    title: "Uncertain"
  }
};

// ── Render Result ──────────────────────────────────────────────────

function renderResult(r) {

  const area =
    document.getElementById("result-area");

  const theme =
    THEMES[r.verdict] || THEMES.UNCERTAIN;

  area.innerHTML = `
  
  <div style="
    padding:20px;
    border-radius:12px;
    background:white;
    margin-top:20px;
    border:2px solid ${theme.color};
  ">

    <h2 style="color:${theme.color}">
      ${theme.title}
    </h2>

    <h3>
      Confidence: ${r.confidence}%
    </h3>

    <p>
      ${r.summary}
    </p>

    <h4>Recommendation</h4>

    <p>
      ${r.recommendation}
    </p>

  </div>
  `;
}

// ── Render Error ───────────────────────────────────────────────────

function renderError(msg) {

  const area =
    document.getElementById("result-area");

  area.innerHTML = `
  
  <div style="
    background:#FCEBEB;
    color:#791F1F;
    padding:20px;
    border-radius:10px;
    margin-top:20px;
  ">

    <strong>Error:</strong>

    <br><br>

    ${msg}

  </div>
  `;
}

// ── History ────────────────────────────────────────────────────────

function addToHistory(text, result) {

  const short =
    text.slice(0, 60) + "...";

  historyData.unshift({
    short,
    verdict: result.verdict,
    confidence: result.confidence
  });

  if (historyData.length > 6) {
    historyData.pop();
  }

  renderHistory();
}

function renderHistory() {

  const list =
    document.getElementById("history-list");

  if (!list) return;

  if (!historyData.length) {

    list.innerHTML =
      "<div>No analyses yet.</div>";

    return;
  }

  list.innerHTML = historyData.map((h) => `
  
    <div style="
      padding:10px;
      margin-bottom:10px;
      border:1px solid #ddd;
      border-radius:8px;
    ">

      <strong>${h.verdict}</strong>

      (${h.confidence}%)

      <br><br>

      ${h.short}

    </div>

  `).join("");
}

// ── Reload History ────────────────────────────────────────────────

function reloadHistory(idx) {

  document
    .getElementById("news-input")
    .scrollIntoView({
      behavior: "smooth"
    });
}

// ── Init ───────────────────────────────────────────────────────────

renderHistory();

// ── Keyboard Shortcut ──────────────────────────────────────────────

const newsInput =
  document.getElementById("news-input");

if (newsInput) {

  newsInput.addEventListener(
    "keydown",
    function (e) {

      if (
        e.key === "Enter" &&
        (e.ctrlKey || e.metaKey)
      ) {
        analyzeNews();
      }
    }
  );
}
