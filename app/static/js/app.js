/**
 * COVID-19 Sentiment AI - Client Application Logic
 * Integrates Chart.js visualizations, real-time ML inference,
 * EDA analytics, and interactive model benchmarks.
 */

// Global State
const state = {
  activeTab: 'predict-view',
  edaData: null,
  benchmarkData: null,
  sampleTweets: [],
  charts: {},
  activeSentimentDistMode: 'five_class',
  activeHashtagFilter: 'Overall',
  activeWordFilter: 'All',
  activeMetricFilter: 'accuracy'
};

// Chart Theme Configuration
const chartTheme = {
  fontFamily: "'Inter', sans-serif",
  textColor: '#94A3B8',
  gridColor: 'rgba(255, 255, 255, 0.05)',
  colors: {
    positive: '#10B981',
    neutral: '#06B6D4',
    negative: '#F43F5E',
    extPositive: '#059669',
    extNegative: '#BE123C',
    accent: '#8B5CF6',
    indigo: '#6366F1',
    amber: '#F59E0B'
  }
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPredictorControls();
  initBatchControls();
  fetchInitialData();
});

/* ==========================================================================
   Navigation & Tabs
   ========================================================================== */
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetViewId = tab.getAttribute('data-tab');
      switchTab(targetViewId);
    });
  });

  // Handle URL hash navigation if present
  if (window.location.hash) {
    const hashView = window.location.hash.replace('#', '') + '-view';
    if (document.getElementById(hashView)) {
      switchTab(hashView);
    }
  }
}

function switchTab(viewId) {
  state.activeTab = viewId;

  // Update nav tab active states
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.getAttribute('data-tab') === viewId);
  });

  // Update view visibility
  document.querySelectorAll('.tab-view').forEach(v => {
    v.classList.toggle('active', v.id === viewId);
  });

  // Update URL hash
  const hashName = viewId.replace('-view', '');
  window.history.replaceState(null, null, `#${hashName}`);

  // Re-render / resize active charts to prevent canvas distortion
  setTimeout(() => {
    Object.values(state.charts).forEach(chart => {
      if (chart && typeof chart.resize === 'function') {
        chart.resize();
      }
    });
  }, 100);
}

/* ==========================================================================
   Initial Data Fetching
   ========================================================================== */
async function fetchInitialData() {
  try {
    // 1. Fetch EDA Stats
    const edaRes = await fetch('/api/eda-stats');
    if (edaRes.ok) {
      state.edaData = await edaRes.json();
      renderEdaSummary(state.edaData);
      renderSentimentDistChart();
      renderTimelineChart();
      renderLocationChart();
      renderIntegrityProfile();
      renderHashtagChart();
      renderWordCloud();
    }

    // 2. Fetch Model Benchmarks
    const benchRes = await fetch('/api/model-benchmark');
    if (benchRes.ok) {
      state.benchmarkData = await benchRes.json();
      renderLeaderboard();
      renderConfusionMatrix();
      renderModelBarChart();
    }

    // 3. Fetch Sample Tweets
    const samplesRes = await fetch('/api/samples');
    if (samplesRes.ok) {
      state.sampleTweets = await samplesRes.json();
      populateSampleChips(state.sampleTweets);
    }

  } catch (err) {
    console.error('Error fetching initial dataset stats:', err);
  }
}

/* ==========================================================================
   View 1: Real-Time Sentiment Predictor Logic
   ========================================================================== */
function initPredictorControls() {
  const tweetInput = document.getElementById('tweet-input');
  const charCount = document.getElementById('char-count');
  const clearBtn = document.getElementById('clear-input-btn');
  const predictBtn = document.getElementById('predict-submit-btn');
  const pipelineToggle = document.getElementById('pipeline-toggle-btn');
  const pipelineContent = document.getElementById('pipeline-content');

  // Character counter
  tweetInput.addEventListener('input', () => {
    const len = tweetInput.value.length;
    charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
  });

  // Clear button
  clearBtn.addEventListener('click', () => {
    tweetInput.value = '';
    charCount.textContent = '0 characters';
    tweetInput.focus();
  });

  // Submit button
  predictBtn.addEventListener('click', () => runPrediction());

  // Keyboard shortcut: Ctrl + Enter
  tweetInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runPrediction();
    }
  });

  // Pipeline Accordion Toggle
  pipelineToggle.addEventListener('click', () => {
    pipelineToggle.classList.toggle('open');
    pipelineContent.classList.toggle('open');
  });
}

function populateSampleChips(samples) {
  const container = document.getElementById('sample-chips-container');
  if (!container || !samples.length) return;

  container.innerHTML = '';
  samples.forEach((sample, idx) => {
    const btn = document.createElement('button');
    btn.className = `chip-btn ${sample.sentiment === 'Positive' ? 'chip-pos' : sample.sentiment === 'Negative' ? 'chip-neg' : 'chip-neu'}`;
    btn.textContent = `${sample.category} • "${sample.text.substring(0, 24)}..."`;
    btn.title = sample.text;
    btn.addEventListener('click', () => {
      const input = document.getElementById('tweet-input');
      input.value = sample.text;
      document.getElementById('char-count').textContent = `${sample.text.length} characters`;
      runPrediction();
    });
    container.appendChild(btn);
  });
}

async function runPrediction() {
  const tweetInput = document.getElementById('tweet-input');
  const text = tweetInput.value.trim();

  if (!text) {
    tweetInput.focus();
    return;
  }

  const modelSelect = document.getElementById('model-select');
  const modeSelect = document.getElementById('mode-select');
  const predictBtn = document.getElementById('predict-submit-btn');

  // Loading animation
  const origBtnContent = predictBtn.innerHTML;
  predictBtn.disabled = true;
  predictBtn.innerHTML = `<span>Analyzing Sentiment...</span>`;

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        model_type: modelSelect.value,
        mode: modeSelect.value
      })
    });

    if (!response.ok) {
      throw new Error(`Inference error: ${response.statusText}`);
    }

    const result = await response.json();
    renderPredictionResult(result);

  } catch (err) {
    console.error('Prediction failed:', err);
    alert('Prediction error. Please ensure the backend is running.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = origBtnContent;
  }
}

function renderPredictionResult(data) {
  const sentimentHero = document.getElementById('sentiment-hero');
  const sentimentText = document.getElementById('sentiment-result-text');
  const confidenceVal = document.getElementById('confidence-val');
  const iconBox = document.getElementById('sentiment-icon-box');
  const activeModelTag = document.getElementById('active-model-tag');

  activeModelTag.textContent = data.model_used;

  // Clean prior state classes
  sentimentHero.classList.remove('state-positive', 'state-negative', 'state-neutral');

  const sentiment = data.predicted_sentiment;
  sentimentText.textContent = sentiment;
  confidenceVal.textContent = `${data.confidence}%`;

  let iconSvg = '';
  if (sentiment.toLowerCase().includes('positive')) {
    sentimentHero.classList.add('state-positive');
    iconSvg = `<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`;
  } else if (sentiment.toLowerCase().includes('negative')) {
    sentimentHero.classList.add('state-negative');
    iconSvg = `<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>`;
  } else {
    sentimentHero.classList.add('state-neutral');
    iconSvg = `<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`;
  }
  iconBox.innerHTML = iconSvg;

  // Render Probability Bars
  renderProbabilityBars(data.probabilities, data.mode);

  // Render Extracted Sentiment Keywords
  renderSentimentKeywords(data.sentiment_keywords);

  // Render Preprocessing Pipeline Steps
  renderPipelineSteps(data.pipeline_steps);
}

function renderProbabilityBars(probs, mode) {
  const container = document.getElementById('probability-bars-container');
  container.innerHTML = '';

  for (const [cls, pct] of Object.entries(probs)) {
    const row = document.createElement('div');
    row.className = 'prob-row';

    let dotClass = 'dot-neu';
    let fillClass = 'fill-neu';
    const lowerCls = cls.toLowerCase();

    if (lowerCls.includes('pos')) {
      dotClass = 'dot-pos';
      fillClass = 'fill-pos';
    } else if (lowerCls.includes('neg')) {
      dotClass = 'dot-neg';
      fillClass = 'fill-neg';
    }

    row.innerHTML = `
      <div class="prob-header">
        <span class="prob-name"><span class="dot ${dotClass}"></span> ${cls}</span>
        <span class="prob-pct">${pct}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill ${fillClass}" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(row);
  }
}

function renderSentimentKeywords(keywords) {
  const container = document.getElementById('keyword-tags-container');
  container.innerHTML = '';

  if (!keywords || keywords.length === 0) {
    container.innerHTML = `<span class="no-keywords">No strong polarity keywords detected in vocabulary.</span>`;
    return;
  }

  keywords.forEach(kw => {
    const badge = document.createElement('span');
    const lower = kw.sentiment.toLowerCase();
    const styleClass = lower === 'positive' ? 'kw-pos' : lower === 'negative' ? 'kw-neg' : 'kw-neu';
    badge.className = `keyword-badge ${styleClass}`;
    badge.innerHTML = `
      <span>${kw.word}</span>
      <span style="opacity: 0.7; font-size: 0.7em;">(${kw.weight > 0 ? '+' : ''}${kw.weight})</span>
    `;
    container.appendChild(badge);
  });
}

function renderPipelineSteps(steps) {
  document.getElementById('step-url-mention').textContent = steps.no_mentions_url || 'None';
  document.getElementById('step-clean-text').textContent = steps.cleaned || 'None';

  const tokensContainer = document.getElementById('step-stemmed-tokens');
  tokensContainer.innerHTML = '';

  if (!steps.stemmed || steps.stemmed.length === 0) {
    tokensContainer.innerHTML = `<span class="token-empty">No tokens remaining after stopword removal.</span>`;
    return;
  }

  steps.stemmed.forEach(t => {
    const tok = document.createElement('span');
    tok.className = 'stem-token';
    tok.textContent = t;
    tokensContainer.appendChild(tok);
  });
}

/* ==========================================================================
   View 2: Exploratory Data Analysis (EDA) Visualizations
   ========================================================================== */
function renderEdaSummary(data) {
  const summary = data.summary;
  document.getElementById('header-total-tweets').textContent = `${summary.total_tweets.toLocaleString()} Tweets`;
  document.getElementById('kpi-total-tweets').textContent = summary.total_tweets.toLocaleString();
  document.getElementById('kpi-pos-pct').textContent = `${summary.positive_pct}%`;
  document.getElementById('kpi-neg-pct').textContent = `${summary.negative_pct}%`;
  document.getElementById('kpi-locations').textContent = summary.unique_locations.toLocaleString();
}

function renderSentimentDistChart() {
  const ctx = document.getElementById('sentimentDistChart').getContext('2d');
  const distData = state.edaData.sentiment_distribution[state.activeSentimentDistMode];

  // Button toggle states
  document.getElementById('btn-dist-5').onclick = () => { updateDistMode('five_class', 'btn-dist-5'); };
  document.getElementById('btn-dist-3').onclick = () => { updateDistMode('three_class', 'btn-dist-3'); };
  document.getElementById('btn-dist-bin').onclick = () => { updateDistMode('binary', 'btn-dist-bin'); };

  function updateDistMode(mode, activeBtnId) {
    state.activeSentimentDistMode = mode;
    ['btn-dist-5', 'btn-dist-3', 'btn-dist-bin'].forEach(id => {
      document.getElementById(id).classList.toggle('active', id === activeBtnId);
    });
    renderSentimentDistChart();
  }

  // Destroy previous instance
  if (state.charts.sentimentDist) {
    state.charts.sentimentDist.destroy();
  }

  // Define palette based on labels
  const palette = distData.labels.map(l => {
    const lower = l.toLowerCase();
    if (lower === 'extremely positive') return '#059669';
    if (lower === 'positive') return '#10B981';
    if (lower === 'neutral') return '#06B6D4';
    if (lower === 'negative') return '#F43F5E';
    if (lower === 'extremely negative') return '#BE123C';
    if (lower.includes('positive / neutral')) return '#10B981';
    return '#8B5CF6';
  });

  state.charts.sentimentDist = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: distData.labels,
      datasets: [{
        data: distData.values,
        backgroundColor: palette,
        borderColor: 'rgba(15, 23, 42, 0.9)',
        borderWidth: 3,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#F8FAFC',
          bodyColor: '#CBD5E1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} tweets (${distData.percentages[ctx.dataIndex]}%)`
          }
        }
      }
    }
  });

  // Render Custom Legend
  const legendContainer = document.getElementById('sentiment-custom-legend');
  legendContainer.innerHTML = '';
  distData.labels.forEach((label, idx) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <span class="legend-color-box" style="background-color: ${palette[idx]}"></span>
      <span>${label} (${distData.percentages[idx]}%)</span>
    `;
    legendContainer.appendChild(item);
  });
}

function renderTimelineChart() {
  const ctx = document.getElementById('timelineChart').getContext('2d');
  const timeline = state.edaData.timeline;

  if (state.charts.timeline) {
    state.charts.timeline.destroy();
  }

  // Create gradient background
  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, 'rgba(99, 102, 241, 0.45)');
  gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

  state.charts.timeline = new Chart(ctx, {
    type: 'line',
    data: {
      labels: timeline.labels,
      datasets: [{
        label: 'Daily Tweets',
        data: timeline.values,
        borderColor: '#818CF8',
        borderWidth: 2.5,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointBackgroundColor: '#C7D2FE',
        pointBorderColor: '#4F46E5',
        pointRadius: 3,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => ` Tweet Count: ${ctx.raw.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: chartTheme.gridColor },
          ticks: { color: chartTheme.textColor, maxRotation: 45, font: { size: 10 } }
        },
        y: {
          grid: { color: chartTheme.gridColor },
          ticks: { color: chartTheme.textColor }
        }
      }
    }
  });
}

function renderLocationChart() {
  const ctx = document.getElementById('locationChart').getContext('2d');
  const loc = state.edaData.locations;

  if (state.charts.location) {
    state.charts.location.destroy();
  }

  state.charts.location = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: loc.labels,
      datasets: [{
        label: 'Tweet Count',
        data: loc.values,
        backgroundColor: 'rgba(14, 165, 233, 0.65)',
        borderColor: '#38BDF8',
        borderWidth: 1.5,
        borderRadius: 6,
        hoverBackgroundColor: 'rgba(14, 165, 233, 0.9)'
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => ` ${ctx.raw.toLocaleString()} tweets from ${ctx.label}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: chartTheme.gridColor },
          ticks: { color: chartTheme.textColor }
        },
        y: {
          grid: { display: false },
          ticks: { color: chartTheme.textColor, font: { size: 11 } }
        }
      }
    }
  });
}

function renderIntegrityProfile() {
  const container = document.getElementById('integrity-grid-container');
  const missing = state.edaData.missing_data;
  container.innerHTML = '';

  missing.columns.forEach((col, idx) => {
    const card = document.createElement('div');
    card.className = 'integrity-card';
    const isComplete = missing.missing_counts[idx] === 0;

    card.innerHTML = `
      <div class="integrity-col-name">${col}</div>
      <span class="integrity-status-badge ${isComplete ? 'status-complete' : 'status-partial'}">
        ${isComplete ? '100% Complete' : `${missing.missing_pct[idx]}% Missing`}
      </span>
      <div class="integrity-detail">
        ${isComplete ? '0 Nulls' : `${missing.missing_counts[idx].toLocaleString()} Null values`}
      </div>
    `;
    container.appendChild(card);
  });
}

/* ==========================================================================
   View 3: Model Benchmarks Leaderboard
   ========================================================================== */
function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  const models = state.benchmarkData.leaderboard;
  tbody.innerHTML = '';

  // Handle metric sorting buttons
  ['filter-acc', 'filter-f1', 'filter-prec', 'filter-rec'].forEach(btnId => {
    const btn = document.getElementById(btnId);
    btn.onclick = () => {
      document.querySelectorAll('.metric-filter-group .pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const metricMap = {
        'filter-acc': 'accuracy',
        'filter-f1': 'f1_score',
        'filter-prec': 'precision',
        'filter-rec': 'recall'
      };
      state.activeMetricFilter = metricMap[btnId];
      sortAndDisplayLeaderboard();
    };
  });

  sortAndDisplayLeaderboard();

  function sortAndDisplayLeaderboard() {
    const sorted = [...models].sort((a, b) => b[state.activeMetricFilter] - a[state.activeMetricFilter]);
    tbody.innerHTML = '';

    sorted.forEach((m, idx) => {
      const tr = document.createElement('tr');
      if (m.highlight) tr.className = 'row-winner';

      const rankClass = idx === 0 ? 'rank-1' : idx === 1 ? 'rank-2' : idx === 2 ? 'rank-3' : '';

      tr.innerHTML = `
        <td><span class="rank-badge ${rankClass}">${idx + 1}</span></td>
        <td>
          <div class="model-name-strong">${m.model}</div>
          <span style="font-size: 0.75rem; color: var(--text-muted);">${m.type}</span>
        </td>
        <td><span class="badge-standard">${m.badge}</span></td>
        <td>
          <div class="score-bar-inline">
            <span class="score-text text-emerald">${m.accuracy}%</span>
            <div class="score-track-mini">
              <div class="score-fill-mini" style="width: ${m.accuracy}%"></div>
            </div>
          </div>
        </td>
        <td>${m.precision}%</td>
        <td>${m.recall}%</td>
        <td>${m.f1_score}%</td>
        <td>
          ${m.highlight ? '<span class="badge-winner">TOP PERFORMER 🏆</span>' : '<span class="badge-standard">Evaluated</span>'}
        </td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function renderConfusionMatrix() {
  const cm = state.benchmarkData.winner_confusion_matrix;
  document.getElementById('val-tn').textContent = cm.true_negative.toLocaleString();
  document.getElementById('val-fp').textContent = cm.false_positive.toLocaleString();
  document.getElementById('val-fn').textContent = cm.false_negative.toLocaleString();
  document.getElementById('val-tp').textContent = cm.true_positive.toLocaleString();
}

function renderModelBarChart() {
  const ctx = document.getElementById('modelBarChart').getContext('2d');
  const models = state.benchmarkData.leaderboard;

  if (state.charts.modelBar) {
    state.charts.modelBar.destroy();
  }

  // Sort descending by accuracy (matches notebook cell 184)
  const sorted = [...models].sort((a, b) => b.accuracy - a.accuracy);

  state.charts.modelBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(m => m.model),
      datasets: [{
        label: 'Test Accuracy (%)',
        data: sorted.map(m => m.accuracy),
        backgroundColor: sorted.map(m => m.highlight ? '#10B981' : 'rgba(99, 102, 241, 0.7)'),
        borderColor: sorted.map(m => m.highlight ? '#34D399' : '#818CF8'),
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          padding: 10,
          callbacks: {
            label: (ctx) => ` Accuracy: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          min: 70,
          max: 90,
          grid: { color: chartTheme.gridColor },
          ticks: { color: chartTheme.textColor, callback: (v) => `${v}%` }
        },
        y: {
          grid: { display: false },
          ticks: { color: chartTheme.textColor, font: { size: 10.5 } }
        }
      }
    }
  });
}

/* ==========================================================================
   View 4: Hashtags & Word Clouds
   ========================================================================== */
function renderHashtagChart() {
  const ctx = document.getElementById('hashtagChart').getContext('2d');
  const hashtags = state.edaData.hashtags;

  // Filter Buttons
  const filterContainer = document.getElementById('hashtag-filters');
  filterContainer.querySelectorAll('.pill-btn').forEach(btn => {
    btn.onclick = () => {
      filterContainer.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeHashtagFilter = btn.getAttribute('data-hashtag-filter');
      renderHashtagChart();
    };
  });

  if (state.charts.hashtag) {
    state.charts.hashtag.destroy();
  }

  const currentData = hashtags[state.activeHashtagFilter] || hashtags['Overall'];

  state.charts.hashtag = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: currentData.labels,
      datasets: [{
        label: 'Hashtag Count',
        data: currentData.values,
        backgroundColor: 'rgba(168, 85, 247, 0.65)',
        borderColor: '#C084FC',
        borderWidth: 1.5,
        borderRadius: 6,
        hoverBackgroundColor: 'rgba(168, 85, 247, 0.9)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          padding: 10,
          callbacks: {
            label: (ctx) => ` ${ctx.raw.toLocaleString()} occurrences`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: chartTheme.textColor, maxRotation: 45 }
        },
        y: {
          grid: { color: chartTheme.gridColor },
          ticks: { color: chartTheme.textColor }
        }
      }
    }
  });
}

function renderWordCloud() {
  const container = document.getElementById('word-cloud-container');
  const wordFreqs = state.edaData.word_frequencies;

  // Word filter buttons
  const filterContainer = document.getElementById('word-filters');
  filterContainer.querySelectorAll('.pill-btn').forEach(btn => {
    btn.onclick = () => {
      filterContainer.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeWordFilter = btn.getAttribute('data-word-filter');
      renderWordCloud();
    };
  });

  container.innerHTML = '';
  const list = wordFreqs[state.activeWordFilter] || wordFreqs['All'];
  if (!list.length) return;

  const maxWeight = Math.max(...list.map(w => w.weight));
  const minWeight = Math.min(...list.map(w => w.weight));

  list.forEach(item => {
    const pill = document.createElement('span');
    pill.className = 'word-pill';

    // Calculate proportional font size between 0.82rem and 1.6rem
    const norm = (item.weight - minWeight) / (maxWeight - minWeight || 1);
    const fontSize = 0.85 + norm * 0.75;

    // Dynamic color tint based on sentiment
    let bg = 'rgba(255, 255, 255, 0.05)';
    let color = '#CBD5E1';
    let border = 'rgba(255, 255, 255, 0.1)';

    if (state.activeWordFilter === 'Positive') {
      bg = 'rgba(16, 185, 129, 0.12)';
      color = '#6EE7B7';
      border = 'rgba(16, 185, 129, 0.3)';
    } else if (state.activeWordFilter === 'Negative') {
      bg = 'rgba(244, 63, 94, 0.12)';
      color = '#FDA4AF';
      border = 'rgba(244, 63, 94, 0.3)';
    } else if (state.activeWordFilter === 'Neutral') {
      bg = 'rgba(6, 182, 212, 0.12)';
      color = '#67E8F9';
      border = 'rgba(6, 182, 212, 0.3)';
    }

    pill.style.fontSize = `${fontSize}rem`;
    pill.style.backgroundColor = bg;
    pill.style.color = color;
    pill.style.border = `1px solid ${border}`;

    pill.innerHTML = `
      <span>${item.text}</span>
      <span class="word-weight">${item.weight.toLocaleString()}</span>
    `;
    container.appendChild(pill);
  });
}

/* ==========================================================================
   View 5: Batch Tweet Analyzer
   ========================================================================== */
function initBatchControls() {
  const batchInput = document.getElementById('batch-input');
  const btnRun = document.getElementById('btn-run-batch');
  const btnPreset = document.getElementById('btn-batch-preset');
  const btnExport = document.getElementById('btn-export-batch-csv');

  btnPreset.addEventListener('click', () => {
    const samples = [
      "Incredible gratitude to NHS doctors, nurses and grocery staff putting their lives on the line!",
      "Supermarkets announce special early morning hours for the elderly and disabled.",
      "Global food supplies are stable and store distribution is operating normally.",
      "Outrageous price gouging online for hand sanitizers and masks during a health crisis.",
      "Complete panic buying at the store today, shelves completely bare with zero canned goods."
    ];
    batchInput.value = samples.join('\n');
    runBatchAnalysis();
  });

  btnRun.addEventListener('click', () => runBatchAnalysis());
  btnExport.addEventListener('click', () => exportBatchCsv());
}

let lastBatchResults = [];

async function runBatchAnalysis() {
  const input = document.getElementById('batch-input').value.trim();
  if (!input) return;

  const lines = input.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (!lines.length) return;

  const btn = document.getElementById('btn-run-batch');
  btn.disabled = true;
  btn.innerHTML = `<span>Processing ${lines.length} tweets...</span>`;

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: lines })
    });

    if (!res.ok) throw new Error('Batch processing failed');

    const data = await res.json();
    lastBatchResults = data.results;
    renderBatchOutput(data);

  } catch (err) {
    console.error('Batch error:', err);
    alert('Batch prediction error.');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
      <span>Run Batch Sentiment Analysis</span>
    `;
  }
}

function renderBatchOutput(data) {
  document.getElementById('batch-total-analyzed').textContent = `${data.total_analyzed} Analyzed`;

  const dist = data.distribution;
  document.getElementById('batch-pos-count').textContent = dist.Positive ? dist.Positive.count : 0;
  document.getElementById('batch-pos-pct').textContent = dist.Positive ? `${dist.Positive.percentage}%` : '0%';

  document.getElementById('batch-neu-count').textContent = dist.Neutral ? dist.Neutral.count : 0;
  document.getElementById('batch-neu-pct').textContent = dist.Neutral ? `${dist.Neutral.percentage}%` : '0%';

  document.getElementById('batch-neg-count').textContent = dist.Negative ? dist.Negative.count : 0;
  document.getElementById('batch-neg-pct').textContent = dist.Negative ? `${dist.Negative.percentage}%` : '0%';

  // Render Table
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = '';

  data.results.forEach(item => {
    const tr = document.createElement('tr');
    const badgeClass = item.sentiment === 'Positive' ? 'pos' : item.sentiment === 'Negative' ? 'neg' : 'neu';

    tr.innerHTML = `
      <td>${item.id}</td>
      <td style="max-width: 480px; word-break: break-word;">${item.text}</td>
      <td><span class="sentiment-badge-sm ${badgeClass}">${item.sentiment}</span></td>
      <td><strong>${item.confidence}%</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function exportBatchCsv() {
  if (!lastBatchResults.length) {
    alert('No batch results to export. Run an analysis first.');
    return;
  }

  let csvContent = 'ID,Tweet,PredictedSentiment,Confidence\n';
  lastBatchResults.forEach(r => {
    const escapedText = `"${r.text.replace(/"/g, '""')}"`;
    csvContent += `${r.id},${escapedText},${r.sentiment},${r.confidence}%\n`;
  });

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `covid_sentiment_batch_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
