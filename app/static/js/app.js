/**
 * COVID-19 Sentiment AI - Client Application Logic
 * Integrates Chart.js visualizations, real-time ML inference,
 * Multi-Model Battle Arena, Text X-Ray, and interactive EDA analytics.
 */

// Global State
const state = {
  activeTab: 'predict-view',
  predictorMode: 'single', // 'single' or 'arena'
  edaData: null,
  benchmarkData: null,
  sampleTweets: [],
  charts: {},
  activeSentimentDistMode: 'five_class',
  activeHashtagFilter: 'Overall',
  activeWordFilter: 'All',
  activeMetricFilter: 'accuracy',
  batchFilter: 'all',
  batchSearchQuery: '',
  lastBatchResults: []
};

// Pure Web Audio API Sound Synthesizer (No external mp3 assets needed)
const soundEngine = {
  audioCtx: null,
  muted: localStorage.getItem('covid_sfx_muted') === 'true',
  init() {
    const btn = document.getElementById('sound-toggle-btn');
    if (!btn) return;
    this.updateUI();
    btn.addEventListener('click', () => {
      this.muted = !this.muted;
      localStorage.setItem('covid_sfx_muted', this.muted);
      this.updateUI();
      if (!this.muted) {
        this.playChime(587.33, 'triangle', 0.12, 0.08); // D5 chime
        showToast('Sound Effects Enabled 🔊', 'info');
      } else {
        showToast('Sound Effects Muted 🔇', 'info');
      }
    });
  },
  updateUI() {
    const btn = document.getElementById('sound-toggle-btn');
    const label = document.getElementById('sound-status-label');
    if (!btn) return;
    if (this.muted) {
      btn.classList.add('muted');
      if (label) label.textContent = 'SFX: OFF';
    } else {
      btn.classList.remove('muted');
      if (label) label.textContent = 'SFX: ON';
    }
  },
  playChime(freq = 440, type = 'sine', duration = 0.12, gainVal = 0.06) {
    if (this.muted) return;
    try {
      if (!this.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.audioCtx = new AudioContext();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
      gain.gain.setValueAtTime(gainVal, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch (e) {
      // Audio context policy safe fallback
    }
  },
  playPredictSound(sentiment) {
    if (this.muted) return;
    if (sentiment.includes('Positive')) {
      // Upbeat C-Major Harmonic Triad
      this.playChime(523.25, 'sine', 0.12, 0.08); // C5
      setTimeout(() => this.playChime(659.25, 'sine', 0.14, 0.08), 80); // E5
      setTimeout(() => this.playChime(783.99, 'sine', 0.22, 0.09), 160); // G5
    } else if (sentiment.includes('Negative')) {
      // Deep ominous minor tone
      this.playChime(220.00, 'triangle', 0.25, 0.08); // A3
      setTimeout(() => this.playChime(207.65, 'sawtooth', 0.35, 0.05), 100); // G#3
    } else {
      // Balanced neutral double chime
      this.playChime(587.33, 'sine', 0.12, 0.06); // D5
      setTimeout(() => this.playChime(880.00, 'sine', 0.16, 0.06), 90); // A5
    }
  },
  playThemeSwitch() {
    this.playChime(659.25, 'triangle', 0.12, 0.08);
    setTimeout(() => this.playChime(880.00, 'sine', 0.15, 0.08), 70);
  },
  playClick() {
    this.playChime(950, 'sine', 0.035, 0.03);
  }
};

// Live Neural Waveform Equalizer Simulation
const neuralWaveformEngine = {
  intervalId: null,
  startDancing() {
    const bars = document.querySelectorAll('.wave-bar');
    const hzEl = document.getElementById('waveform-hz');
    const neuralStatus = document.getElementById('neural-status-tag');
    if (neuralStatus) {
      neuralStatus.innerHTML = `<span class="neural-pulse-dot"></span> INFERENCING...`;
    }
    if (this.intervalId) clearInterval(this.intervalId);

    this.intervalId = setInterval(() => {
      bars.forEach(bar => {
        const h = Math.floor(Math.random() * 13) + 3;
        bar.style.height = `${h}px`;
      });
      if (hzEl) {
        hzEl.textContent = `${(Math.random() * 60 + 20).toFixed(1)} Hz`;
      }
    }, 70);
  },
  settle(sentiment, confidence = 85.0) {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    const bars = document.querySelectorAll('.wave-bar');
    const hzEl = document.getElementById('waveform-hz');
    const neuralStatus = document.getElementById('neural-status-tag');

    if (neuralStatus) {
      neuralStatus.innerHTML = `<span class="neural-pulse-dot"></span> NEURAL READY`;
    }

    const profiles = {
      positive: [4, 7, 12, 16, 14, 11, 8, 12, 15, 13, 8, 5],
      negative: [15, 13, 10, 8, 5, 4, 6, 8, 11, 14, 15, 12],
      neutral: [6, 8, 10, 11, 9, 7, 8, 10, 11, 9, 7, 5]
    };

    let key = 'neutral';
    if (sentiment.toLowerCase().includes('positive')) key = 'positive';
    else if (sentiment.toLowerCase().includes('negative')) key = 'negative';

    const heights = profiles[key];
    bars.forEach((bar, idx) => {
      bar.style.height = `${heights[idx % heights.length]}px`;
    });

    if (hzEl) {
      hzEl.textContent = `${confidence.toFixed(1)}% CAL`;
    }
  }
};

// Ambient Vibe Switcher (Cyber / Nebula / Solar)
function initThemeSwitcher() {
  const dots = document.querySelectorAll('.theme-dot');
  const savedTheme = localStorage.getItem('sentinel_theme') || 'theme-cyber';
  applyTheme(savedTheme);

  dots.forEach(dot => {
    dot.addEventListener('click', () => {
      const theme = dot.getAttribute('data-theme');
      applyTheme(theme);
      localStorage.setItem('sentinel_theme', theme);
      soundEngine.playThemeSwitch();

      const themeNames = {
        'theme-cyber': 'Cyber Matrix (Emerald)',
        'theme-nebula': 'Deep Nebula (Violet)',
        'theme-solar': 'Solar Flare (Amber)'
      };
      showToast(`Vibe Switch: ${themeNames[theme] || theme} 🌌`, 'info');
    });
  });

  function applyTheme(theme) {
    document.body.classList.remove('theme-cyber', 'theme-nebula', 'theme-solar');
    document.body.classList.add(theme);

    dots.forEach(d => {
      d.classList.toggle('active', d.getAttribute('data-theme') === theme);
    });
  }
}

// Living HTML5 Canvas Neural Mesh Particle Network
function initNeuralMeshCanvas() {
  const canvas = document.getElementById('neural-mesh-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particleCount = Math.min(Math.floor(width * 0.045), 75);
  const particles = [];
  const mouse = { x: null, y: null, radius: 170 };

  window.addEventListener('pointermove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  }, { passive: true });

  window.addEventListener('pointerleave', () => {
    mouse.x = null;
    mouse.y = null;
  });

  function getThemeColors() {
    if (document.body.classList.contains('theme-nebula')) {
      return { node: 'rgba(168, 85, 247, 0.8)', line: '168, 85, 247', glow: '#A855F7' };
    } else if (document.body.classList.contains('theme-solar')) {
      return { node: 'rgba(245, 158, 11, 0.8)', line: '245, 158, 11', glow: '#F59E0B' };
    }
    return { node: 'rgba(16, 185, 129, 0.8)', line: '16, 185, 129', glow: '#10B981' };
  }

  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.75;
      this.vy = (Math.random() - 0.5) * 0.75;
      this.radius = Math.random() * 1.8 + 1.2;
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;

      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;

      if (mouse.x !== null && mouse.y !== null) {
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < mouse.radius && dist > 10) {
          const force = (1 - dist / mouse.radius) * 0.025;
          this.x += dx * force;
          this.y += dy * force;
        }
      }
    }

    draw(colors) {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = colors.node;
      ctx.shadowColor = colors.glow;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    const colors = getThemeColors();

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 125) {
          const alpha = (1 - dist / 125) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${colors.line}, ${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }

    particles.forEach(p => {
      p.update();
      p.draw(colors);
    });

    requestAnimationFrame(animate);
  }

  animate();
}

// 3D Perspective Card Tilt & Specular Glare Tracker
function init3DCardTilt() {
  const cards = document.querySelectorAll('.glass-card');
  cards.forEach(card => {
    card.addEventListener('pointermove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);

      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const tiltX = -((y - centerY) / centerY) * 4;
      const tiltY = ((x - centerX) / centerX) * 4;

      card.style.transform = `perspective(1200px) rotateX(${tiltX.toFixed(2)}deg) rotateY(${tiltY.toFixed(2)}deg) translateY(-2px)`;
    });

    card.addEventListener('pointerleave', () => {
      card.style.transform = 'perspective(1200px) rotateX(0deg) rotateY(0deg) translateY(0)';
    });
  });
}

// Toast Notifications System
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  let icon = 'ℹ️';
  if (type === 'success') icon = '✨';
  if (type === 'warn') icon = '⚠️';
  if (type === 'error') icon = '🚨';
  
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  container.appendChild(toast);
  
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 320);
  }, 3200);
}

// CountUp Animated Number Engine
function animateCountUp(elementId, target, suffix = '', duration = 1100) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const startTime = performance.now();
  const isInt = Number.isInteger(target);
  
  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = target * ease;
    
    if (isInt) {
      el.textContent = `${Math.round(current).toLocaleString()}${suffix}`;
    } else {
      el.textContent = `${current.toFixed(1)}${suffix}`;
    }
    
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  requestAnimationFrame(update);
}

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
  soundEngine.init();
  initThemeSwitcher();
  initNeuralMeshCanvas();
  init3DCardTilt();
  initNavigation();
  initPredictorControls();
  initBatchControls();
  initAmbientMouseTracker();
  fetchInitialData();
});

// Ambient Mouse-Tracker Glow for subtle desktop interactive light
function initAmbientMouseTracker() {
  const glow1 = document.querySelector('.glow-1');
  const glow3 = document.querySelector('.glow-3');
  if (!glow1 || !glow3) return;
  
  window.addEventListener('pointermove', (e) => {
    const x = e.clientX;
    const y = e.clientY;
    requestAnimationFrame(() => {
      glow1.style.transform = `translate(${x * 0.05}px, ${y * 0.05}px)`;
      glow3.style.transform = `translate(${x * -0.03}px, ${y * -0.03}px)`;
    });
  }, { passive: true });
}

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
   View 1: Real-Time Sentiment Predictor & Arena Logic
   ========================================================================== */
function initPredictorControls() {
  const tweetInput = document.getElementById('tweet-input');
  const charCount = document.getElementById('char-count');
  const clearBtn = document.getElementById('clear-input-btn');
  const predictBtn = document.getElementById('predict-submit-btn');
  const pipelineToggle = document.getElementById('pipeline-toggle-btn');
  const pipelineContent = document.getElementById('pipeline-content');
  const btnSingleMode = document.getElementById('btn-mode-single');
  const btnArenaMode = document.getElementById('btn-mode-arena');
  const btnSurprise = document.getElementById('btn-surprise-me');
  const btnCopyPrediction = document.getElementById('btn-copy-prediction');

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
    soundEngine.playClick();
  });

  // Mode Switcher (Single vs Arena)
  btnSingleMode.addEventListener('click', () => setPredictorMode('single'));
  btnArenaMode.addEventListener('click', () => setPredictorMode('arena'));

  function setPredictorMode(mode) {
    state.predictorMode = mode;
    soundEngine.playClick();
    btnSingleMode.classList.toggle('active', mode === 'single');
    btnArenaMode.classList.toggle('active', mode === 'arena');

    const singleGrid = document.getElementById('single-predictor-grid');
    const arenaContainer = document.getElementById('arena-container');
    const btnText = document.getElementById('predict-btn-text');

    if (mode === 'single') {
      singleGrid.style.display = 'grid';
      arenaContainer.style.display = 'none';
      btnText.textContent = 'Run Sentiment Prediction';
      showToast('Switched to Single Model Deep Dive', 'info');
    } else {
      singleGrid.style.display = 'grid';
      arenaContainer.style.display = 'flex';
      btnText.textContent = '⚔️ Run Multi-Model Arena Battle';
      showToast('Multi-Model Arena Activated ⚔️', 'info');
    }
  }

  // Submit button
  predictBtn.addEventListener('click', () => handlePredictSubmit());

  // Keyboard shortcut: Ctrl + Enter
  tweetInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handlePredictSubmit();
    }
  });

  // Surprise Me (Typewriter Tweet Stream)
  btnSurprise.addEventListener('click', () => {
    soundEngine.playChime(659.25, 'triangle', 0.1, 0.08);
    const tweetBank = [
      "Incredible gratitude to all supermarket workers and healthcare heroes stocking shelves and saving lives during COVID-19! #Heroes",
      "Stores have established special priority morning shopping hours for seniors and vulnerable customers. Super helpful!",
      "Disgusting price gouging online and in local shops. $40 for generic hand sanitizer is outright robbery during a pandemic.",
      "Complete panic buying madness at the store today. Total shortage of toilet paper, pasta, and canned beans.",
      "Local supermarket update: normal operations continue, pharmacy remains open 8am to 8pm daily with social distancing rules.",
      "Feeling anxious and scared about quarantine isolation, but staying home to protect our community and loved ones. #StaySafe",
      "Public health announcement: please wash hands with soap for at least 20 seconds and maintain 6ft distance in public markets."
    ];
    const randomTweet = tweetBank[Math.floor(Math.random() * tweetBank.length)];
    typewriterEffect(tweetInput, randomTweet, 15, () => {
      handlePredictSubmit();
    });
  });

  // Copy Prediction Result
  if (btnCopyPrediction) {
    btnCopyPrediction.addEventListener('click', () => {
      const text = tweetInput.value.trim();
      const sentiment = document.getElementById('sentiment-result-text').textContent;
      const conf = document.getElementById('confidence-val').textContent;
      if (!text || sentiment === 'Ready for Input') {
        showToast('Run an analysis first to copy results.', 'warn');
        return;
      }
      const snippet = `[COVID-19 Sentiment AI]\nTweet: "${text}"\nResult: ${sentiment} (${conf} confidence)`;
      navigator.clipboard.writeText(snippet).then(() => {
        soundEngine.playClick();
        showToast('Prediction copied to clipboard! 📋', 'success');
      });
    });
  }

  // Pipeline Accordion Toggle
  pipelineToggle.addEventListener('click', () => {
    pipelineToggle.classList.toggle('open');
    pipelineContent.classList.toggle('open');
  });
}

function handlePredictSubmit() {
  if (state.predictorMode === 'arena') {
    runArenaBattle();
  } else {
    runPrediction();
  }
}

// Typewriter Simulation Function
function typewriterEffect(inputElement, text, speed = 15, onComplete = null) {
  inputElement.value = '';
  let i = 0;
  function type() {
    if (i < text.length) {
      inputElement.value += text.charAt(i);
      document.getElementById('char-count').textContent = `${inputElement.value.length} characters`;
      i++;
      setTimeout(type, speed);
    } else if (onComplete) {
      onComplete();
    }
  }
  type();
}

function populateSampleChips(samples) {
  const container = document.getElementById('sample-chips-container');
  if (!container || !samples.length) return;

  container.innerHTML = '';
  samples.forEach((sample) => {
    const btn = document.createElement('button');
    btn.className = `chip-btn ${sample.sentiment === 'Positive' ? 'chip-pos' : sample.sentiment === 'Negative' ? 'chip-neg' : 'chip-neu'}`;
    btn.textContent = `${sample.category} • "${sample.text.substring(0, 22)}..."`;
    btn.title = sample.text;
    btn.addEventListener('click', () => {
      soundEngine.playClick();
      const input = document.getElementById('tweet-input');
      input.value = sample.text;
      document.getElementById('char-count').textContent = `${sample.text.length} characters`;
      handlePredictSubmit();
    });
    container.appendChild(btn);
  });
}

async function runPrediction() {
  const tweetInput = document.getElementById('tweet-input');
  const text = tweetInput.value.trim();

  if (!text) {
    tweetInput.focus();
    showToast('Please type or select a tweet to analyze.', 'warn');
    return;
  }

  const modelSelect = document.getElementById('model-select');
  const modeSelect = document.getElementById('mode-select');
  const predictBtn = document.getElementById('predict-submit-btn');

  const origBtnContent = predictBtn.innerHTML;
  predictBtn.disabled = true;
  predictBtn.innerHTML = `<span>Running Neural Inference...</span>`;

  // Start animated frequency equalizer
  neuralWaveformEngine.startDancing();
  const startTime = performance.now();

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
    const clientLatency = Math.round(performance.now() - startTime);
    renderPredictionResult(result, clientLatency);
    soundEngine.playPredictSound(result.predicted_sentiment);

  } catch (err) {
    console.error('Prediction failed:', err);
    neuralWaveformEngine.settle('neutral', 0);
    showToast('Prediction error. Check backend connection.', 'error');
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = origBtnContent;
  }
}

function renderPredictionResult(data, latencyMs = 12) {
  const sentimentHero = document.getElementById('sentiment-hero');
  const sentimentText = document.getElementById('sentiment-result-text');
  const confidenceVal = document.getElementById('confidence-val');
  const iconBox = document.getElementById('sentiment-icon-box');
  const activeModelTag = document.getElementById('active-model-tag');
  const latencyPill = document.getElementById('single-latency-pill');

  activeModelTag.textContent = data.model_used;
  if (latencyPill) latencyPill.textContent = `Latency: ${latencyMs}ms`;

  sentimentHero.classList.remove('state-positive', 'state-negative', 'state-neutral');

  const sentiment = data.predicted_sentiment;
  sentimentText.textContent = sentiment;
  confidenceVal.textContent = `${data.confidence}%`;

  let iconSvg = '';
  if (sentiment.toLowerCase().includes('positive')) {
    sentimentHero.classList.add('state-positive');
    iconSvg = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`;
  } else if (sentiment.toLowerCase().includes('negative')) {
    sentimentHero.classList.add('state-negative');
    iconSvg = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>`;
  } else {
    sentimentHero.classList.add('state-neutral');
    iconSvg = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`;
  }
  iconBox.innerHTML = iconSvg;

  // Settle waveform equalizer into calibrated state
  neuralWaveformEngine.settle(sentiment, data.confidence);

  // Update SVG Circular Gauge & Orbital Reactor
  updateCircularGauge(data.confidence, sentiment);

  // Render Probability Bars
  renderProbabilityBars(data.probabilities, data.mode);

  // Render Text X-Ray Word Attributions
  renderTextXRay(data.attributions);

  // Render Emotion Dimensions Radar Chart
  if (data.emotions) {
    renderEmotionRadar(data.emotions);
  }

  // Render Extracted Sentiment Keywords
  renderSentimentKeywords(data.sentiment_keywords);

  // Render Preprocessing Pipeline Steps
  renderPipelineSteps(data.pipeline_steps);
}

function updateCircularGauge(confidence, sentiment) {
  const circle = document.getElementById('gauge-fill-circle');
  const verdictDesc = document.getElementById('sentiment-verdict-desc');
  const plasmaCore = document.getElementById('plasma-core');
  if (!circle) return;

  const circumference = 301.59; // 2 * PI * 48
  const offset = circumference - (confidence / 100) * circumference;
  circle.style.strokeDashoffset = offset;

  circle.classList.remove('pos', 'neu', 'neg');
  if (sentiment.includes('Positive')) {
    circle.classList.add('pos');
    if (plasmaCore) {
      plasmaCore.style.background = 'radial-gradient(circle, rgba(16, 185, 129, 0.65) 0%, transparent 70%)';
    }
    if (verdictDesc) verdictDesc.textContent = `High positive sentiment detected. Reflects gratitude for frontline workers, community solidarity, or pandemic relief optimism.`;
  } else if (sentiment.includes('Negative')) {
    circle.classList.add('neg');
    if (plasmaCore) {
      plasmaCore.style.background = 'radial-gradient(circle, rgba(244, 63, 94, 0.65) 0%, transparent 70%)';
    }
    if (verdictDesc) verdictDesc.textContent = `Negative sentiment detected. Captures pandemic anxiety, supermarket panic buying, price gouging, or supply shortages.`;
  } else {
    circle.classList.add('neu');
    if (plasmaCore) {
      plasmaCore.style.background = 'radial-gradient(circle, rgba(6, 182, 212, 0.65) 0%, transparent 70%)';
    }
    if (verdictDesc) verdictDesc.textContent = `Neutral / Factual observation. Statements regarding store hours, supply chain logistics, or public health guidelines.`;
  }
}

function renderTextXRay(attributions) {
  const container = document.getElementById('xray-container');
  if (!container) return;

  if (!attributions || !attributions.length) {
    container.innerHTML = `<span class="xray-placeholder">No distinctive polarity weights detected for this text.</span>`;
    return;
  }

  container.innerHTML = '';
  attributions.forEach(token => {
    if (!token.is_word || token.polarity === 'none') {
      container.appendChild(document.createTextNode((token.is_word ? ' ' : '') + token.text));
    } else {
      const span = document.createElement('span');
      span.className = `xray-token token-${token.polarity}`;
      span.textContent = token.text;

      const sign = token.polarity === 'positive' ? '+' : (token.polarity === 'negative' ? '-' : '~');
      const hud = document.createElement('span');
      hud.className = 'token-score-hud';
      hud.textContent = `${token.polarity.toUpperCase()}: ${sign}${token.score}`;
      span.appendChild(hud);

      container.appendChild(document.createTextNode(' '));
      container.appendChild(span);
    }
  });
}

function renderEmotionRadar(emotions) {
  const canvas = document.getElementById('emotionRadarChart');
  if (!canvas) return;

  const labels = Object.keys(emotions);
  const values = Object.values(emotions);

  if (state.charts.emotionRadar) {
    state.charts.emotionRadar.data.datasets[0].data = values;
    state.charts.emotionRadar.update();
    return;
  }

  state.charts.emotionRadar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Pandemic Nuance Level',
        data: values,
        backgroundColor: 'rgba(99, 102, 241, 0.25)',
        borderColor: '#818cf8',
        borderWidth: 2,
        pointBackgroundColor: '#c084fc',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 1.5,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { display: false, stepSize: 25 },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.12)' },
          pointLabels: {
            font: { family: "'Inter', sans-serif", size: 10.5, weight: '600' },
            color: '#CBD5E1'
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          callbacks: {
            label: (ctx) => ` Nuance Score: ${ctx.raw}/100`
          }
        }
      }
    }
  });
}

async function runArenaBattle() {
  const tweetInput = document.getElementById('tweet-input');
  const text = tweetInput.value.trim();

  if (!text) {
    tweetInput.focus();
    showToast('Enter a tweet to launch the arena battle.', 'warn');
    return;
  }

  const modeSelect = document.getElementById('mode-select');
  const predictBtn = document.getElementById('predict-submit-btn');

  const origBtnContent = predictBtn.innerHTML;
  predictBtn.disabled = true;
  predictBtn.innerHTML = `<span>⚔️ Running 3-Model Battle...</span>`;

  // Start animated frequency equalizer
  neuralWaveformEngine.startDancing();

  try {
    const res = await fetch('/api/predict-arena', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        mode: modeSelect.value
      })
    });

    if (!res.ok) throw new Error('Arena battle failed');

    const data = await res.json();
    renderArenaOutput(data);
    neuralWaveformEngine.settle(data.winning_sentiment, 92.5);
    soundEngine.playPredictSound(data.winning_sentiment);
    showToast(`Arena Verdict: ${data.consensus_summary} ⚖️`, 'success');

  } catch (err) {
    console.error('Arena battle failed:', err);
    neuralWaveformEngine.settle('neutral', 0);
    showToast('Arena battle failed. Check backend.', 'error');
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = origBtnContent;
  }
}

function renderArenaOutput(data) {
  // Consensus Banner
  const banner = document.getElementById('arena-consensus-banner');
  const icon = document.getElementById('consensus-icon');
  const statusBadge = document.getElementById('consensus-status-badge');
  const modeBadge = document.getElementById('arena-mode-badge');
  const title = document.getElementById('arena-consensus-title');
  const desc = document.getElementById('arena-consensus-desc');

  statusBadge.className = `consensus-badge ${data.consensus_status}`;
  statusBadge.textContent = data.consensus_status.toUpperCase();
  modeBadge.textContent = data.mode === 'binary' ? 'Binary Mode' : '3-Class Mode';
  title.textContent = data.consensus_summary;

  if (data.consensus_status === 'unanimous') {
    icon.textContent = '🌟';
    desc.textContent = `All 3 machine learning algorithms independently agreed on "${data.winning_sentiment}" sentiment with high confidence.`;
  } else if (data.consensus_status === 'majority') {
    icon.textContent = '⚖️';
    desc.textContent = `2 out of 3 algorithms favored "${data.winning_sentiment}", indicating slight model divergence on edge vocabulary.`;
  } else {
    icon.textContent = '⚡';
    desc.textContent = `Split decision across classifiers. Nuanced sentiment with borderline probability weights.`;
  }

  // Model Cards (SGD, LR, NB)
  data.models.forEach(m => {
    let cardKey = m.model_id === 'logistic_regression' ? 'lr' : (m.model_id === 'naive_bayes' ? 'nb' : 'sgd');
    const predEl = document.getElementById(`arena-pred-${cardKey}`);
    const pctEl = document.getElementById(`arena-pct-${cardKey}`);
    const barEl = document.getElementById(`arena-bar-${cardKey}`);
    const latEl = document.getElementById(`arena-lat-${cardKey}`);

    if (predEl) {
      predEl.textContent = m.prediction;
      predEl.className = `arena-pred-val ${m.prediction.includes('Positive') ? 'text-emerald' : (m.prediction.includes('Negative') ? 'text-rose' : 'text-cyan')}`;
    }
    if (pctEl) pctEl.textContent = `${m.confidence}%`;
    if (barEl) {
      barEl.style.width = `${m.confidence}%`;
      barEl.className = `arena-conf-fill ${m.prediction.includes('Positive') ? 'fill-pos' : (m.prediction.includes('Negative') ? 'fill-neg' : 'fill-neu')}`;
    }
    if (latEl) latEl.textContent = `${m.latency_ms} ms`;

    // Mini probability breakdown
    const probPos = document.getElementById(`${cardKey}-prob-pos`);
    const probNeu = document.getElementById(`${cardKey}-prob-neu`);
    const probNeg = document.getElementById(`${cardKey}-prob-neg`);

    if (data.mode === 'binary') {
      if (probPos) probPos.textContent = `${m.probabilities['Positive / Neutral'] || '--'}%`;
      if (probNeu) probNeu.textContent = 'N/A';
      if (probNeg) probNeg.textContent = `${m.probabilities['Negative'] || '--'}%`;
    } else {
      if (probPos) probPos.textContent = `${m.probabilities['Positive'] || '--'}%`;
      if (probNeu) probNeu.textContent = `${m.probabilities['Neutral'] || '--'}%`;
      if (probNeg) probNeg.textContent = `${m.probabilities['Negative'] || '--'}%`;
    }
  });

  // Also update Radar, X-Ray, and Single Output to keep both in sync
  if (data.emotions) renderEmotionRadar(data.emotions);
  if (data.attributions) renderTextXRay(data.attributions);
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
  animateCountUp('kpi-total-tweets', summary.total_tweets);
  animateCountUp('kpi-locations', summary.unique_locations);
  animateCountUp('kpi-pos-pct', summary.positive_pct, '%');
  animateCountUp('kpi-neg-pct', summary.negative_pct, '%');
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
   View 5: Batch Tweet Analyzer & File Dropzone
   ========================================================================== */
function initBatchControls() {
  const batchInput = document.getElementById('batch-input');
  const btnRun = document.getElementById('btn-run-batch');
  const btnPreset = document.getElementById('btn-batch-preset');
  const btnExport = document.getElementById('btn-export-batch-csv');
  const btnCopyBatch = document.getElementById('btn-copy-batch-table');
  const dropzone = document.getElementById('file-dropzone');
  const fileInput = document.getElementById('batch-file-input');
  const browseBtn = document.getElementById('btn-browse-file');
  const searchInput = document.getElementById('batch-search-input');
  const filterPills = document.querySelectorAll('#batch-filter-pills .batch-filter-btn');

  // Preset tweets
  btnPreset.addEventListener('click', () => {
    soundEngine.playClick();
    const samples = [
      "Incredible gratitude to NHS doctors, nurses and grocery staff putting their lives on the line!",
      "Supermarkets announce special early morning hours for the elderly and disabled.",
      "Global food supplies are stable and store distribution is operating normally.",
      "Outrageous price gouging online for hand sanitizers and masks during a health crisis.",
      "Complete panic buying at the store today, shelves completely bare with zero canned goods.",
      "Vaccine trials are showing great early results across international research teams. Hope is on the horizon!",
      "Store management update: facial masks required for entry, maximum 30 shoppers allowed at once."
    ];
    batchInput.value = samples.join('\n');
    showToast('Loaded 7 pandemic test tweets into batch analyzer', 'info');
    runBatchAnalysis();
  });

  // Run batch button
  btnRun.addEventListener('click', () => {
    soundEngine.playClick();
    runBatchAnalysis();
  });

  // Export CSV
  btnExport.addEventListener('click', () => exportBatchCsv());

  // Copy batch table summary
  if (btnCopyBatch) {
    btnCopyBatch.addEventListener('click', () => {
      if (!state.lastBatchResults.length) {
        showToast('No batch results to copy. Run an analysis first.', 'warn');
        return;
      }
      let summary = `[COVID-19 Sentiment AI Batch Summary]\nTotal Analyzed: ${state.lastBatchResults.length}\n`;
      state.lastBatchResults.forEach((r, idx) => {
        summary += `${idx + 1}. [${r.sentiment} - ${r.confidence}%] ${r.text}\n`;
      });
      navigator.clipboard.writeText(summary).then(() => {
        soundEngine.playClick();
        showToast('Batch summary copied to clipboard! 📋', 'success');
      });
    });
  }

  // File Dropzone handlers
  if (dropzone && fileInput) {
    if (browseBtn) {
      browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        handleBatchFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length) {
        handleBatchFile(e.target.files[0]);
      }
    });
  }

  // Live Search filter
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      state.batchSearchQuery = searchInput.value.toLowerCase().trim();
      renderFilteredBatchTable();
    });
  }

  // Sentiment Filter pills
  filterPills.forEach(btn => {
    btn.addEventListener('click', () => {
      soundEngine.playClick();
      filterPills.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.batchFilter = btn.getAttribute('data-filter');
      renderFilteredBatchTable();
    });
  });
}

function handleBatchFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const content = e.target.result;
    const lines = content.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    
    // Check if CSV with header
    let extractedTweets = [];
    if (file.name.endsWith('.csv')) {
      const firstLine = lines[0].toLowerCase();
      let textColIdx = -1;
      const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
      
      headers.forEach((h, idx) => {
        if (h.includes('tweet') || h.includes('text') || h.includes('content') || h.includes('body')) {
          textColIdx = idx;
        }
      });

      const startIdx = textColIdx !== -1 ? 1 : 0;
      for (let i = startIdx; i < Math.min(lines.length, 120); i++) {
        if (textColIdx !== -1) {
          const cols = lines[i].split(',');
          if (cols[textColIdx]) {
            extractedTweets.push(cols[textColIdx].trim().replace(/^["']|["']$/g, ''));
          }
        } else {
          extractedTweets.push(lines[i].replace(/^["']|["']$/g, ''));
        }
      }
    } else {
      extractedTweets = lines.slice(0, 120);
    }

    if (extractedTweets.length) {
      document.getElementById('batch-input').value = extractedTweets.join('\n');
      showToast(`Loaded ${extractedTweets.length} tweets from "${file.name}" 📂`, 'success');
      runBatchAnalysis();
    } else {
      showToast('Could not extract valid text lines from file.', 'warn');
    }
  };
  reader.readAsText(file);
}

async function runBatchAnalysis() {
  const input = document.getElementById('batch-input').value.trim();
  if (!input) {
    showToast('Please enter or upload tweets to run batch analysis.', 'warn');
    return;
  }

  const lines = input.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (!lines.length) return;

  const btn = document.getElementById('btn-run-batch');
  btn.disabled = true;
  btn.innerHTML = `<span>Processing ${lines.length} tweets in parallel...</span>`;

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: lines })
    });

    if (!res.ok) throw new Error('Batch processing failed');

    const data = await res.json();
    state.lastBatchResults = data.results;
    renderBatchOutput(data);
    soundEngine.playPredictSound('Positive');
    showToast(`Successfully classified ${data.total_analyzed} tweets! ✨`, 'success');

  } catch (err) {
    console.error('Batch error:', err);
    showToast('Batch prediction error. Check server logs.', 'error');
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
  const posCount = dist.Positive ? dist.Positive.count : 0;
  const neuCount = dist.Neutral ? dist.Neutral.count : 0;
  const negCount = dist.Negative ? dist.Negative.count : 0;

  animateCountUp('batch-pos-count', posCount);
  document.getElementById('batch-pos-pct').textContent = dist.Positive ? `${dist.Positive.percentage}%` : '0%';

  animateCountUp('batch-neu-count', neuCount);
  document.getElementById('batch-neu-pct').textContent = dist.Neutral ? `${dist.Neutral.percentage}%` : '0%';

  animateCountUp('batch-neg-count', negCount);
  document.getElementById('batch-neg-pct').textContent = dist.Negative ? `${dist.Negative.percentage}%` : '0%';

  // Update filter pill counts
  const countAll = document.getElementById('count-filter-all');
  const countPos = document.getElementById('count-filter-pos');
  const countNeu = document.getElementById('count-filter-neu');
  const countNeg = document.getElementById('count-filter-neg');

  if (countAll) countAll.textContent = data.total_analyzed;
  if (countPos) countPos.textContent = posCount;
  if (countNeu) countNeu.textContent = neuCount;
  if (countNeg) countNeg.textContent = negCount;

  renderFilteredBatchTable();
}

function renderFilteredBatchTable() {
  const tbody = document.getElementById('batch-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  let filtered = state.lastBatchResults;

  // Sentiment Filter
  if (state.batchFilter && state.batchFilter !== 'all') {
    filtered = filtered.filter(item => item.sentiment.toLowerCase() === state.batchFilter.toLowerCase());
  }

  // Keyword Search
  if (state.batchSearchQuery) {
    filtered = filtered.filter(item => item.text.toLowerCase().includes(state.batchSearchQuery));
  }

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center py-6 text-muted">No tweets match the current filter or search criteria.</td></tr>`;
    return;
  }

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    const badgeClass = item.sentiment === 'Positive' ? 'pos' : item.sentiment === 'Negative' ? 'neg' : 'neu';

    tr.innerHTML = `
      <td>${item.id}</td>
      <td style="max-width: 480px; word-break: break-word;">${escapeHtml(item.text)}</td>
      <td><span class="sentiment-badge-sm ${badgeClass}">${item.sentiment}</span></td>
      <td><strong>${item.confidence}%</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag)
  );
}

function exportBatchCsv() {
  if (!state.lastBatchResults.length) {
    showToast('No batch results to export. Run an analysis first.', 'warn');
    return;
  }

  let csvContent = 'ID,Tweet,PredictedSentiment,Confidence\n';
  state.lastBatchResults.forEach(r => {
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
  soundEngine.playClick();
  showToast('CSV export downloaded successfully! 📊', 'success');
}
