/* =====================================================================
   SDNN Web UI — app.js
   Vanilla JS: drag-and-drop, file validation, fetch predict, render
   ===================================================================== */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const previewCard   = document.getElementById('previewCard');
const previewImg    = document.getElementById('previewImg');
const changeBtn     = document.getElementById('changeBtn');
const analyzeWrap   = document.getElementById('analyzeWrap');
const analyzeBtn    = document.getElementById('analyzeBtn');
const analyzeBtnText= document.getElementById('analyzeBtnText');
const btnSpinner    = document.getElementById('btnSpinner');
const errorToast    = document.getElementById('errorToast');
const errorMsg      = document.getElementById('errorMsg');
const toastClose    = document.getElementById('toastClose');
const resultsPanel  = document.getElementById('resultsPanel');

// Result elements
const resultEmoji     = document.getElementById('resultEmoji');
const resultClass     = document.getElementById('resultClass');
const resultContext   = document.getElementById('resultContext');
const reliabilityBadge= document.getElementById('reliabilityBadge');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar   = document.getElementById('confidenceBar');
const errorProbValue  = document.getElementById('errorProbValue');
const errorProbBar    = document.getElementById('errorProbBar');
const softmaxValue    = document.getElementById('softmaxValue');
const softmaxBar      = document.getElementById('softmaxBar');
const distChart       = document.getElementById('distChart');

// ── State ─────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Allowed types ─────────────────────────────────────────────────────
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/gif'];

// ── Helpers ───────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.hidden = false;
}
function hideError() { errorToast.hidden = true; }

function pct(val) {
  return (val * 100).toFixed(1) + '%';
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtnText.textContent = on ? 'Analyzing…' : 'Analyze with SDNN';
  btnSpinner.hidden = !on;
}

// Animated number counter (0 → target %)
function animateValue(el, target, suffix = '%', duration = 900) {
  const start = performance.now();
  const from = 0;
  const to = parseFloat((target * 100).toFixed(1));

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (to - from) * eased;
    el.textContent = current.toFixed(1) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── File selection ────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;

  if (!ALLOWED_TYPES.includes(file.type)) {
    showError(`Unsupported file type "${file.type}". Please upload a JPEG, PNG, WebP, or BMP image.`);
    return;
  }

  hideError();
  selectedFile = file;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  uploadZone.hidden = true;
  previewCard.hidden = false;
  analyzeWrap.hidden = false;
  resultsPanel.hidden = true;
}

// Click to browse
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// Drag and drop
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// Change image
changeBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewCard.hidden = true;
  analyzeWrap.hidden = true;
  resultsPanel.hidden = true;
  uploadZone.hidden = false;
  hideError();
});

// Close toast
toastClose.addEventListener('click', hideError);

// ── Predict ───────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  hideError();
  setLoading(true);

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);

    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Prediction failed. Please try again.');
      return;
    }

    renderResults(data);

  } catch (err) {
    showError('Could not reach the server. Is it running?');
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// ── Render Results ────────────────────────────────────────────────────
function renderResults(d) {
  // Hero card
  resultEmoji.textContent = d.emoji;
  resultClass.textContent = d.predicted_class;
  resultContext.textContent = d.context;

  if (d.reliability === 'reliable') {
    reliabilityBadge.className = 'reliability-badge badge-reliable';
    reliabilityBadge.textContent = '✅ Reliable';
  } else {
    reliabilityBadge.className = 'reliability-badge badge-unsafe';
    reliabilityBadge.textContent = '⚠️ Safe Fallback';
  }

  // Stat cards — show panel first so bars animate correctly
  resultsPanel.hidden = false;

  // Animate values with slight stagger
  setTimeout(() => {
    animateValue(confidenceValue, d.confidence);
    confidenceBar.style.width = pct(d.confidence);
  }, 80);

  setTimeout(() => {
    animateValue(errorProbValue, d.error_prob);
    errorProbBar.style.width = pct(d.error_prob);

    // Color confidence bar based on value
    if (d.confidence > 0.7) {
      confidenceBar.style.background = 'linear-gradient(90deg, #22c55e, #06b6d4)';
    } else if (d.confidence > 0.4) {
      confidenceBar.style.background = 'linear-gradient(90deg, #3d8ef8, #06b6d4)';
    } else {
      confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
    }

    // Color error bar based on threshold
    if (d.error_prob > d.threshold) {
      errorProbBar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
    } else {
      errorProbBar.style.background = 'linear-gradient(90deg, #22c55e, #06b6d4)';
    }
  }, 160);

  setTimeout(() => {
    animateValue(softmaxValue, d.max_softmax);
    softmaxBar.style.width = pct(d.max_softmax);
  }, 240);

  // Distribution chart
  renderDistChart(d.classes, d.softmax_probs, d.predicted_idx);

  // Scroll into view
  setTimeout(() => {
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function renderDistChart(classes, probs, topIdx) {
  // Sort by probability descending
  const items = classes.map((cls, i) => ({ cls, prob: probs[i], idx: i }));
  items.sort((a, b) => b.prob - a.prob);

  distChart.innerHTML = '';

  items.forEach((item, rank) => {
    const isTop = item.idx === topIdx;
    const delay = rank * 60; // ms stagger

    const row = document.createElement('div');
    row.className = 'dist-row';

    const label = document.createElement('span');
    label.className = 'dist-label' + (isTop ? ' is-top' : '');
    label.textContent = item.cls;

    const track = document.createElement('div');
    track.className = 'dist-track';

    const fill = document.createElement('div');
    fill.className = 'dist-fill' + (isTop ? ' is-top' : '');

    const pctLabel = document.createElement('span');
    pctLabel.className = 'dist-pct' + (isTop ? ' is-top' : '');
    pctLabel.textContent = (item.prob * 100).toFixed(1) + '%';

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pctLabel);
    distChart.appendChild(row);

    // Animate bar with delay
    setTimeout(() => {
      fill.style.width = pct(item.prob);
    }, delay);
  });
}
