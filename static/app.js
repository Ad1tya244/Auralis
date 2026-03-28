/* =====================================================================
   SDNN Web UI — app.js
   Particle animation · Donut chart · History · Copy · Export PNG
   ===================================================================== */
'use strict';

// ── Constants ─────────────────────────────────────────────────────────
const MAX_ENTROPY  = Math.log(10); // ≈ 2.3026 (uniform over 10 classes)
const CLASS_COLORS = [
  '#3d8ef8','#8b5cf6','#06b6d4','#22c55e',
  '#f59e0b','#ef4444','#ec4899','#14b8a6',
  '#f97316','#a855f7'
];
const ALLOWED_TYPES = ['image/jpeg','image/png','image/webp','image/bmp','image/gif'];

// ── State ─────────────────────────────────────────────────────────────
let selectedFile   = null;
let lastResult     = null;
let predHistory    = [];
let donutState     = null; // live donut render state
let activeHistoryId = null;

// ── DOM refs ──────────────────────────────────────────────────────────
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const previewCard    = document.getElementById('previewCard');
const previewImg     = document.getElementById('previewImg');
const changeBtn      = document.getElementById('changeBtn');
const analyzeWrap    = document.getElementById('analyzeWrap');
const analyzeBtn     = document.getElementById('analyzeBtn');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const btnSpinner     = document.getElementById('btnSpinner');
const errorToast     = document.getElementById('errorToast');
const errorMsg       = document.getElementById('errorMsg');
const toastClose     = document.getElementById('toastClose');
const resultsPanel   = document.getElementById('resultsPanel');
const resultHeroCard = document.getElementById('resultHeroCard');

const resultEmoji      = document.getElementById('resultEmoji');
const resultClass      = document.getElementById('resultClass');
const resultContext    = document.getElementById('resultContext');
const reliabilityBadge = document.getElementById('reliabilityBadge');
const confidenceValue  = document.getElementById('confidenceValue');
const confidenceBar    = document.getElementById('confidenceBar');
const errorProbValue   = document.getElementById('errorProbValue');
const errorProbBar     = document.getElementById('errorProbBar');
const softmaxValue     = document.getElementById('softmaxValue');
const softmaxBar       = document.getElementById('softmaxBar');
const entropyValue     = document.getElementById('entropyValue');
const entropyBar       = document.getElementById('entropyBar');
const distCanvas       = document.getElementById('distCanvas');
const distLegend       = document.getElementById('distLegend');
const copyBtn          = document.getElementById('copyBtn');
const exportBtn        = document.getElementById('exportBtn');
const historyPanel     = document.getElementById('historyPanel');
const historyCount     = document.getElementById('historyCount');
const historyList      = document.getElementById('historyList');
const clearHistoryBtn  = document.getElementById('clearHistoryBtn');

// ── Particle Animation ────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('heroParticles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, nodes, animId;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = canvas.parentElement.offsetWidth;
    H = canvas.parentElement.offsetHeight;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    nodes = Array.from({ length: 40 }, () => ({
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r:  Math.random() * 1.8 + 0.8,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const MAX_D = 130;
    // edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < MAX_D) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(61,142,248,${(1 - d / MAX_D) * 0.12})`;
          ctx.lineWidth = 1;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }
    // nodes
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(61,142,248,0.28)';
      ctx.fill();
    });
  }

  function update() {
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
  }

  function loop() { draw(); update(); animId = requestAnimationFrame(loop); }

  resize();
  loop();
  window.addEventListener('resize', () => {
    cancelAnimationFrame(animId);
    resize();
    loop();
  });
})();

// ── Helpers ───────────────────────────────────────────────────────────
function showError(msg) { errorMsg.textContent = msg; errorToast.hidden = false; }
function hideError()    { errorToast.hidden = true; }

function pct(val) { return (val * 100).toFixed(1) + '%'; }

function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtnText.textContent = on ? 'Analyzing…' : 'Analyze with SDNN';
  btnSpinner.hidden = !on;
}

function animateValue(el, target, decimals = 1, suffix = '%', duration = 900) {
  const start = performance.now();
  const to = parseFloat((target * 100).toFixed(decimals));
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (to * eased).toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function animateRaw(el, from, to, decimals = 2, suffix = '', duration = 900) {
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (from + (to - from) * eased).toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function getThumbDataURL(imgEl) {
  try {
    const c = document.createElement('canvas');
    c.width = 60; c.height = 60;
    c.getContext('2d').drawImage(imgEl, 0, 0, 60, 60);
    return c.toDataURL('image/jpeg', 0.75);
  } catch (_) { return ''; }
}

// ── File handling ─────────────────────────────────────────────────────
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

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

changeBtn.addEventListener('click', () => {
  selectedFile = null; fileInput.value = ''; previewImg.src = '';
  previewCard.hidden = true; analyzeWrap.hidden = true;
  resultsPanel.hidden = true; uploadZone.hidden = false;
  hideError();
});

toastClose.addEventListener('click', hideError);

// Enter key shortcut
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !analyzeWrap.hidden && !analyzeBtn.disabled) analyzeBtn.click();
});

// ── Predict ───────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  hideError();
  setLoading(true);
  try {
    const fd = new FormData();
    fd.append('image', selectedFile);
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Prediction failed.'); return; }
    const thumb = getThumbDataURL(previewImg);
    renderResults(data, thumb);
    addHistory(data, thumb);
  } catch (err) {
    showError('Could not reach the server. Is it running?');
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// ── Render Results ────────────────────────────────────────────────────
function renderResults(d, _thumb) {
  lastResult = d;

  // Hero card shimmer
  resultHeroCard.classList.remove('shimmer-once');
  void resultHeroCard.offsetWidth; // reflow
  resultHeroCard.classList.add('shimmer-once');

  resultEmoji.textContent   = d.emoji;
  resultClass.textContent   = d.predicted_class;
  resultContext.textContent = d.context;

  if (d.reliability === 'reliable') {
    reliabilityBadge.className   = 'reliability-badge badge-reliable';
    reliabilityBadge.textContent = '✅ Reliable';
  } else {
    reliabilityBadge.className   = 'reliability-badge badge-unsafe';
    reliabilityBadge.textContent = '⚠️ Safe Fallback';
  }

  resultsPanel.hidden = false;

  // Staggered stat animations
  setTimeout(() => {
    animateValue(confidenceValue, d.confidence);
    confidenceBar.style.width      = pct(d.confidence);
    confidenceBar.style.background = d.confidence > 0.7
      ? 'linear-gradient(90deg,#22c55e,#06b6d4)'
      : d.confidence > 0.4
        ? 'linear-gradient(90deg,#3d8ef8,#06b6d4)'
        : 'linear-gradient(90deg,#f59e0b,#ef4444)';
  }, 60);

  setTimeout(() => {
    animateValue(errorProbValue, d.error_prob);
    errorProbBar.style.width      = pct(d.error_prob);
    errorProbBar.style.background = d.error_prob > d.threshold
      ? 'linear-gradient(90deg,#f59e0b,#ef4444)'
      : 'linear-gradient(90deg,#22c55e,#06b6d4)';
  }, 140);

  setTimeout(() => {
    animateValue(softmaxValue, d.max_softmax);
    softmaxBar.style.width = pct(d.max_softmax);
  }, 220);

  setTimeout(() => {
    // Entropy is already in nats; display as raw value / max
    const ent    = d.entropy;
    const entPct = Math.min(ent / MAX_ENTROPY, 1);
    animateRaw(entropyValue, 0, ent, 3, '');
    entropyBar.style.width      = (entPct * 100).toFixed(1) + '%';
    entropyBar.style.background = entPct < 0.35
      ? 'linear-gradient(90deg,#22c55e,#06b6d4)'
      : entPct < 0.65
        ? 'linear-gradient(90deg,#06b6d4,#8b5cf6)'
        : 'linear-gradient(90deg,#f59e0b,#ef4444)';
  }, 300);

  // Donut chart (needs resultsPanel visible for sizing)
  setTimeout(() => drawDonut(d.classes, d.softmax_probs, d.predicted_idx), 50);

  // Scroll into view
  setTimeout(() => resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

// ── Donut Chart ───────────────────────────────────────────────────────
function drawDonut(classes, probs, topIdx) {
  const dpr  = window.devicePixelRatio || 1;
  const wrap  = distCanvas.parentElement;
  const avail = wrap.offsetWidth;
  const size  = Math.min(Math.floor(avail * 0.38), 280);
  const px    = Math.round(size * dpr);

  distCanvas.width  = px;
  distCanvas.height = px;
  distCanvas.style.width  = size + 'px';
  distCanvas.style.height = size + 'px';

  const ctx = distCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const cx = size / 2, cy = size / 2;
  const outerR = size / 2 - 10;
  const innerR = outerR * 0.52;

  // Build slices
  const slices = [];
  let angle = -Math.PI / 2;
  classes.forEach((cls, i) => {
    const sweep = probs[i] * Math.PI * 2;
    slices.push({ cls, prob: probs[i], i, color: CLASS_COLORS[i], start: angle, end: angle + sweep });
    angle += sweep;
  });

  donutState = { ctx, slices, cx, cy, outerR, innerR, classes, probs, topIdx, size, dpr };
  renderDonut(donutState, -1);

  // Hover detection
  distCanvas.onmousemove = e => {
    const rect = distCanvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const dx   = mx - cx, dy = my - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let hov = -1;
    if (dist > innerR && dist < outerR + 12) {
      let a = Math.atan2(dy, dx);
      if (a < -Math.PI / 2) a += 2 * Math.PI;
      for (let k = 0; k < slices.length; k++) {
        if (a >= slices[k].start && a < slices[k].end) { hov = k; break; }
      }
    }
    renderDonut(donutState, hov);
  };
  distCanvas.onmouseleave = () => renderDonut(donutState, -1);

  // Legend  (all 10, sorted by prob)
  const sorted = [...slices].sort((a, b) => b.prob - a.prob);
  distLegend.innerHTML = sorted.map(s => `
    <div class="legend-row ${s.i === topIdx ? 'is-top' : ''}">
      <span class="legend-dot" style="background:${s.color}"></span>
      <span class="legend-cls">${s.cls}</span>
      <span class="legend-pct">${(s.prob * 100).toFixed(1)}%</span>
    </div>`).join('');
}

function renderDonut({ ctx, slices, cx, cy, outerR, innerR, classes, probs, topIdx, size }, hovIdx) {
  ctx.clearRect(0, 0, size, size);

  slices.forEach((s, k) => {
    const isTop = s.i === topIdx;
    const isHov = k === hovIdx;
    const r     = (isTop || isHov) ? outerR + 5 : outerR;

    ctx.beginPath();
    ctx.arc(cx, cy, r,      s.start, s.end);
    ctx.arc(cx, cy, innerR, s.end,   s.start, true);
    ctx.closePath();

    ctx.globalAlpha = hovIdx === -1 ? (isTop ? 1 : 0.68) : (isHov ? 1 : 0.35);
    ctx.fillStyle   = s.color;

    if (isTop || isHov) {
      ctx.shadowColor = s.color;
      ctx.shadowBlur  = isHov ? 18 : 10;
    } else {
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  });

  // Donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
  ctx.fillStyle = '#0d1220';
  ctx.fill();

  // Center text
  const dispIdx = hovIdx >= 0 ? slices[hovIdx].i : topIdx;
  const dispCls = classes[dispIdx];
  const dispPct = (probs[dispIdx] * 100).toFixed(1) + '%';
  const fSize   = Math.round(innerR * 0.27);

  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.font         = `700 ${fSize}px Inter,system-ui,sans-serif`;
  ctx.fillStyle    = '#f0f4ff';
  ctx.fillText(dispCls, cx, cy - fSize * 0.65);

  ctx.font      = `${Math.round(fSize * 0.85)}px Inter,system-ui,sans-serif`;
  ctx.fillStyle = '#94a3b8';
  ctx.fillText(dispPct, cx, cy + fSize * 0.75);
}

// ── Copy to Clipboard ─────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  if (!lastResult) return;
  const d = lastResult;
  const text = [
    'SDNN Prediction Report',
    '──────────────────────────',
    `Class       : ${d.predicted_class} ${d.emoji}`,
    `Reliability : ${d.reliability === 'reliable' ? '✅ Reliable' : '⚠️ Safe Fallback'}`,
    `Confidence  : ${(d.confidence  * 100).toFixed(1)}%`,
    `Error Prob  : ${(d.error_prob  * 100).toFixed(1)}%`,
    `Top-1 Prob  : ${(d.max_softmax * 100).toFixed(1)}%`,
    `Entropy     : ${d.entropy.toFixed(4)} nats`,
    '──────────────────────────',
    'Powered by Auralis SDNN',
  ].join('\n');

  try {
    await navigator.clipboard.writeText(text);
    copyBtn.classList.add('copied');
    copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
    setTimeout(() => {
      copyBtn.classList.remove('copied');
      copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
    }, 2000);
  } catch (_) {
    showError('Clipboard access denied. Try copying manually.');
  }
});

// ── Export PNG ────────────────────────────────────────────────────────
exportBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const d = lastResult;

  exportBtn.disabled = true;
  exportBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Generating…';

  const W = 820, H = 480;
  const c = document.createElement('canvas');
  c.width = W * 2; c.height = H * 2;
  const x = c.getContext('2d');
  x.scale(2, 2);

  // Background
  const bg = x.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#06090f');
  bg.addColorStop(1, '#0d1220');
  x.fillStyle = bg;
  x.fillRect(0, 0, W, H);

  // Header strip
  x.fillStyle = 'rgba(61,142,248,0.08)';
  x.fillRect(0, 0, W, 52);

  // Accent line
  const accent = x.createLinearGradient(0, 51, W, 51);
  accent.addColorStop(0, '#3d8ef8');
  accent.addColorStop(1, '#8b5cf6');
  x.fillStyle = accent;
  x.fillRect(0, 51, W, 2);

  // Header text
  x.font      = '600 11px Inter,system-ui,sans-serif';
  x.fillStyle = '#3d8ef8';
  x.textBaseline = 'middle';
  x.fillText('AURALIS SDNN', 28, 26);
  x.fillStyle = '#475569';
  x.textAlign = 'right';
  x.fillText('Prediction Report · ' + new Date().toLocaleString(), W - 28, 26);
  x.textAlign = 'left';

  // Thumb (attempt)
  function drawContent() {
    const thumbX = 36, thumbY = 76, thumbS = 180;
    // Class area
    const infoX = thumbX + thumbS + 36;
    const infoW = W - infoX - 36;

    // Class name
    x.font      = '800 38px Inter,system-ui,sans-serif';
    x.fillStyle = '#f0f4ff';
    x.textBaseline = 'top';
    x.fillText(d.predicted_class.charAt(0).toUpperCase() + d.predicted_class.slice(1), infoX, thumbY + 2);

    // Reliability pill
    const isRel = d.reliability === 'reliable';
    const pillX = infoX, pillY = thumbY + 56;
    x.fillStyle = isRel ? 'rgba(34,197,94,0.15)' : 'rgba(245,158,11,0.15)';
    roundRect(x, pillX, pillY, 135, 28, 14); x.fill();
    x.font      = '700 12px Inter,system-ui,sans-serif';
    x.fillStyle = isRel ? '#86efac' : '#fcd34d';
    x.textBaseline = 'middle';
    x.fillText(isRel ? '✅  Reliable' : '⚠️  Safe Fallback', pillX + 14, pillY + 14);

    // Stats
    const stats = [
      { label: 'Confidence',   val: (d.confidence  * 100).toFixed(1) + '%', color: '#3d8ef8' },
      { label: 'Error Prob',   val: (d.error_prob  * 100).toFixed(1) + '%', color: '#f59e0b' },
      { label: 'Top-1 Prob',  val: (d.max_softmax * 100).toFixed(1) + '%', color: '#8b5cf6' },
      { label: 'Entropy',      val: d.entropy.toFixed(3) + ' nats',          color: '#06b6d4' },
    ];

    stats.forEach((s, i) => {
      const sy = thumbY + 108 + i * 52;
      // Label
      x.font      = '500 11px Inter,system-ui,sans-serif';
      x.fillStyle = '#475569';
      x.textBaseline = 'top';
      x.fillText(s.label.toUpperCase(), infoX, sy);
      // Value
      x.font      = '700 20px Inter,system-ui,sans-serif';
      x.fillStyle = '#f0f4ff';
      x.fillText(s.val, infoX, sy + 14);
      // Bar bg
      x.fillStyle = 'rgba(255,255,255,0.06)';
      roundRect(x, infoX + 100, sy + 22, infoW - 108, 5, 3); x.fill();
      // Bar fill — parse numeric portion
      const num = parseFloat(s.val) / (s.label === 'Entropy' ? (MAX_ENTROPY * 100) : 100);
      const barW = Math.max(5, (infoW - 108) * Math.min(num, 1));
      x.fillStyle = s.color;
      roundRect(x, infoX + 100, sy + 22, barW, 5, 3); x.fill();
    });

    // Footer
    x.font      = '500 11px Inter,system-ui,sans-serif';
    x.fillStyle = '#1e293b';
    x.fillRect(0, H - 38, W, 38);
    x.fillStyle = '#334155';
    x.textBaseline = 'middle';
    x.textAlign    = 'center';
    x.fillText('Generated by Auralis SDNN · ResNet-18 Backbone · CIFAR-10', W / 2, H - 19);

    // Download
    const a = document.createElement('a');
    a.href     = c.toDataURL('image/png');
    a.download = `sdnn-${d.predicted_class}-${Date.now()}.png`;
    a.click();

    exportBtn.disabled = false;
    exportBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export PNG`;
  }

  // Draw preview thumbnail
  const img = new Image();
  img.onload = () => {
    // Rounded rect clip for thumb
    x.save();
    roundRect(x, 36, 76, 180, 180, 12);
    x.clip();
    x.drawImage(img, 36, 76, 180, 180);
    x.restore();
    // Border
    x.strokeStyle = 'rgba(255,255,255,0.08)';
    x.lineWidth   = 1;
    roundRect(x, 36, 76, 180, 180, 12);
    x.stroke();
    drawContent();
  };
  img.onerror = () => { drawContent(); };
  img.src = previewImg.src;
});

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ── History ───────────────────────────────────────────────────────────
function addHistory(data, thumb) {
  const entry = { id: Date.now(), data, thumb };
  predHistory.unshift(entry);
  activeHistoryId = entry.id;
  renderHistory();
}

function renderHistory() {
  historyCount.textContent = predHistory.length;
  historyPanel.hidden = predHistory.length === 0;

  historyList.innerHTML = '';
  predHistory.forEach(entry => {
    const d   = entry.data;
    const div = document.createElement('div');
    div.className = 'history-entry' + (entry.id === activeHistoryId ? ' active' : '');
    div.dataset.id = entry.id;
    div.innerHTML = `
      ${entry.thumb ? `<img class="history-thumb" src="${entry.thumb}" alt="${d.predicted_class}" />` : '<div class="history-thumb" style="background:rgba(255,255,255,0.04)"></div>'}
      <span class="history-class">${d.predicted_class}</span>
      <span class="history-conf">${(d.confidence * 100).toFixed(0)}% conf</span>
      <div class="history-rel ${d.reliability === 'reliable' ? 'ok' : 'warn'}"></div>`;
    div.addEventListener('click', () => {
      activeHistoryId = entry.id;
      // Re-render results from history
      renderResults(d, entry.thumb);
      // Update active state
      document.querySelectorAll('.history-entry').forEach(el => el.classList.remove('active'));
      div.classList.add('active');
    });
    historyList.appendChild(div);
  });
}

clearHistoryBtn.addEventListener('click', () => {
  predHistory = [];
  activeHistoryId = null;
  renderHistory();
});
