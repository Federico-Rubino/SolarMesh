// Solar Camera: each physical panel is one grayscale pixel (brightness = voltage).
// 4 px mode adds pixels linearly interpolated between neighbouring panels.

const PARAMS = new URLSearchParams(location.search);
const SOURCE = PARAMS.get('source');
const STRIP_ROWS = 240;
const PEAK_DECAY = 0.9995;

const state = { px: 2, levels: false, peak: 0, strip: null };

const $ = id => document.getElementById(id);

// ---------- pixels ----------

function physical(panels) {
  return [...panels]
    .filter(p => p.voltage != null)
    .sort((a, b) => a.x - b.x || a.y - b.y || a.id.localeCompare(b.id, undefined, { numeric: true }));
}

function pixels(panels) {
  const real = physical(panels).map(p => ({ id: p.id, voltage: p.voltage, real: true }));
  if (state.px !== 4 || real.length < 2) return real;
  const out = [];
  for (let i = 0; i < real.length - 1; i++) {
    const a = real[i];
    const b = real[i + 1];
    out.push(a);
    for (const f of [1 / 3, 2 / 3]) {
      out.push({ id: `${a.id}·${b.id}`, voltage: a.voltage + (b.voltage - a.voltage) * f, real: false });
    }
  }
  out.push(real[real.length - 1]);
  return out;
}

function gray(v) {
  if (state.peak <= 0) return 0;
  let g = Math.min(Math.max(v / state.peak, 0), 1);
  if (state.levels) g = Math.round(g * 7) / 7;
  return g;
}

// ---------- drawing ----------

function drawFrame(px) {
  const canvas = $('frame');
  canvas.width = px.length;
  canvas.height = 1;
  canvas.style.aspectRatio = `${Math.max(px.length, 1)} / 1`;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(px.length, 1);
  px.forEach((p, i) => {
    const g = Math.round(gray(p.voltage) * 255);
    img.data.set([g, g, g, 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);

  const labels = $('labels');
  labels.style.gridTemplateColumns = `repeat(${px.length}, 1fr)`;
  labels.innerHTML = px
    .map(p => `<div class="${p.real ? '' : 'interp'}">
      <b>${p.voltage.toFixed(2)} V</b>${Math.round(gray(p.voltage) * 100)}% · ${p.real ? p.id : 'interp.'}</div>`)
    .join('');
}

function drawStrip(px) {
  const canvas = $('strip');
  if (canvas.width !== px.length || canvas.height !== STRIP_ROWS) {
    canvas.width = px.length;
    canvas.height = STRIP_ROWS;
  }
  const ctx = canvas.getContext('2d');
  ctx.drawImage(canvas, 0, -1);
  const row = ctx.createImageData(px.length, 1);
  px.forEach((p, i) => {
    const g = Math.round(gray(p.voltage) * 255);
    row.data.set([g, g, g, 255], i * 4);
  });
  ctx.putImageData(row, 0, STRIP_ROWS - 1);
}

function clearStrip() {
  const canvas = $('strip');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// ---------- update ----------

function update(twin) {
  if (!twin || !twin.panels || !twin.panels.length) return;
  const real = physical(twin.panels);
  const vmax = Math.max(...real.map(p => p.voltage));
  state.peak = Math.max(state.peak * PEAK_DECAY, vmax);
  $('ref').textContent = state.peak.toFixed(2);

  const px = pixels(twin.panels);
  $('caption').textContent = state.px === 4 && real.length >= 2
    ? `${real.length} panels → ${px.length} pixels (2 interpolated between each pair)`
    : `${px.length} pixels = ${px.length} physical panels`;
  drawFrame(px);
  drawStrip(px);
}

// ---------- controls ----------

document.querySelectorAll('[data-px]').forEach(b =>
  b.addEventListener('click', () => {
    state.px = Number(b.dataset.px);
    document.querySelectorAll('[data-px]').forEach(x => x.classList.toggle('active', x === b));
    clearStrip();
  }));
$('levels').addEventListener('change', e => { state.levels = e.target.checked; });
$('reset').addEventListener('click', () => { state.peak = 0; });
if (SOURCE) $('back').href = `index.html?source=${encodeURIComponent(SOURCE)}`;

// ---------- connection (same data as the dashboard) ----------

function setStatus(text) {
  const s = $('status');
  s.textContent = text;
  s.className = `status ${text}`;
}

async function poll() {
  try {
    const r = await fetch(SOURCE, { cache: 'no-store' });
    update(await r.json());
    setStatus('live');
  } catch (e) {
    setStatus('offline');
  }
  setTimeout(poll, 500);
}

if (SOURCE) {
  poll();
} else {
  try {
    const ui = new WebUI();
    ui.on_connect(() => setStatus('live'));
    ui.on_disconnect(() => setStatus('offline'));
    ui.on_message('twin', update);
  } catch (e) {
    console.error('WebUI unavailable', e);
  }
}
