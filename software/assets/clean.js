// Motor cleaning demo: choose panel + duration, run the cleaning motor, and compare
// the panel voltage before and after.

const NS = 'http://www.w3.org/2000/svg';
const HISTORY_S = 120;
const COMPARE_S = 10;

const state = {
  panel: null,
  cleaner: null,
  samples: [],     // {t, v} for the selected panel
  runs: [],        // {start, end} motor intervals (ms)
  pending: null,   // run waiting for its "after" window
};

const $ = id => document.getElementById(id);
let ui = null;

// ---------- controls ----------

document.querySelectorAll('[data-s]').forEach(b => b.addEventListener('click', () => { $('seconds').value = b.dataset.s; }));

$('vibrate').addEventListener('click', () => {
  if (!ui || !state.panel) return;
  ui.send_message('clean', { panel: state.panel, seconds: Number($('seconds').value) });
  $('message').textContent = 'Sending…';
});
$('stop').addEventListener('click', () => ui && ui.send_message('stop_clean', {}));

function renderPanels(panels) {
  const box = $('panels');
  const ids = panels.map(p => p.id).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (box.dataset.ids === ids.join(',') && box.dataset.cleaner === (state.cleaner?.panel || '')) return;
  box.dataset.ids = ids.join(',');
  box.dataset.cleaner = state.cleaner?.panel || '';
  box.innerHTML = '';
  for (const id of ids) {
    const hasMotor = state.cleaner && state.cleaner.panel === id;
    const b = document.createElement('button');
    b.textContent = hasMotor ? `${id} · motor` : `${id} · no motor`;
    b.disabled = !hasMotor;
    b.classList.toggle('active', id === state.panel);
    b.addEventListener('click', () => selectPanel(id));
    box.appendChild(b);
  }
  if (!state.panel && state.cleaner) selectPanel(state.cleaner.panel);
}

function selectPanel(id) {
  state.panel = id;
  state.samples = [];
  $('chart-title').textContent = `Panel ${id} voltage`;
  document.querySelectorAll('#panels button').forEach(b => b.classList.toggle('active', b.textContent.startsWith(id + ' ')));
}

// ---------- before / after ----------

function mean(from, to) {
  const xs = state.samples.filter(s => s.t >= from && s.t < to).map(s => s.v);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function checkResult(now) {
  const run = state.pending;
  if (!run || run.end == null || now < run.end + COMPARE_S * 1000) return;
  const before = mean(run.start - COMPARE_S * 1000, run.start);
  const after = mean(run.end, run.end + COMPARE_S * 1000);
  state.pending = null;
  if (before == null || after == null) return;
  const gain = (after - before) / before;
  $('result').innerHTML = `Motor ${(run.ms / 1000).toFixed(0)} s on ${run.panel}: ` +
    `before <b>${before.toFixed(3)} V</b> → after <b>${after.toFixed(3)} V</b> ` +
    `<b class="${gain >= 0 ? 'up' : 'down'}">${gain >= 0 ? '+' : ''}${(gain * 100).toFixed(1)}%</b>`;
}

// ---------- chart ----------

function renderChart(now) {
  const svg = $('chart');
  svg.replaceChildren();
  const w = svg.clientWidth || 600;
  const h = svg.clientHeight || 200;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const t0 = now - HISTORY_S * 1000;
  const pts = state.samples.filter(s => s.t >= t0);
  if (!pts.length) return;
  const max = Math.max(0.1, ...pts.map(s => s.v)) * 1.1;
  const x = t => ((t - t0) / (HISTORY_S * 1000)) * w;
  const y = v => h - 6 - (Math.max(v, 0) / max) * (h - 12);

  for (const r of state.runs) {
    const end = r.end ?? now;
    if (end < t0) continue;
    const rect = document.createElementNS(NS, 'rect');
    rect.setAttribute('x', x(Math.max(r.start, t0)));
    rect.setAttribute('y', 0);
    rect.setAttribute('width', Math.max(2, x(end) - x(Math.max(r.start, t0))));
    rect.setAttribute('height', h);
    rect.setAttribute('class', 'vibe');
    svg.appendChild(rect);
  }
  const line = document.createElementNS(NS, 'polyline');
  line.setAttribute('points', pts.map(s => `${x(s.t)},${y(s.v)}`).join(' '));
  line.setAttribute('class', 'line');
  svg.appendChild(line);
}

// ---------- updates ----------

function update(twin) {
  if (!twin || !twin.panels) return;
  const now = Date.now();
  state.cleaner = twin.cleaner || null;
  renderPanels(twin.panels);

  const on = !!(state.cleaner && state.cleaner.on);
  const last = state.runs[state.runs.length - 1];
  if (on && (!last || last.end != null)) {
    const run = { start: now, end: null, ms: 0, panel: state.cleaner.panel };
    state.runs.push(run);
    state.pending = run;
  } else if (!on && last && last.end == null) {
    last.end = now;
    last.ms = last.end - last.start;
  }
  $('state').textContent = on ? 'motor running' : 'idle';
  $('state').classList.toggle('on', on);
  $('vibrate').disabled = !state.cleaner;

  const p = twin.panels.find(q => q.id === state.panel);
  if (p && p.voltage != null) {
    state.samples.push({ t: now, v: p.voltage });
    state.samples = state.samples.filter(s => s.t >= now - (HISTORY_S + COMPARE_S) * 1000);
    $('volts').textContent = p.voltage.toFixed(3);
  }
  checkResult(now);
  renderChart(now);
}

function setStatus(text) {
  $('status').textContent = text;
  $('status').className = `status ${text}`;
}

try {
  ui = new WebUI();
  ui.on_connect(() => setStatus('live'));
  ui.on_disconnect(() => setStatus('offline'));
  ui.on_message('twin', update);
  ui.on_message('clean_status', msg => {
    if (msg.stopped) $('message').textContent = 'Stopped.';
    else if (msg.ok) $('message').textContent = `Motor on ${msg.panel} for ${msg.ms / 1000} s.`;
    else $('message').textContent = `Error: ${msg.error || 'rejected by the board'}`;
  });
} catch (e) {
  console.error('WebUI unavailable', e);
}
