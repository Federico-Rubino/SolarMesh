const NS = 'http://www.w3.org/2000/svg';
const PANEL_W = 110;
const PANEL_H = 70;
const GAP = 40;
const HISTORY = 120;
const PARAMS = new URLSearchParams(location.search);
const DEMO = PARAMS.has('demo');
const SOURCE = PARAMS.get('source');

const DIAGNOSES = {
  ok: { label: 'OK', color: 'var(--d-ok)', icon: '' },
  shade: { label: 'Cloud shade', color: 'var(--d-shade)', icon: '☁' },
  soiling: { label: 'Soiling', color: 'var(--d-soiling)', icon: '🧽' },
  obstruction: { label: 'Obstruction', color: 'var(--d-obstruction)', icon: '🔍' },
  fault: { label: 'Fault', color: 'var(--d-fault)', icon: '🔧' },
  dead: { label: 'Dead', color: 'var(--d-dead)', icon: '🔧' },
  unknown: { label: 'Unknown', color: 'var(--d-unknown)', icon: '?' },
};

const ACTIONS = {
  none: 'No action',
  clean: 'Schedule cleaning',
  inspect: 'Inspect for obstruction',
  technician: 'Send technician',
};

const state = {
  twin: null,
  metric: 'diagnosis',
  selected: null,
  history: [],
  panelHistory: {},
};

// ---------- helpers ----------

const $ = id => document.getElementById(id);

function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const fmt = {
  watt: v => (v == null ? '–' : `${v.toFixed(2)} W`),
  pct: v => (v == null ? '–' : `${Math.round(v * 100)}%`),
  volt: v => (v == null ? '–' : `${v.toFixed(2)} V`),
  secs: v => (v == null ? '–' : `${v.toFixed(1)} s`),
};

const diag = p => DIAGNOSES[p.diagnosis] || DIAGNOSES.unknown;

function healthColor(h) {
  if (h == null) return 'var(--unknown)';
  const x = Math.min(Math.max(h, 0), 1);
  return `hsl(${x * 120} 60% 50%)`;
}

function panelColor(p, ref) {
  if (state.metric === 'diagnosis') return diag(p).color;
  if (state.metric === 'voltage') return healthColor(ref > 0 && p.voltage != null ? p.voltage / ref : null);
  return healthColor(p.health);
}

function panelText(p) {
  if (state.metric === 'diagnosis') return diag(p).label;
  if (state.metric === 'voltage') return fmt.volt(p.voltage);
  return fmt.pct(p.health);
}

function neighbours(id, edges) {
  return edges
    .filter(e => e.source === id || e.target === id)
    .map(e => ({ id: e.source === id ? e.target : e.source, wired: !!e.wired }));
}

// ---------- digital twin ----------

// Layout comes from the plant (any units, any shape): the closest pair of
// modules defines one grid cell, so 2 panels or 500 panels both fit.
let geo = { key: null };

function geometry(panels) {
  const key = panels.map(p => `${p.id}:${p.x},${p.y}`).join('|');
  if (geo.key === key) return geo;
  let unit = Infinity;
  for (let i = 0; i < panels.length; i++) {
    for (let j = i + 1; j < panels.length; j++) {
      const d = Math.hypot(panels[i].x - panels[j].x, panels[i].y - panels[j].y);
      if (d > 1e-9 && d < unit) unit = d;
    }
  }
  if (!isFinite(unit)) unit = 1;
  const xs = panels.map(p => p.x);
  const ys = panels.map(p => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  geo = {
    key,
    unit,
    minX,
    minY,
    spanX: (Math.max(...xs) - minX) / unit,
    spanY: (Math.max(...ys) - minY) / unit,
  };
  return geo;
}

const px = p => ((p.x - geo.minX) / geo.unit) * (PANEL_W + GAP);
const py = p => ((p.y - geo.minY) / geo.unit) * (PANEL_H + GAP);
const center = p => [px(p) + PANEL_W / 2, py(p) + PANEL_H / 2];

function detailLevel(svg, vbW, vbH) {
  const scale = Math.min(svg.clientWidth / vbW, svg.clientHeight / vbH);
  const size = PANEL_W * scale;
  return size >= 70 ? 'full' : size >= 28 ? 'medium' : 'small';
}

function renderTwin(t) {
  const svg = $('twin');
  svg.replaceChildren();
  if (!t.panels.length) return;

  geometry(t.panels);
  const w = (geo.spanX + 1) * (PANEL_W + GAP) - GAP;
  const h = (geo.spanY + 1) * (PANEL_H + GAP) - GAP;
  const vbW = w + 60;
  const vbH = h + 60;
  svg.setAttribute('viewBox', `-30 -30 ${vbW} ${vbH}`);
  const width = svg.clientWidth || 800;
  svg.style.height = `${Math.min(Math.max(width * (vbH / vbW), 220), 720)}px`;
  const level = detailLevel(svg, vbW, vbH);

  const defs = svgEl('defs', {}, svg);
  const marker = svgEl('marker', {
    id: 'arrow-head', viewBox: '0 0 10 10', refX: 5, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto',
  }, defs);
  svgEl('path', { d: 'M0,0 L10,5 L0,10 z', class: 'cloud-arrow-head' }, marker);

  const byId = Object.fromEntries(t.panels.map(p => [p.id, p]));
  const edges = [...t.graph.edges].sort((a, b) => a.wired - b.wired);
  for (const e of edges) {
    if (level === 'small' && !e.wired) continue;
    const a = byId[e.source];
    const b = byId[e.target];
    if (!a || !b) continue;
    const [x1, y1] = center(a);
    const [x2, y2] = center(b);
    svgEl('line', { x1, y1, x2, y2, class: `edge ${e.wired ? 'wired' : 'adjacent'}` }, svg);
  }

  const ref = t.metrics.reference;
  for (const p of t.panels) {
    const g = svgEl('g', {
      class: `panel${p.id === state.selected ? ' selected' : ''}`,
      transform: `translate(${px(p)} ${py(p)})`,
    }, svg);
    const body = svgEl('rect', { class: 'body', width: PANEL_W, height: PANEL_H, rx: 6 }, g);
    body.style.fill = panelColor(p, ref);
    if (p.eta != null) svgEl('rect', { class: 'eta-ring', x: -4, y: -4, width: PANEL_W + 8, height: PANEL_H + 8, rx: 9 }, g);

    if (level === 'full') {
      svgEl('text', { x: 8, y: 18 }, g).textContent = p.id;
      if (diag(p).icon) svgEl('text', { x: PANEL_W - 8, y: 20, 'text-anchor': 'end', class: 'icon' }, g).textContent = diag(p).icon;
      svgEl('text', { x: PANEL_W / 2, y: PANEL_H / 2 + 8, 'text-anchor': 'middle', class: 'value' }, g).textContent = panelText(p);
      if (p.eta != null) {
        svgEl('text', { x: PANEL_W / 2, y: PANEL_H - 8, 'text-anchor': 'middle', class: 'eta' }, g).textContent = `☁ in ${p.eta.toFixed(1)} s`;
      }
    } else if (level === 'medium' && diag(p).icon && p.diagnosis !== 'shade') {
      svgEl('text', { x: PANEL_W / 2, y: PANEL_H / 2 + 14, 'text-anchor': 'middle', class: 'icon', style: 'font-size:40px' }, g).textContent = diag(p).icon;
    }

    body.addEventListener('click', () => {
      state.selected = p.id;
      render();
    });
  }

  renderCloudArrow(svg, t);
}

function renderCloudArrow(svg, t) {
  const c = t.cloud;
  if (!c || !c.active || !c.velocity) return;
  const shaded = t.panels.filter(p => p.diagnosis === 'shade');
  if (!shaded.length) return;

  const cx = shaded.reduce((s, p) => s + center(p)[0], 0) / shaded.length;
  const cy = shaded.reduce((s, p) => s + center(p)[1], 0) / shaded.length;
  const [vx, vy] = c.velocity;
  const norm = Math.hypot(vx, vy) || 1;
  const len = (PANEL_W + GAP) * Math.max(1, Math.max(geo.spanX, geo.spanY) / 5);
  const x2 = cx + (vx / norm) * len;
  const y2 = cy + (vy / norm) * len;
  svgEl('line', { x1: cx, y1: cy, x2, y2, class: 'cloud-arrow', 'marker-end': 'url(#arrow-head)' }, svg);
}

function renderLegend() {
  const box = $('twin-legend');
  if (state.metric === 'diagnosis') {
    box.innerHTML = Object.entries(DIAGNOSES)
      .filter(([k]) => k !== 'unknown')
      .map(([, d]) => `<span><i class="dot" style="background:${d.color}"></i> ${d.icon} ${d.label}</span>`)
      .join('') + '<span>┅ blue ring = cloud arriving</span>';
  } else {
    box.innerHTML = '<span><i class="scale"></i> 0% → 100% of best module</span><span>━ wired · ┄ adjacent</span>';
  }
}

// ---------- banners & KPIs ----------

function renderBanners(t) {
  const tech = t.technician;
  const techBanner = $('tech-banner');
  if (tech.needed) {
    techBanner.className = 'banner alert';
    $('tech-title').textContent = `Technician needed: ${tech.modules.join(', ')}`;
    const lost = t.panels.filter(p => tech.modules.includes(p.id)).reduce((s, p) => s + (p.loss_w || 0), 0);
    $('tech-detail').textContent = `~${lost.toFixed(2)} W lost until repaired`;
  } else {
    const other = t.panels.filter(p => p.action === 'clean' || p.action === 'inspect');
    techBanner.className = other.length ? 'banner warn' : 'banner ok';
    $('tech-title').textContent = 'No technician needed';
    $('tech-detail').textContent = other.length
      ? `${other.length} module(s) need cleaning or inspection`
      : 'Drops are temporary (clouds) or absent';
  }

  const c = t.cloud;
  const cloudBanner = $('cloud-banner');
  if (c && c.active && c.next_eta != null) {
    cloudBanner.className = 'banner warn';
    $('cloud-title').textContent = `Cloud approaching: first module in ${c.next_eta.toFixed(1)} s`;
    $('cloud-detail').textContent =
      `${Object.keys(c.eta).length} modules affected · ${c.speed_mps.toFixed(2)} m/s · heading ${Math.round(c.heading_deg)}°`;
  } else if (t.panels.some(p => p.diagnosis === 'shade')) {
    cloudBanner.className = 'banner warn';
    $('cloud-title').textContent = 'Cloud over the plant';
    $('cloud-detail').textContent = 'Tracking its movement…';
  } else {
    cloudBanner.className = 'banner ok';
    $('cloud-title').textContent = 'Sky clear';
    $('cloud-detail').textContent = 'No shadow detected';
  }
}

function renderKpis(t) {
  const m = t.metrics;
  $('k-actual').textContent = fmt.watt(m.power_actual);
  $('k-ideal').textContent = fmt.watt(m.power_ideal);
  $('k-lost').textContent = fmt.watt(m.power_lost);
  $('k-eff').textContent = fmt.pct(m.efficiency);
  $('k-action').textContent = t.panels.filter(p => p.action !== 'none').length;
  $('k-mse').textContent = m.mse.toFixed(3);
  $('temp').textContent = t.temp == null ? '–' : t.temp.toFixed(1);
  $('current').textContent = t.current == null ? '–' : t.current.toFixed(2);
  $('clock').textContent = new Date(t.time || Date.now()).toLocaleTimeString();
}

// ---------- charts ----------

const POWER_SERIES = [
  { key: 'power_actual', label: 'Actual', color: 'var(--accent)' },
  { key: 'power_ideal', label: 'Ideal', color: 'var(--muted)', dash: '5 4' },
];

function chartFrame(svg) {
  svg.replaceChildren();
  const width = svg.clientWidth || 600;
  const height = svg.clientHeight || 200;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  return { width, height, left: 44, right: 8, top: 10, bottom: 26 };
}

function yAxis(svg, f, max, unit) {
  for (let i = 0; i <= 4; i++) {
    const v = (max * i) / 4;
    const y = f.height - f.bottom - (i / 4) * (f.height - f.top - f.bottom);
    svgEl('line', { x1: f.left, x2: f.width - f.right, y1: y, y2: y, class: 'axis' }, svg);
    svgEl('text', { x: f.left - 6, y: y + 4, 'text-anchor': 'end' }, svg).textContent = `${v.toFixed(1)}${unit}`;
  }
}

function renderPower() {
  const svg = $('power');
  const f = chartFrame(svg);
  const data = state.history;
  if (!data.length) return;

  const max = Math.max(0.1, ...data.flatMap(d => POWER_SERIES.map(s => d[s.key] || 0))) * 1.1;
  yAxis(svg, f, max, 'W');
  const x = i => f.left + (i / Math.max(HISTORY - 1, 1)) * (f.width - f.left - f.right);
  const y = v => f.height - f.bottom - (v / max) * (f.height - f.top - f.bottom);

  for (const s of POWER_SERIES) {
    const points = data.map((d, i) => `${x(i)},${y(d[s.key] || 0)}`).join(' ');
    const line = svgEl('polyline', { points, fill: 'none', 'stroke-width': 2 }, svg);
    line.style.stroke = s.color;
    if (s.dash) line.setAttribute('stroke-dasharray', s.dash);
  }
  svgEl('text', { x: f.left, y: f.height - 6 }, svg).textContent = `last ${data.length} samples`;
}

function renderPowerLegend() {
  $('power-legend').innerHTML = POWER_SERIES
    .map(s => `<span><i class="dot" style="background:${s.color}"></i> ${s.label}</span>`)
    .join('');
}

function renderSummary(t) {
  const rows = Object.entries(DIAGNOSES)
    .map(([key, d]) => {
      const ps = t.panels.filter(p => p.diagnosis === key);
      return { d, count: ps.length, lost: ps.reduce((s, p) => s + (p.loss_w || 0), 0) };
    })
    .filter(r => r.count > 0);
  $('summary').innerHTML = rows
    .map(r => `<tr>
      <td><i class="dot" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${r.d.color}"></i></td>
      <td>${r.d.icon} ${r.d.label}</td><td>${r.count}</td><td>${r.lost.toFixed(2)} W</td></tr>`)
    .join('');
}

function sparkline(values) {
  const w = 260;
  const h = 60;
  const max = Math.max(0.1, ...values);
  const points = values
    .map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${h - 4 - (v / max) * (h - 8)}`)
    .join(' ');
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${points}" fill="none" style="stroke: var(--accent)" stroke-width="2"/></svg>`;
}

// ---------- module detail ----------

function renderDetail(t) {
  const box = $('detail');
  const p = t.panels.find(q => q.id === state.selected);
  if (!p) {
    box.innerHTML = '<h2>Module</h2><p class="hint">Click a module in the twin.</p>';
    return;
  }
  const d = diag(p);
  const strings = t.graph.strings || [];
  const si = strings.findIndex(s => s.includes(p.id));
  const where = si >= 0 ? `string ${si + 1} of ${strings.length}, #${strings[si].indexOf(p.id) + 1}` : '–';
  const near = neighbours(p.id, t.graph.edges).map(n => (n.wired ? `<b>${n.id}</b>` : n.id)).join(', ') || '–';

  box.innerHTML = `
    <h2>Module ${p.id}</h2>
    <p><span class="tag" style="background:${d.color}">${d.icon} ${d.label}</span></p>
    <dl>
      <dt>Action</dt><dd><b>${ACTIONS[p.action] || p.action}</b></dd>
      <dt>Since</dt><dd>${p.diagnosis === 'ok' ? '–' : fmt.secs(p.age)}</dd>
      <dt>Voltage</dt><dd>${fmt.volt(p.voltage)}</dd>
      <dt>Health</dt><dd>${fmt.pct(p.health)}</dd>
      <dt>Power lost</dt><dd>${fmt.watt(p.loss_w)}</dd>
      <dt>Cloud ETA</dt><dd>${p.eta == null ? '–' : fmt.secs(p.eta)}</dd>
      <dt>Position</dt><dd>${where}</dd>
      <dt>Neighbours</dt><dd>${near}</dd>
    </dl>
    <h2>Voltage history</h2>
    ${sparkline(state.panelHistory[p.id] || [])}`;
}

// ---------- alerts ----------

function renderAlerts(t) {
  const alerts = t.alerts || [];
  $('alerts').innerHTML = alerts.length
    ? alerts.map(a => `<li class="warn">${a}</li>`).join('')
    : '<li class="ok">No alerts</li>';
}

// ---------- update loop ----------

function record(t) {
  state.history.push({ ...t.metrics });
  if (state.history.length > HISTORY) state.history.shift();
  for (const p of t.panels) {
    const h = (state.panelHistory[p.id] ||= []);
    h.push(p.voltage || 0);
    if (h.length > HISTORY) h.shift();
  }
}

function render() {
  const t = state.twin;
  if (!t) return;
  renderTwin(t);
  renderBanners(t);
  renderKpis(t);
  renderPower();
  renderSummary(t);
  renderDetail(t);
  renderAlerts(t);
}

function update(twin) {
  if (!twin || !twin.panels) return;
  state.twin = twin;
  record(twin);
  render();
}

function setStatus(text) {
  const s = $('status');
  s.textContent = text;
  s.className = `status ${text}`;
}

document.querySelectorAll('[data-metric]').forEach(b =>
  b.addEventListener('click', () => {
    state.metric = b.dataset.metric;
    document.querySelectorAll('[data-metric]').forEach(x => x.classList.toggle('active', x === b));
    renderLegend();
    if (state.twin) renderTwin(state.twin);
  }));
window.addEventListener('resize', renderPower);

renderLegend();
renderPowerLegend();

// ---------- connection ----------

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

if (SOURCE) $('camera-link').href = `camera.html?source=${encodeURIComponent(SOURCE)}`;

if (DEMO) {
  setStatus('demo');
} else if (SOURCE) {
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

// ---------- demo mode (?demo): scripted scenario, no board needed ----------
// Labels here come from the script's ground truth, not from the Python diagnosis.

const demo = {
  cols: 6,
  rows: 4,
  speed: 0.5,
  t: 0,

  layout() {
    const panels = [];
    const edges = [];
    const series = [];
    for (let y = 0; y < this.rows; y++) {
      const row = [];
      for (let x = 0; x < this.cols; x++) {
        const id = `n${y * this.cols + x + 1}`;
        panels.push({ id, x, y });
        row.push(id);
        if (x > 0) edges.push({ source: `n${y * this.cols + x}`, target: id, wired: 1 });
        if (y > 0) edges.push({ source: `n${(y - 1) * this.cols + x + 1}`, target: id, wired: 0 });
      }
      series.push(...(y % 2 ? row.reverse() : row));
    }
    for (let y = 1; y < this.rows; y++) {
      const x = y % 2 ? this.cols - 1 : 0;
      const e = edges.find(q => q.source === `n${(y - 1) * this.cols + x + 1}` && q.target === `n${y * this.cols + x + 1}`);
      if (e) e.wired = 1;
    }
    return { panels, edges, series };
  },

  step() {
    this.t += 0.5;
    const cycle = this.t % 40;
    const front = -2 + this.speed * cycle;
    const { panels, edges, series } = this.layout();
    const current = 0.45;
    const vBest = 3.0;

    for (const p of panels) {
      let h = 1;
      let label = 'ok';
      const behind = front - p.x;
      if (behind >= 0 && behind <= 2 && p.y <= 2) {
        h = 0.35;
        label = 'shade';
      }
      if (p.id === 'n15') {
        h = Math.min(h, Math.max(0.55, 1 - 0.004 * this.t));
        if (h < 0.85 && label === 'ok') label = 'soiling';
      }
      if (p.id === 'n10' && this.t > 15) {
        h = 0.25;
        label = 'fault';
      }
      p.voltage = vBest * h + (Math.random() - 0.5) * 0.04;
      p.health = h;
      p.diagnosis = label;
      p.action = { ok: 'none', shade: 'none', soiling: 'clean', fault: 'technician' }[label];
      p.age = label === 'ok' ? 0 : 5;
      p.loss_w = current * vBest * (1 - h);
      const ahead = p.x - front;
      p.eta = label === 'ok' && p.y <= 2 && ahead > 0 && ahead / this.speed <= 60 && front > -1 ? ahead / this.speed : null;
    }

    const etas = Object.fromEntries(panels.filter(p => p.eta != null).map(p => [p.id, p.eta]));
    const actual = current * Math.min(...panels.map(p => p.health)) * panels.reduce((s, p) => s + p.voltage, 0);
    const ideal = current * vBest * panels.length;
    const techs = panels.filter(p => p.action === 'technician').map(p => p.id);
    const alerts = panels
      .filter(p => p.action !== 'none')
      .map(p => `${p.id}: ${p.diagnosis} → ${ACTIONS[p.action].toLowerCase()} (demo)`);

    update({
      time: Date.now(),
      temp: 24 + Math.sin(this.t / 30) * 2,
      current,
      panels,
      graph: { nodes: panels, edges, strings: [series] },
      metrics: {
        reference: vBest,
        mse: panels.reduce((s, p) => s + (vBest - p.voltage) ** 2, 0) / panels.length,
        power_actual: actual,
        power_ideal: ideal,
        power_lost: ideal - actual,
        efficiency: actual / ideal,
      },
      technician: { needed: techs.length > 0, modules: techs },
      cloud: Object.keys(etas).length
        ? {
            active: true,
            velocity: [this.speed, 0],
            speed_mps: this.speed,
            heading_deg: 0,
            eta: etas,
            next_eta: Math.min(...Object.values(etas)),
          }
        : { active: false, eta: {} },
      alerts,
    });
  },
};

if (DEMO) {
  demo.step();
  setInterval(() => demo.step(), 500);
}
