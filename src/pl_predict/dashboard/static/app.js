/* PL Predict Dashboard — vanilla JS + Plotly */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const API = '';

const TABS = ['overview', 'predict', 'season', 'performance', 'players', 'shots', 'teams', 'referees', 'features'];
const TAB_LABEL = {
  overview: 'Overview', predict: 'Predict', season: 'Season Sim', performance: 'Model Performance',
  players: 'Players', shots: 'Shot Maps', teams: 'Teams', referees: 'Referees', features: 'Features',
};
const TEAM_MAP = { 'Nottm Forest': "Nott'm Forest", 'Nottingham Forest': "Nott'm Forest", "Nott'm Forest": "Nott'm Forest", 'Wolves': 'Wolves', 'Man City': 'Man City', 'Man United': 'Man United', 'Manchester City': 'Man City', 'Manchester United': 'Man United', 'Newcastle': 'Newcastle' };

/* ---------- helpers ---------- */
function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function fmtPct(x, d = 1) { return x == null ? '--' : (x * 100).toFixed(d) + '%'; }
function fmtNum(x, d = 1) { return x == null ? '--' : Number(x).toFixed(d); }
function shortDate(d) { return d ? new Date(d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' }) : '--'; }

function toast(msg, type = 'error') {
  let wrap = $('#toast-wrap');
  if (!wrap) { wrap = document.createElement('div'); wrap.id = 'toast-wrap'; wrap.className = 'toast-wrap'; document.body.appendChild(wrap); }
  const t = document.createElement('div');
  t.className = 'toast' + (type === 'info' ? ' info' : '');
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => { t.remove(); }, 4500);
}

async function fetchJSON(url, opts) {
  const res = await fetch(API + url, opts);
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try { const j = await res.json(); if (j.detail) detail = typeof j.detail === 'string' ? j.detail : detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

function plotBase(extra = {}) {
  return Object.assign({
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#e2e8f0' }, margin: { l: 60, r: 30, t: 40, b: 45 },
    xaxis: { gridcolor: '#263258' }, yaxis: { gridcolor: '#263258' },
  }, extra);
}

function showLoading(id) { const el = $(id); if (el) el.innerHTML = '<div class="spinner" style="margin:40px auto"></div>'; }

/* ---------- router ---------- */
function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [path, qs] = h.split('?');
  const params = {};
  if (qs) new URLSearchParams(qs).forEach((v, k) => { params[k] = decodeURIComponent(v); });
  const parts = path.split('/').filter(Boolean);
  return { tab: parts[0] || 'overview', sub: parts.slice(1), params };
}

function render() {
  const { tab, sub, params } = parseHash();
  const name = TABS.includes(tab) ? tab : 'overview';
  $$('.nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  TABS.forEach(t => { $('#tab-' + t).classList.toggle('hidden', t !== name); });
  const loaders = {
    overview: loadOverview,
    predict: () => loadPredict(params),
    season: loadSeason,
    performance: loadPerformance,
    players: loadPlayers,
    shots: loadShots,
    teams: () => loadTeamsView(sub[0] || null),
    referees: loadReferees,
    features: loadFeatures,
  };
  if (loaders[name]) loaders[name]();
}
window.addEventListener('hashchange', render);

function navTo(tab, params) {
  let hash = '#/' + tab;
  if (params) hash += '?' + new URLSearchParams(params).toString();
  location.hash = hash;
}

/* ---------- Overview ---------- */
const overviewState = { fixtures: [], q: '', team: '', sort: 'date' };

async function loadOverview() {
  showLoading('#overview-content');
  try {
    const [health, fx, perf, sea] = await Promise.all([
      fetchJSON('/api/health'), fetchJSON('/api/fixtures_2627'),
      fetchJSON('/api/performance').catch(() => null), fetchJSON('/api/season').catch(() => null),
    ]);
    const teams = (health.teams || 0);
    const acc = perf && perf.n_matches ? perf.ens_acc : null;
    const sims = sea && sea.teams ? sea.teams[0] ? 'Yes' : 'No' : 'No';
    overviewState.fixtures = fx.fixtures || [];
    overviewState.team = overviewState.team || (teams ? (await fetchJSON('/api/teams')).teams[0] : '');

    const h = document.createElement('div');
    h.innerHTML = `
      <div class="grid-3">
        <div class="card"><h3>Total Matches</h3><div class="value blue">${(health.total_matches || 0).toLocaleString()}</div></div>
        <div class="card"><h3>Walk-Forward Accuracy</h3><div class="value green">${acc != null ? fmtPct(acc) : 'n/a'}</div></div>
        <div class="card"><h3>Season Simulation</h3><div class="value yellow">${sims}</div></div>
      </div>
      <div class="section-title">2026/27 Fixtures &amp; Predictions</div>
      <div class="card">
        <div class="toolbar">
          <div><label>Search</label><input type="text" id="fx-search" placeholder="Team or opponent..." value="${esc(overviewState.q)}"></div>
          <div><label>Team filter</label><select id="fx-team">
            <option value="">All teams</option>
            ${overviewState.fixtures.map(f => f.home_team).filter((v, i, a) => a.indexOf(v) === i).map(t => `<option value="${esc(t)}"${overviewState.team === t ? ' selected' : ''}>${esc(t)}</option>`).join('')}
          </select></div>
          <div><label>Sort</label><select id="fx-sort">
            <option value="date"${overviewState.sort === 'date' ? ' selected' : ''}>Date</option>
            <option value="conf"${overviewState.sort === 'conf' ? ' selected' : ''}>Confidence</option>
            <option value="o23"${overviewState.sort === 'o23' ? ' selected' : ''}>Over 2.5</option>
          </select></div>
          <div><label>Only with predictions</label><button id="fx-toggle">Toggle</button></div>
        </div>
        <div style="overflow-x:auto;">
          <table id="fx-table"><thead><tr>
            <th>Date</th><th>Home</th><th>vs</th><th>Away</th>
            <th>Home Win</th><th>Draw</th><th>Away Win</th>
            <th>Score</th><th>O/U 2.5</th><th>Conformal</th><th></th>
          </tr></thead><tbody id="fx-body"></tbody></table>
        </div>
      </div>`;
    $('#overview-content').innerHTML = '';
    $('#overview-content').appendChild(h);
    $('#fx-search').addEventListener('input', e => { overviewState.q = e.target.value.toLowerCase(); renderFixturesTable(); });
    $('#fx-team').addEventListener('change', e => { overviewState.team = e.target.value; renderFixturesTable(); });
    $('#fx-sort').addEventListener('change', e => { overviewState.sort = e.target.value; renderFixturesTable(); });
    $('#fx-toggle').addEventListener('click', () => { overviewState.onlyPred = !overviewState.onlyPred; $('#fx-toggle').style.opacity = overviewState.onlyPred ? 1 : .6; renderFixturesTable(); });
    renderFixturesTable();
  } catch (e) { toast('Failed to load overview: ' + e.message); $('#overview-content').innerHTML = '<div class="empty">Failed to load.</div>'; }
}

function normTeam(t) { return TEAM_MAP[t] || t; }

function renderFixturesTable() {
  let rows = overviewState.fixtures.filter(f => {
    const hasPred = f.prediction && !f.prediction.error;
    if (overviewState.onlyPred && !hasPred) return false;
    if (overviewState.team && !(f.home_team === overviewState.team || f.away_team === overviewState.team)) return false;
    if (overviewState.q) {
      const hay = (f.home_team + ' ' + f.away_team).toLowerCase();
      if (!hay.includes(overviewState.q)) return false;
    }
    return true;
  });
  if (overviewState.sort === 'conf') rows.sort((a, b) => {
    const ca = a.prediction ? Math.max(a.prediction.home_win, a.prediction.draw, a.prediction.away_win) : 0;
    const cb = b.prediction ? Math.max(b.prediction.home_win, b.prediction.draw, b.prediction.away_win) : 0;
    return cb - ca;
  });
  else if (overviewState.sort === 'o23') rows.sort((a, b) => (b.prediction ? b.prediction.over_2_5 : 0) - (a.prediction ? a.prediction.over_2_5 : 0));
  else rows.sort((a, b) => (a.date || '').localeCompare(b.date || ''));

  const tbody = $('#fx-body');
  if (!rows.length) { tbody.innerHTML = '<tr><td colspan="11" class="empty">No fixtures match.</td></tr>'; return; }
  tbody.innerHTML = rows.map(f => {
    const p = f.prediction && !f.prediction.error ? f.prediction : null;
    const conf = p ? Math.max(p.home_win, p.draw, p.away_win) : null;
    const ml = p ? p.most_likely_score : null;
    const confBadge = p ? (p.conformal_90 || []).map(r => `<span class="badge badge-${r === 'H' ? 'h' : r === 'D' ? 'd' : 'a'}">${r}</span>`).join(' ') : '';
    return `<tr onclick="navTo('predict', {home:'${esc(normTeam(f.home_team))}', away:'${esc(normTeam(f.away_team))}'})" style="cursor:pointer">
      <td>${shortDate(f.date)}</td>
      <td class="fixture-team home">${esc(f.home_team)}</td><td class="fixture-vs">vs</td><td class="fixture-team away">${esc(f.away_team)}</td>
      <td>${p ? `<div class="prob-bar small"><div class="prob-fill" style="width:${p.home_win * 100}%;background:var(--green)"></div><span>${fmtPct(p.home_win)}</span></div>` : '<span class="empty">--</span>'}</td>
      <td>${p ? `<div class="prob-bar small"><div class="prob-fill" style="width:${p.draw * 100}%;background:var(--yellow)"></div><span>${fmtPct(p.draw)}</span></div>` : '--'}</td>
      <td>${p ? `<div class="prob-bar small"><div class="prob-fill" style="width:${p.away_win * 100}%;background:var(--red)"></div><span>${fmtPct(p.away_win)}</span></div>` : '--'}</td>
      <td>${ml ? `${ml.home}-${ml.away}` : '--'}</td>
      <td>${p ? (p.over_2_5 > 0.5 ? '<span class="chip chip-green">Over</span>' : '<span class="chip chip-gray">Under</span>') : '--'}</td>
      <td>${confBadge || '--'}</td>
      <td style="color:var(--muted);font-size:11px">${conf != null ? (conf * 100).toFixed(0) + '%' : ''}</td>
    </tr>`;
  }).join('');
}

/* ---------- Predict ---------- */
const predState = { home: '', away: '' };

async function loadPredict(params) {
  showLoading('#predict-content');
  try {
    const teamsRes = await fetchJSON('/api/teams');
    const teams = teamsRes.teams || [];
    const home = normTeam(params.home) || predState.home || 'Arsenal';
    const away = normTeam(params.away) || predState.away || 'Chelsea';
    predState.home = home; predState.away = away;

    const h = document.createElement('div');
    h.innerHTML = `
      <div class="section-title">Predict a Match</div>
      <div class="card">
        <div class="predict-form">
          <div><label>Home Team</label><select id="home-select"></select></div>
          <div><label>Away Team</label><select id="away-select"></select></div>
          <button id="predict-btn">Predict</button>
        </div>
        <div class="result-box" id="prediction-result"><div class="spinner"></div></div>
        <div id="pred-chips" style="margin-top:14px"></div>
      </div>
      <div class="grid-2">
        <div class="card"><h3>Score Probability Matrix</h3><div class="matrix-wrap" id="pred-heatmap"></div></div>
        <div class="card"><h3>Model Breakdown</h3><div id="pred-breakdown"></div>
          <div style="margin-top:10px" id="pred-topsc"></div></div>
      </div>
      <div class="section-title">Head to Head</div>
      <div class="card" id="h2h-panel"><div class="spinner"></div></div>`;
    $('#predict-content').innerHTML = '';
    $('#predict-content').appendChild(h);

    [['home-select', home, 'Home Team'], ['away-select', away, 'Away Team']].forEach(([id, sel, label]) => {
      const s = $('#' + id);
      s.innerHTML = teams.map(t => `<option value="${esc(t)}"${t === sel ? ' selected' : ''}>${esc(t)}</option>`).join('');
      s.addEventListener('change', () => { predState[id === 'home-select' ? 'home' : 'away'] = s.value; });
    });
    $('#predict-btn').addEventListener('click', runPrediction);
    $('#home-select').addEventListener('change', () => { predState.home = $('#home-select').value; runPrediction(); });
    $('#away-select').addEventListener('change', () => { predState.away = $('#away-select').value; runPrediction(); });
    runPrediction();
  } catch (e) { toast('Failed to load: ' + e.message); }
}

async function runPrediction() {
  const home = $('#home-select').value, away = $('#away-select').value;
  if (!home || !away || home === away) return;
  predState.home = home; predState.away = away;
  const btn = $('#predict-btn'); if (btn) btn.disabled = true;
  try {
    const [res, h2h] = await Promise.all([
      fetchJSON('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ home_team: home, away_team: away }) }),
      fetchJSON('/api/h2h?home=' + encodeURIComponent(home) + '&away=' + encodeURIComponent(away)).catch(() => null),
    ]);
    renderPrediction(res, home, away);
    renderH2H(h2h);
  } catch (e) { toast('Prediction failed: ' + e.message); }
  if (btn) btn.disabled = false;
}

function renderPrediction(res, home, away) {
  $('#prediction-result').innerHTML = `
    <div class="outcome"><div class="label">${esc(home)}</div><div class="pct h">${fmtPct(res.home_win)}</div></div>
    <div class="outcome"><div class="label">Draw</div><div class="pct d">${fmtPct(res.draw)}</div></div>
    <div class="outcome"><div class="label">${esc(away)}</div><div class="pct a">${fmtPct(res.away_win)}</div></div>
    <div class="xg">Expected goals<span>${fmtNum(res.expected_home_goals)} - ${fmtNum(res.expected_away_goals)}</span></div>`;

  const ml = res.most_likely_score || {};
  let chips = '';
  chips += `<span class="chip chip-blue">Score ${ml.home}-${ml.away} (${ml.prob != null ? (ml.prob * 100).toFixed(1) : '--'}%)</span>`;
  if (res.over_1_5 != null) chips += `<span class="chip chip-gray">O1.5 ${fmtPct(res.over_1_5)}</span>`;
  if (res.over_2_5 != null) chips += `<span class="chip ${res.over_2_5 > 0.5 ? 'chip-green' : 'chip-gray'}">O2.5 ${fmtPct(res.over_2_5)}</span>`;
  if (res.over_3_5 != null) chips += `<span class="chip chip-gray">O3.5 ${fmtPct(res.over_3_5)}</span>`;
  if (res.btts != null) chips += `<span class="chip ${res.btts > 0.5 ? 'chip-green' : 'chip-gray'}">BTTS ${fmtPct(res.btts)}</span>`;
  if (res.clean_sheet_home != null) chips += `<span class="chip chip-blue">CS ${esc(home)} ${fmtPct(res.clean_sheet_home)}</span>`;
  if (res.clean_sheet_away != null) chips += `<span class="chip chip-blue">CS ${esc(away)} ${fmtPct(res.clean_sheet_away)}</span>`;
  const c80 = res.conformal_80 || [], c90 = res.conformal_90 || [];
  chips += `<span class="chip chip-yellow">90% set: ${c90.map(r => ({ H: 'Home', D: 'Draw', A: 'Away' })[r]).join(', ') || '--'}</span>`;
  $('#pred-chips').innerHTML = chips;

  // Score matrix
  const sm = res.score_matrix || [];
  if (sm.length) {
    const maxP = Math.max(...sm.flat().map(x => x || 0));
    let g = '<div class="heat-grid" style="grid-template-columns:auto repeat(' + sm.length + ',1fr)">';
    g += '<div class="heat-hdr"></div>' + sm.map((_, i) => `<div class="heat-hdr">${i}</div>`).join('');
    sm.forEach((row, hi) => {
      g += `<div class="heat-hdr">${hi}</div>`;
      row.forEach((p, ai) => {
        const alpha = maxP ? Math.max(0.04, (p || 0) / maxP * 0.85) : 0.04;
        g += `<div class="heat-cell" title="${hi}-${ai}: ${((p || 0) * 100).toFixed(1)}%" style="background:rgba(59,130,246,${alpha})">${p ? (p * 100).toFixed(0) : ''}</div>`;
      });
    });
    g += '</div>';
    $('#pred-heatmap').innerHTML = '<div style="font-size:12px;color:var(--muted);margin-bottom:8px">Home goals ↓ / Away goals →</div>' + g;
  } else { $('#pred-heatmap').innerHTML = '<div class="empty">No matrix.</div>'; }

  // Model breakdown
  const mb = res.model_breakdown || {};
  const order = [['dc', 'Dixon-Coles'], ['xgboost', 'XGBoost'], ['deep', 'Deep LSTM'], ['xgboost_nomarket', 'XGB (no market)'], ['ensemble', 'Ensemble'], ['market_blended', 'Market blend']];
  let b = '';
  order.forEach(([k, label]) => {
    if (!mb[k]) return;
    const [h, d, a] = mb[k];
    const seg = p => p * 100;
    b += `<div class="model-row"><div class="model-name">${label}</div>
      <div class="prob-bar" style="display:flex;overflow:hidden">
        <div class="prob-fill" style="width:${seg(h)}%;background:var(--green)"></div>
        <div class="prob-fill" style="width:${seg(d)}%;background:var(--yellow)"></div>
        <div class="prob-fill" style="width:${seg(a)}%;background:var(--red)"></div>
      </div></div>`;
  });
  b += '<div class="legend"><span><span class="dot" style="background:var(--green)"></span>Home</span><span><span class="dot" style="background:var(--yellow)"></span>Draw</span><span><span class="dot" style="background:var(--red)"></span>Away</span></div>';
  $('#pred-breakdown').innerHTML = b;

  // Top scores
  const ts = res.top_scores || [];
  $('#pred-topsc').innerHTML = '<h3 style="margin-top:4px">Most Likely Scores</h3>' +
    '<table><thead><tr><th>Score</th><th>Prob</th></tr></thead><tbody>' +
    ts.slice(0, 5).map(s => `<tr><td>${s.home} - ${s.away}</td><td>${(s.prob * 100).toFixed(1)}%</td></tr>`).join('') +
    '</tbody></table>';
}

function renderH2H(h2h) {
  const panel = $('#h2h-panel');
  if (!h2h || !h2h.history || !h2h.history.length) { panel.innerHTML = '<div class="empty">No head-to-head history.</div>'; return; }
  const hist = h2h.history;
  const sum = hist.reduce((acc, m) => {
    acc.n++;
    if (m.result === 'H') acc.h++;
    else if (m.result === 'D') acc.d++;
    else acc.a++;
    return acc;
  }, { n: 0, h: 0, d: 0, a: 0 });
  const p = h2h.prediction;
  let predHtml = '';
  if (p) {
    const ml = p.most_likely_score || {};
    predHtml = `<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
      <span class="chip chip-blue">Model ${fmtPct(p.home_win)} - ${fmtPct(p.draw)} - ${fmtPct(p.away_win)}</span>
      <span class="chip chip-yellow">xG ${fmtNum(p.expected_home_goals)} - ${fmtNum(p.expected_away_goals)}</span>
      <span class="chip chip-green">Score ${ml.home}-${ml.away}</span>
      <span class="chip chip-gray">O2.5 ${fmtPct(p.over_2_5)}</span>
      ${p.conformal_90 ? `<span class="chip chip-yellow">90% set: ${p.conformal_90.map(r => ({ H: 'Home', D: 'Draw', A: 'Away' })[r]).join(', ')}</span>` : ''}
    </div>`;
  }
  panel.innerHTML = `
    <div class="legend">
      <span>Last ${sum.n} meetings</span>
      <span><span class="dot" style="background:var(--green)"></span>Home ${sum.h} (${fmtPct(sum.h / sum.n)})</span>
      <span><span class="dot" style="background:var(--yellow)"></span>Draw ${sum.d} (${fmtPct(sum.d / sum.n)})</span>
      <span><span class="dot" style="background:var(--red)"></span>Away ${sum.a} (${fmtPct(sum.a / sum.n)})</span>
    </div>
    <div style="overflow-x:auto;"><table><thead><tr><th>Date</th><th>Season</th><th>Home</th><th>Away</th><th>Result</th></tr></thead><tbody>
    ${hist.map(m => `<tr><td>${shortDate(m.date)}</td><td>${esc(m.season)}</td><td>${esc(m.home_team)}</td><td>${esc(m.away_team)}</td><td><span class="badge badge-${m.result === 'H' ? 'h' : m.result === 'D' ? 'd' : 'a'}">${m.result}</span></td></tr>`).join('')}
    </tbody></table></div>${predHtml}`;
}

/* ---------- Season Sim ---------- */
async function loadSeason() {
  showLoading('#season-content');
  try {
    const res = await fetchJSON('/api/season');
    const teams = res.teams || [];
    const content = $('#season-content');
    if (!teams.length) { content.innerHTML = '<div class="empty">No season simulation yet. Run <code>predict-pl season</code>.</div>'; return; }

    const top = teams[0];
    const titleRace = teams.filter(t => (t.title_pct || 0) > 1);
    const relRace = teams.filter(t => (t.relegation_pct || 0) > 5);

    const h = document.createElement('div');
    h.innerHTML = `
      <div class="grid-3">
        <div class="card"><h3>Title Favourite</h3><div class="value green">${esc(top.team)}</div></div>
        <div class="card"><h3>Expected Champion Pts</h3><div class="value blue">${fmtNum(top.mean_pts)}</div></div>
        <div class="card"><h3>Relegation Zone</h3><div class="value red">${esc(relRace.slice(-3).map(t => t.team).join(', ') || '--')}</div></div>
      </div>
      <div class="card"><div id="sim-chart" class="chart-container" style="min-height:520px"></div></div>
      <div class="section-title">Full Season Projection</div>
      <div class="card">
        <div class="sim-table"><table><thead><tr>
          <th>#</th><th>Team</th><th>Exp Pts</th><th>Median</th><th>P10-P90</th>
          <th>Title</th><th>Top 4</th><th>Relegation</th>
        </tr></thead><tbody id="sim-table-body"></tbody></table></div>
      </div>`;
    content.innerHTML = '';
    content.appendChild(h);

    const y = teams.map(t => t.team);
    const colors = teams.map((t, i) => (t.title_pct || 0) > 10 ? '#22c55e' : (t.relegation_pct || 0) > 40 ? '#ef4444' : i < 5 ? '#3b82f6' : '#6b7fa3');
    Plotly.newPlot('sim-chart', [{
      x: teams.map(t => t.mean_pts), y, type: 'bar', orientation: 'h',
      marker: { color: colors },
      text: teams.map(t => t.mean_pts.toFixed(1)), textposition: 'outside',
      error_x: { type: 'data', symmetric: false, array: teams.map(t => t.p90_pts - t.mean_pts), arrayminus: teams.map(t => t.mean_pts - t.p10_pts), color: '#6b7fa3', thickness: 1.5 },
    }], plotBase({
      title: { text: 'Expected Final Points (with P10-P90 bands)', font: { size: 16, color: '#e2e8f0' } },
      height: 520, margin: { l: 140, r: 40, t: 40, b: 40 },
      yaxis: { autorange: 'reversed', gridcolor: 'transparent' }, xaxis: { gridcolor: '#263258', title: 'Points' },
    }), { responsive: true });

    $('#sim-table-body').innerHTML = teams.map((t, i) => {
      const fav = (t.title_pct || 0) > 10 ? ' <span class="badge badge-h">fav</span>' : '';
      const risk = (t.relegation_pct || 0) > 40 ? ' <span class="badge badge-a">risk</span>' : '';
      return `<tr><td>${i + 1}</td><td>${esc(t.team)}${fav}${risk}</td>
        <td><b>${fmtNum(t.mean_pts)}</b></td><td>${fmtNum(t.median_pts, 0)}</td>
        <td>${t.p10_pts != null ? t.p10_pts.toFixed(0) : '--'}-${t.p90_pts != null ? t.p90_pts.toFixed(0) : '--'}</td>
        <td>${fmtPct(t.title_pct / 100)}</td><td>${fmtPct(t.top4_pct / 100)}</td><td>${fmtPct(t.relegation_pct / 100)}</td></tr>`;
    }).join('');

    if (relRace.length) {
      const rdiv = document.createElement('div');
      rdiv.className = 'card'; rdiv.style.marginTop = '20px';
      rdiv.innerHTML = `<h3>Relegation Battleground</h3><div id="rel-chart" class="chart-container" style="min-height:${Math.max(240, relRace.length * 34)}px"></div>`;
      content.appendChild(rdiv);
      Plotly.newPlot('rel-chart', [{
        x: relRace.map(t => t.relegation_pct / 100), y: relRace.map(t => t.team), type: 'bar', orientation: 'h',
        marker: { color: '#ef4444' }, text: relRace.map(t => fmtPct(t.relegation_pct / 100)), textposition: 'outside',
      }], plotBase({ title: { text: 'Relegation Probability', font: { size: 14, color: '#e2e8f0' } }, height: Math.max(240, relRace.length * 34), xaxis: { gridcolor: '#263258', tickformat: ',.0%' }, yaxis: { autorange: 'reversed', gridcolor: 'transparent' } }), { responsive: true });
    }
  } catch (e) { toast('Failed to load season: ' + e.message); }
}

/* ---------- Model Performance ---------- */
async function loadPerformance() {
  showLoading('#performance-content');
  try {
    const res = await fetchJSON('/api/performance');
    const content = $('#performance-content');
    if (!res.n_matches) { content.innerHTML = '<div class="empty">' + (res.message || 'No walk-forward data.') + '</div>'; return; }

    const h = document.createElement('div');
    h.innerHTML = `
      <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Held-out matches</div><div class="kpi-value blue">${res.n_matches}</div></div>
        <div class="kpi"><div class="kpi-label">Ensemble Acc</div><div class="kpi-value green">${fmtPct(res.ens_acc)}</div></div>
        <div class="kpi"><div class="kpi-label">Dixon-Coles Acc</div><div class="kpi-value">${fmtPct(res.dc_acc)}</div></div>
        <div class="kpi"><div class="kpi-label">Ensemble RPS</div><div class="kpi-value yellow">${fmtNum(res.ens_rps, 4)}</div></div>
        <div class="kpi"><div class="kpi-label">Dixon-Coles RPS</div><div class="kpi-value">${fmtNum(res.dc_rps, 4)}</div></div>
      </div>
      <div class="grid-2">
        <div class="card"><h3>Accuracy by Season</h3><div id="perf-acc" class="chart-container"></div></div>
        <div class="card"><h3>RPS by Season (lower is better)</h3><div id="perf-rps" class="chart-container"></div></div>
        <div class="card"><h3>Calibration Curve</h3><div id="perf-cal" class="chart-container"></div></div>
        <div class="card"><h3>RPS Distribution</h3><div id="perf-hist" class="chart-container"></div></div>
      </div>
      <div class="section-title">Worst Predictions (high confidence, wrong)</div>
      <div class="card"><div class="sim-table"><table><thead><tr>
        <th>Date</th><th>Season</th><th>Match</th><th>Predicted</th><th>Actual</th>
        <th>H</th><th>D</th><th>A</th><th>Conf</th><th>RPS</th>
      </tr></thead><tbody id="perf-worst"></tbody></table></div></div>`;
    content.innerHTML = '';
    content.appendChild(h);

    const seas = res.by_season || [];
    const sNames = seas.map(s => s.season);
    Plotly.newPlot('perf-acc', [
      { x: sNames, y: seas.map(s => s.ens_acc), type: 'bar', name: 'Ensemble', marker: { color: '#3b82f6' } },
      { x: sNames, y: seas.map(s => s.dc_acc), type: 'bar', name: 'Dixon-Coles', marker: { color: '#6b7fa3' } },
    ], plotBase({ barmode: 'group', title: { text: 'Accuracy', font: { size: 14, color: '#e2e8f0' } }, yaxis: { gridcolor: '#263258', tickformat: ',.0%', range: [0, 1] } }), { responsive: true });
    Plotly.newPlot('perf-rps', [
      { x: sNames, y: seas.map(s => s.ens_rps), type: 'scatter', mode: 'lines+markers', name: 'Ensemble', line: { color: '#3b82f6' } },
      { x: sNames, y: seas.map(s => s.dc_rps), type: 'scatter', mode: 'lines+markers', name: 'Dixon-Coles', line: { color: '#f97316', dash: 'dot' } },
    ], plotBase({ title: { text: 'Ranked Probability Score', font: { size: 14, color: '#e2e8f0' } } }), { responsive: true });

    const cal = res.calibration || [];
    Plotly.newPlot('perf-cal', [
      { x: cal.map(c => c[0]), y: cal.map(c => c[1]), type: 'scatter', mode: 'lines+markers', name: 'Model', line: { color: '#22c55e' }, marker: { color: '#22c55e' } },
      { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'Perfect', line: { color: '#6b7fa3', dash: 'dash' } },
    ], plotBase({ title: { text: 'Predicted vs Actual (max prob)', font: { size: 14, color: '#e2e8f0' } }, xaxis: { gridcolor: '#263258', title: 'Predicted probability' }, yaxis: { gridcolor: '#263258', title: 'Actual frequency', tickformat: ',.0%' } }), { responsive: true });

    const rh = res.rps_hist || [[], []];
    Plotly.newPlot('perf-hist', [{
      x: rh[0], y: rh[1], type: 'bar', marker: { color: '#8b5cf6' },
    }], plotBase({ title: { text: 'Ensemble RPS histogram', font: { size: 14, color: '#e2e8f0' } }, xaxis: { gridcolor: '#263258', title: 'RPS' } }), { responsive: true });

    $('#perf-worst').innerHTML = (res.worst || []).map(w => `
      <tr><td>${shortDate(w.date)}</td><td>${esc(w.season)}</td>
      <td>${esc(w.home_team)} vs ${esc(w.away_team)}</td>
      <td><span class="badge badge-${w.predicted === 'H' ? 'h' : w.predicted === 'D' ? 'd' : 'a'}">${w.predicted}</span></td>
      <td><span class="badge badge-${w.result === 'H' ? 'h' : w.result === 'D' ? 'd' : 'a'}">${w.result}</span></td>
      <td>${fmtPct(w.ph)}</td><td>${fmtPct(w.pd)}</td><td>${fmtPct(w.pa)}</td>
      <td>${fmtPct(w.conf)}</td><td>${fmtNum(w.rps, 4)}</td></tr>`).join('') || '<tr><td colspan="10" class="empty">None</td></tr>';
  } catch (e) { toast('Failed to load performance: ' + e.message); }
}

/* ---------- Players (Fantasy Guide) ---------- */
const playerState = { season: '', team: '', position: '', stat: 'fantasy_pts', top: 25, q: '', sortDir: 'desc' };

function posBadge(pos) {
  if (!pos) return '';
  const u = pos.toUpperCase();
  if (u.includes('F')) return '<span class="pos-badge pos-fwd">FWD</span>';
  if (u.includes('M')) return '<span class="pos-badge pos-mid">MID</span>';
  if (u.includes('D') || u.includes('B')) return '<span class="pos-badge pos-def">DEF</span>';
  return '<span class="pos-badge pos-gk">GK</span>';
}

function fantasyTier(pts) {
  if (pts == null) return '';
  if (pts >= 150) return '<span class="chip chip-green">Elite</span>';
  if (pts >= 100) return '<span class="chip chip-blue">Strong</span>';
  if (pts >= 60) return '<span class="chip chip-gray">Solid</span>';
  return '<span class="chip chip-gray" style="opacity:.6">Budget</span>';
}

function xgDelta(actual, expected) {
  if (actual == null || expected == null) return '--';
  const d = actual - expected;
  const cls = d > 0 ? 'color:var(--green)' : d < 0 ? 'color:var(--red)' : '';
  return `<span style="${cls}">${d > 0 ? '+' : ''}${d.toFixed(1)}</span>`;
}

async function loadPlayers() {
  showLoading('#players-content');
  try {
    const [seas, teamsRes] = await Promise.all([fetchJSON('/api/seasons'), fetchJSON('/api/teams')]);
    const seasons = seas.player_seasons || [];
    if (!playerState.season && seasons.length) playerState.season = seasons[seasons.length - 1];

    const h = document.createElement('div');
    h.innerHTML = `
      <div class="section-title">Fantasy Premier League — Player Guide</div>
      <div class="card">
        <div class="toolbar">
          <div><label>Season</label><select id="pl-season"></select></div>
          <div><label>Team</label><select id="pl-team"><option value="">All teams</option></select></div>
          <div><label>Position</label><select id="pl-position">
            <option value="">All</option><option value="F">Forward</option><option value="M">Midfielder</option><option value="D">Defender</option><option value="G">Goalkeeper</option>
          </select></div>
          <div><label>Search</label><input type="text" id="pl-search" placeholder="Player name..." value="${esc(playerState.q)}"></div>
          <div><label>Sort by</label><select id="pl-stat">
            <option value="fantasy_pts">Fantasy Pts</option><option value="xG">xG</option><option value="xA">xA</option>
            <option value="goals">Goals</option><option value="assists">Assists</option>
            <option value="npxG">npxG</option><option value="shots">Shots</option>
          </select></div>
          <div><label>Show</label><select id="pl-top">
            <option value="10">Top 10</option><option value="25" selected>Top 25</option><option value="50">Top 50</option><option value="100">Top 100</option><option value="999">All</option>
          </select></div>
          <button id="pl-go">Load</button>
        </div>
      </div>

      <div class="section-title">Top Picks by Position</div>
      <div class="grid-2" id="pl-top-picks"><div class="card"><div class="spinner"></div></div></div>

      <div class="section-title">Player Leaderboard</div>
      <div class="card">
        <div id="pl-chart" class="chart-container" style="min-height:420px"></div>
        <div class="sim-table"><table><thead><tr>
          <th>#</th><th>Player</th><th>Team</th><th>Pos</th><th>FPL Pts</th><th>Tier</th>
          <th>G</th><th>A</th><th>xG</th><th>xA</th><th>G-xG</th><th>npxG</th>
          <th>Mins</th><th>90s</th><th>Shots</th>
        </tr></thead><tbody id="pl-body"></tbody></table></div>
      </div>

      <div id="pl-detail"></div>`;
    $('#players-content').innerHTML = '';
    $('#players-content').appendChild(h);

    const ss = $('#pl-season');
    ss.innerHTML = seasons.map(s => `<option value="${s}"${s === playerState.season ? ' selected' : ''}>${s}</option>`).join('');
    const ts = $('#pl-team');
    ts.innerHTML = '<option value="">All teams</option>' + teamsRes.teams.map(t => `<option value="${esc(t)}"${t === playerState.team ? ' selected' : ''}>${esc(t)}</option>`).join('');
    $('#pl-stat').value = playerState.stat; $('#pl-top').value = String(playerState.top);
    $('#pl-position').value = playerState.position;

    const triggerLoad = () => {
      playerState.season = ss.value; playerState.team = ts.value;
      playerState.stat = $('#pl-stat').value; playerState.top = +$('#pl-top').value;
      playerState.position = $('#pl-position').value; playerState.q = $('#pl-search').value.trim();
      loadPlayerBoard();
    };
    $('#pl-go').addEventListener('click', triggerLoad);
    $('#pl-search').addEventListener('keyup', e => { if (e.key === 'Enter') triggerLoad(); });
    $('#pl-stat').addEventListener('change', triggerLoad);
    loadPlayerBoard();
  } catch (e) { toast('Failed to load players: ' + e.message); }
}

async function loadPlayerBoard() {
  try {
    const [allRes, picksRes] = await Promise.all([
      fetchJSON('/api/players?season=' + encodeURIComponent(playerState.season) + '&team=' + encodeURIComponent(playerState.team) + '&position=' + encodeURIComponent(playerState.position) + '&q=' + encodeURIComponent(playerState.q) + '&stat=' + playerState.stat + '&top=' + playerState.top + '&all=true'),
      fetchJSON('/api/players?season=' + encodeURIComponent(playerState.season) + '&stat=fantasy_pts&top=5&all=false'),
    ]);

    const allPlayers = allRes.players || [];
    const topPicks = picksRes.players || [];

    renderTopPicks(allPlayers, topPicks);
    renderLeaderboardChart(allPlayers);
    renderLeaderboardTable(allPlayers);
  } catch (e) { toast('Failed to load leaderboard: ' + e.message); }
}

function renderTopPicks(allPlayers, topPicks) {
  const positions = [
    { key: 'F', label: 'Forwards', icon: '&#9917;', color: 'var(--green)' },
    { key: 'M', label: 'Midfielders', icon: '&#9918;', color: 'var(--accent)' },
    { key: 'D', label: 'Defenders', icon: '&#9899;', color: 'var(--yellow)' },
    { key: 'G', label: 'Goalkeepers', icon: '&#9899;', color: 'var(--muted)' },
  ];
  const grid = $('#pl-top-picks');
  grid.innerHTML = positions.map(pos => {
    const picks = allPlayers.filter(p => {
      const u = (p.position || '').toUpperCase();
      return u.includes(pos.key) || (pos.key === 'D' && (u.includes('B')));
    }).slice(0, 5);
    return `<div class="card">
      <h3 style="color:${pos.color};margin-bottom:12px">${pos.icon} ${pos.label}</h3>
      ${picks.length ? '<table><thead><tr><th>Player</th><th>Team</th><th>Pts</th><th>G</th><th>A</th></tr></thead><tbody>' +
        picks.map(p => `<tr>
          <td><span class="player-link" onclick="openPlayer('${esc(p.player_id)}','${esc(p.player_name)}')">${esc(p.player_name)}</span></td>
          <td>${esc(p.team)}</td>
          <td><b>${fmtNum(p.fantasy_pts, 0)}</b></td>
          <td>${p.goals || 0}</td><td>${p.assists || 0}</td>
        </tr>`).join('') + '</tbody></table>' : '<div class="empty" style="padding:16px">No data for this position.</div>'}
    </div>`;
  }).join('');
}

function renderLeaderboardChart(allPlayers) {
  const chartPlayers = allPlayers.slice(0, 25);
  if (!chartPlayers.length) { Plotly.purge('pl-chart'); return; }
  const statLabel = { fantasy_pts: 'Fantasy Points', xG: 'xG', xA: 'xA', goals: 'Goals', assists: 'Assists', npxG: 'npxG', shots: 'Shots' };
  Plotly.newPlot('pl-chart', [{
    x: chartPlayers.map(p => p[playerState.stat] || 0),
    y: chartPlayers.map(p => p.player_name),
    type: 'bar', orientation: 'h',
    marker: {
      color: chartPlayers.map(p => {
        const u = (p.position || '').toUpperCase();
        if (u.includes('F')) return '#22c55e';
        if (u.includes('M')) return '#3b82f6';
        if (u.includes('D') || u.includes('B')) return '#eab308';
        return '#6b7fa3';
      }),
    },
    text: chartPlayers.map(p => fmtNum(p[playerState.stat], 1)), textposition: 'outside',
  }], plotBase({
    title: { text: `Top 25 by ${statLabel[playerState.stat] || playerState.stat} (${playerState.season})`, font: { size: 14, color: '#e2e8f0' } },
    height: Math.max(420, chartPlayers.length * 28), margin: { l: 150, r: 50, t: 40, b: 40 },
    yaxis: { autorange: 'reversed', gridcolor: 'transparent' },
    showlegend: false,
  }), { responsive: true });
}

function renderLeaderboardTable(allPlayers) {
  const tbody = $('#pl-body');
  if (!allPlayers.length) { tbody.innerHTML = '<tr><td colspan="15" class="empty">No players match filters.</td></tr>'; return; }
  tbody.innerHTML = allPlayers.map((p, i) => `<tr onclick="openPlayer('${esc(p.player_id)}','${esc(p.player_name)}')" style="cursor:pointer">
    <td style="color:var(--muted)">${i + 1}</td>
    <td><span class="player-link">${esc(p.player_name)}</span></td>
    <td>${esc(p.team)}</td>
    <td>${posBadge(p.position)}</td>
    <td><b class="fpl-pts">${fmtNum(p.fantasy_pts, 0)}</b></td>
    <td>${fantasyTier(p.fantasy_pts)}</td>
    <td>${p.goals || 0}</td>
    <td>${p.assists || 0}</td>
    <td>${fmtNum(p.xG)}</td>
    <td>${fmtNum(p.xA)}</td>
    <td>${xgDelta(p.goals, p.xG)}</td>
    <td>${fmtNum(p.npxG)}</td>
    <td>${p.minutes != null ? Math.round(p.minutes).toLocaleString() : '--'}</td>
    <td>${fmtNum(p.games_90, 1)}</td>
    <td>${p.shots || 0}</td>
  </tr>`).join('');
}

async function openPlayer(pid, name) {
  const detail = $('#pl-detail');
  detail.innerHTML = `<div class="section-title">${esc(name)} — Season Breakdown</div><div class="card"><div class="spinner"></div></div>`;
  try {
    const res = await fetchJSON('/api/players/' + pid);
    const rows = res.matches || [];
    const tot = rows.reduce((a, r) => ({
      xG: a.xG + (r.xG || 0), xA: a.xA + (r.xA || 0),
      g: a.g + (r.goals || 0), a: a.a + (r.assists || 0),
      m: a.m + (r.minutes || 0), s: a.s + (r.shots || 0),
      kp: a.kp + (r.key_passes || 0), yc: a.yc + (r.yellow_card || 0),
    }), { xG: 0, xA: 0, g: 0, a: 0, m: 0, s: 0, kp: 0, yc: 0 });

    const pos = rows[0] ? (rows[0].team || '') : '';
    const rolling = [];
    let acc = 0;
    rows.forEach(r => { acc += (r.xG || 0); rolling.push(acc); });

    detail.innerHTML = `
      <div class="section-title">${esc(name)} — Season Breakdown</div>
      <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Minutes</div><div class="kpi-value blue">${Math.round(tot.m).toLocaleString()}</div></div>
        <div class="kpi"><div class="kpi-label">Goals</div><div class="kpi-value green">${tot.g}</div></div>
        <div class="kpi"><div class="kpi-label">Assists</div><div class="kpi-value yellow">${tot.a}</div></div>
        <div class="kpi"><div class="kpi-label">xG</div><div class="kpi-value blue">${fmtNum(tot.xG)}</div></div>
        <div class="kpi"><div class="kpi-label">xA</div><div class="kpi-value">${fmtNum(tot.xA)}</div></div>
        <div class="kpi"><div class="kpi-label">G-xG</div><div class="kpi-value" style="color:${tot.g - tot.xG > 0 ? 'var(--green)' : 'var(--red)'}">${(tot.g - tot.xG).toFixed(1)}</div></div>
        <div class="kpi"><div class="kpi-label">Shots</div><div class="kpi-value">${tot.s}</div></div>
        <div class="kpi"><div class="kpi-label">Key Passes</div><div class="kpi-value">${tot.kp}</div></div>
        <div class="kpi"><div class="kpi-label">Yellows</div><div class="kpi-value">${tot.yc}</div></div>
      </div>
      <div class="card" style="margin-bottom:16px"><div id="pl-xg-chart" class="chart-container" style="min-height:320px"></div></div>
      <div class="card"><div class="sim-table" style="max-height:400px"><table><thead><tr>
        <th>Date</th><th>Team</th><th>Mins</th><th>G</th><th>A</th><th>xG</th><th>xA</th><th>Shots</th><th>KP</th><th>YC</th>
      </tr></thead><tbody>${rows.slice().reverse().map(r => `<tr>
        <td>${shortDate(r.date)}</td><td>${esc(r.team)}</td>
        <td>${r.minutes != null ? Math.round(r.minutes) : '--'}</td>
        <td>${r.goals || 0}</td><td>${r.assists || 0}</td>
        <td>${fmtNum(r.xG)}</td><td>${fmtNum(r.xA)}</td>
        <td>${r.shots || 0}</td><td>${r.key_passes || 0}</td>
        <td>${r.yellow_card || 0}</td></tr>`).join('')}
      </tbody></table></div></div>`;

    Plotly.newPlot('pl-xg-chart', [
      { x: rows.map(r => shortDate(r.date)), y: rows.map(r => r.xG || 0), type: 'bar', name: 'xG', marker: { color: '#3b82f6' } },
      { x: rows.map(r => shortDate(r.date)), y: rows.map(r => r.goals || 0), type: 'bar', name: 'Goals', marker: { color: '#22c55e', opacity: .6 } },
      { x: rows.map(r => shortDate(r.date)), y: rolling, type: 'scatter', mode: 'lines', name: 'Cumulative xG', line: { color: '#eab308' } },
    ], plotBase({
      title: { text: `${name} — ${tot.g}G ${tot.a}A | ${fmtNum(tot.xG)} xG ${fmtNum(tot.xA)} xA (${tot.m} mins)`, font: { size: 14, color: '#e2e8f0' } },
      barmode: 'group', margin: { l: 50, r: 30, t: 50, b: 60 }, xaxis: { gridcolor: '#263258', nticks: 12 },
    }), { responsive: true });

    detail.scrollIntoView({ behavior: 'smooth' });
  } catch (e) { toast('Failed to load player: ' + e.message); }
}

/* ---------- Live FPL predictions ---------- */
async function loadPlayers() {
  showLoading('#players-content');
  try {
    const result = await fetchJSON('/api/fpl/picks?top=100');
    renderFplPicks(result);
  } catch (e) {
    toast('Failed to load official FPL data: ' + e.message);
  }
}

function renderFplPicks(result, position = '') {
  const players = (result.players || []).filter(p => !position || p.position === position);
  const fixtureLabel = (p) => (p.fixtures || []).map(f => {
    const side = f.home ? 'vs' : '@';
    const model = f.win_probability == null ? '' : ` · ${fmtPct(f.win_probability)} win`;
    return `${side} ${esc(f.opponent)}${model}`;
  }).join('<br>') || 'No fixture';
  const availability = (p) => p.availability == null || p.availability >= 100
    ? '<span class="chip chip-green">Available</span>'
    : `<span class="chip chip-yellow">${p.availability}% available</span>`;

  $('#players-content').innerHTML = `
    <div class="section-title">FPL Next Gameweek Picks — GW ${result.gameweek}</div>
    <div class="card" style="margin-bottom:16px">
      <div style="color:var(--muted);margin-bottom:12px">Hybrid xP blends official FPL xP (55%) with PL Predict's expected goals, win and clean-sheet probabilities (45%). Official xP remains visible for comparison.</div>
      <div class="toolbar"><div><label>Position</label><select id="fpl-position">
        <option value="">All positions</option><option value="GK">Goalkeepers</option><option value="DEF">Defenders</option><option value="MID">Midfielders</option><option value="FWD">Forwards</option>
      </select></div><div><label>Deadline</label><div style="padding-top:8px">${shortDate(result.deadline)}</div></div></div>
    </div>
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">Data source</div><div class="kpi-value blue">Official FPL</div></div>
      <div class="kpi"><div class="kpi-label">Gameweek</div><div class="kpi-value green">${result.gameweek}</div></div>
      <div class="kpi"><div class="kpi-label">Picks shown</div><div class="kpi-value yellow">${players.length}</div></div>
    </div>
    <div class="card"><div class="sim-table"><table><thead><tr>
      <th>#</th><th>Player</th><th>Team</th><th>Pos</th><th>Price</th><th>Hybrid xP</th><th>Official xP</th><th>Model xP</th><th>Form</th><th>Total</th><th>Owned</th><th>Fixture / model outlook</th><th>Status</th>
    </tr></thead><tbody>
      ${players.map((p, i) => `<tr>
        <td style="color:var(--muted)">${i + 1}</td><td><b>${esc(p.name)}</b></td><td>${esc(p.team)}</td><td>${posBadge(p.position)}</td>
        <td>£${fmtNum(p.price, 1)}m</td><td><b class="fpl-pts">${fmtNum(p.hybrid_xp, 1)}</b></td><td>${fmtNum(p.official_xp, 1)}</td><td>${fmtNum(p.model_xp, 1)}</td><td>${fmtNum(p.form, 1)}</td>
        <td>${p.total_points}</td><td>${fmtNum(p.ownership, 1)}%</td><td>${fixtureLabel(p)}</td><td>${availability(p)}</td>
      </tr>`).join('') || '<tr><td colspan="13" class="empty">No picks in this position.</td></tr>'}
    </tbody></table></div></div>`;

  $('#fpl-position').value = position;
  $('#fpl-position').addEventListener('change', e => renderFplPicks(result, e.target.value));
}

/* ---------- Shot Maps ---------- */
const shotState = { season: '2024', team: '', player: '' };

async function loadShots() {
  showLoading('#shots-content');
  try {
    const [seas, teamsRes] = await Promise.all([fetchJSON('/api/seasons'), fetchJSON('/api/teams')]);
    const seasons = seas.player_seasons || [];
    const h = document.createElement('div');
    h.innerHTML = `
      <div class="section-title">Shot Maps (xG locations)</div>
      <div class="card">
        <div class="toolbar">
          <div><label>Season</label><select id="sh-season"></select></div>
          <div><label>Team</label><select id="sh-team"><option value="">All teams</option></select></div>
          <div><label>Player</label><input type="text" id="sh-player" placeholder="Player name" value="${esc(shotState.player)}"></div>
          <button id="sh-go">Load</button>
        </div>
        <div id="sh-kpis" class="kpi-row" style="margin-top:12px"></div>
        <div id="sh-pitch" class="chart-container" style="min-height:560px"></div>
        <div style="margin-top:12px"><h3>Top shooters</h3><div class="sim-table" style="max-height:300px"><table><thead><tr><th>Player</th><th>Shots</th><th>Goals</th><th>xG</th><th>G-xG</th></tr></thead><tbody id="sh-top"></tbody></table></div></div>
      </div>`;
    $('#shots-content').innerHTML = '';
    $('#shots-content').appendChild(h);

    const ss = $('#sh-season');
    ss.innerHTML = seasons.map(s => `<option value="${s}"${s === shotState.season ? ' selected' : ''}>${s}</option>`).join('');
    const ts = $('#sh-team');
    ts.innerHTML = '<option value="">All teams</option>' + teamsRes.teams.map(t => `<option value="${esc(t)}"${t === shotState.team ? ' selected' : ''}>${esc(t)}</option>`).join('');
    $('#sh-go').addEventListener('click', () => {
      shotState.season = ss.value; shotState.team = ts.value; shotState.player = $('#sh-player').value.trim();
      loadShotBoard();
    });
    loadShotBoard();
  } catch (e) { toast('Failed to load shots: ' + e.message); }
}

function pitchShapes() {
  const S = (x0, y0, x1, y1, c = '#263258') => ({ type: 'rect', xref: 'x', yref: 'y', x0, y0, x1, y1, line: { color: c, width: 1.5 }, fillcolor: 'rgba(0,0,0,0)' });
  return [
    S(0, 0, 1, 1),
    S(0, 0.206, 0.165, 0.794), S(0.835, 0.206, 1, 0.794),
    S(0, 0.365, 0.055, 0.635), S(0.945, 0.365, 1, 0.635),
    { type: 'line', x0: 0.5, y0: 0, x1: 0.5, y1: 1, line: { color: '#263258', width: 1.5 } },
    { type: 'circle', xref: 'x', yref: 'y', x0: 0.5 - 0.12, y0: 0.5 - 0.12, x1: 0.5 + 0.12, y1: 0.5 + 0.12, line: { color: '#263258', width: 1.5 } },
    { type: 'circle', xref: 'x', yref: 'y', x0: 0.105 - 0.012, y0: 0.5 - 0.012, x1: 0.105 + 0.012, y1: 0.5 + 0.012, line: { color: '#263258', width: 1.5 }, fillcolor: '#263258' },
    { type: 'circle', xref: 'x', yref: 'y', x0: 0.895 - 0.012, y0: 0.5 - 0.012, x1: 0.895 + 0.012, y1: 0.5 + 0.012, line: { color: '#263258', width: 1.5 }, fillcolor: '#263258' },
  ];
}

const SHOT_COLORS = { 'Goal': '#22c55e', 'SavedShot': '#eab308', 'MissedShots': '#ef4444', 'BlockedShot': '#6b7fa3', 'ShotOnPost': '#f97316' };
const SHOT_LABEL = { 'Goal': 'Goal', 'SavedShot': 'Saved', 'MissedShots': 'Missed', 'BlockedShot': 'Blocked', 'ShotOnPost': 'Post' };

async function loadShotBoard() {
  const url = '/api/shots?season=' + encodeURIComponent(shotState.season) + '&team=' + encodeURIComponent(shotState.team) + '&player=' + encodeURIComponent(shotState.player) + '&limit=20000';
  try {
    const res = await fetchJSON(url);
    const shots = res.shots || [];
    if (!shots.length) { $('#sh-kpis').innerHTML = '<div class="empty">No shots for this filter.</div>'; Plotly.purge('sh-pitch'); return; }

    const byRes = {};
    shots.forEach(s => { byRes[s.result] = (byRes[s.result] || 0) + 1; });
    const totalXg = shots.reduce((a, s) => a + (s.xG || 0), 0);
    $('#sh-kpis').innerHTML = `
      <div class="kpi"><div class="kpi-label">Shots</div><div class="kpi-value blue">${shots.length.toLocaleString()}</div></div>
      <div class="kpi"><div class="kpi-label">Goals</div><div class="kpi-value green">${byRes.Goal || 0}</div></div>
      <div class="kpi"><div class="kpi-label">Total xG</div><div class="kpi-value yellow">${fmtNum(totalXg)}</div></div>
      <div class="kpi"><div class="kpi-label">Shot conversion</div><div class="kpi-value">${fmtPct((byRes.Goal || 0) / shots.length)}</div></div>`;

    const traces = Object.keys(SHOT_COLORS).map(k => ({
      x: shots.filter(s => s.result === k).map(s => s.X),
      y: shots.filter(s => s.result === k).map(s => s.Y),
      mode: 'markers', name: SHOT_LABEL[k], marker: {
        color: SHOT_COLORS[k], size: shots.filter(s => s.result === k).map(s => Math.min(16, Math.max(4, (s.xG || 0) * 18))),
        opacity: 0.7, line: { color: 'rgba(255,255,255,0.2)', width: 0.5 },
      }, text: shots.filter(s => s.result === k).map(s => `${s.player_name}<br>${s.minute}' ${s.situation} ${s.shotType}<br>xG ${fmtNum(s.xG)}`), hoverinfo: 'text',
    }));

    Plotly.newPlot('sh-pitch', traces, plotBase({
      title: { text: `Shots — ${shotState.team || 'All teams'} ${shotState.season}${shotState.player ? ' · ' + shotState.player : ''}`, font: { size: 15, color: '#e2e8f0' } },
      shapes: pitchShapes(), height: 560, margin: { l: 40, r: 20, t: 50, b: 40 },
      xaxis: { range: [0, 1], showticklabels: false, gridcolor: 'transparent', fixedrange: true },
      yaxis: { range: [0, 1], showticklabels: false, gridcolor: 'transparent', fixedrange: true, scaleanchor: 'x', scaleratio: 1 },
      hovermode: 'closest', showlegend: true, legend: { orientation: 'h', y: 1.05 },
    }), { responsive: true });

    const byPlayer = {};
    shots.forEach(s => {
      if (!byPlayer[s.player_name]) byPlayer[s.player_name] = { shots: 0, goals: 0, xg: 0 };
      byPlayer[s.player_name].shots++; byPlayer[s.player_name].xg += s.xG || 0;
      if (s.result === 'Goal') byPlayer[s.player_name].goals++;
    });
    const topP = Object.entries(byPlayer).sort((a, b) => b[1].shots - a[1].shots).slice(0, 15);
    $('#sh-top').innerHTML = topP.map(([name, v]) => `<tr><td>${esc(name)}</td><td>${v.shots}</td><td>${v.goals}</td><td>${fmtNum(v.xg)}</td><td style="color:${v.goals - v.xg > 0 ? 'var(--green)' : 'var(--red)'}">${(v.goals - v.xg).toFixed(1)}</td></tr>`).join('');
  } catch (e) { toast('Failed to load shots: ' + e.message); }
}

/* ---------- Teams ---------- */
const teamState = { name: '' };

async function loadTeamsView(preselected) {
  showLoading('#teams-content');
  try {
    const teamsRes = await fetchJSON('/api/teams');
    const name = teamState.name || preselected || teamsRes.teams[0] || '';
    teamState.name = name;
    const h = document.createElement('div');
    h.innerHTML = `
      <div class="section-title">Team Analysis</div>
      <div class="card">
        <div class="toolbar">
          <div><label>Team</label><select id="tm-select"></select></div>
          <button id="tm-go">Load</button>
        </div>
        <div id="tm-kpis" class="kpi-row" style="margin-top:12px"></div>
        <div class="grid-2">
          <div class="card"><h3>Recent Form (last 20)</h3><div id="tm-form" class="chart-container" style="min-height:300px"></div></div>
          <div class="card"><h3>Home / Away Splits</h3><div id="tm-splits" class="chart-container" style="min-height:300px"></div></div>
          <div class="card"><h3>Goals per match</h3><div id="tm-goals" class="chart-container" style="min-height:300px"></div></div>
          <div class="card"><h3>Avg Attendance by season</h3><div id="tm-att" class="chart-container" style="min-height:300px"></div></div>
        </div>
        <div style="margin-top:16px"><h3>Recent Results</h3><div class="sim-table" style="max-height:360px"><table><thead><tr><th>Date</th><th>Home</th><th>Away</th><th>Score</th><th>xG</th><th>Result</th></tr></thead><tbody id="tm-recent"></tbody></table></div></div>
      </div>`;
    $('#teams-content').innerHTML = '';
    $('#teams-content').appendChild(h);
    const sel = $('#tm-select');
    sel.innerHTML = teamsRes.teams.map(t => `<option value="${esc(t)}"${t === name ? ' selected' : ''}>${esc(t)}</option>`).join('');
    $('#tm-go').addEventListener('click', () => { teamState.name = sel.value; navTo('teams/' + encodeURIComponent(sel.value)); });
    sel.addEventListener('change', () => { teamState.name = sel.value; navTo('teams/' + encodeURIComponent(sel.value)); });
    loadTeamData(name);
  } catch (e) { toast('Failed to load teams: ' + e.message); }
}

async function loadTeamData(name) {
  try {
    const res = await fetchJSON('/api/team/' + encodeURIComponent(name));
    const rec = res.recent || [];
    const sh = res.splits.home || {}, sa = res.splits.away || {};
    const total = (sh.n || 0) + (sa.n || 0);
    const wins = (sh.w || 0) + (sa.w || 0);
    $('#tm-kpis').innerHTML = `
      <div class="kpi"><div class="kpi-label">Matches</div><div class="kpi-value blue">${total.toLocaleString()}</div></div>
      <div class="kpi"><div class="kpi-label">Win rate</div><div class="kpi-value green">${total ? fmtPct(wins / total) : '--'}</div></div>
      <div class="kpi"><div class="kpi-label">Home avg GF</div><div class="kpi-value">${fmtNum(sh.avg_gf)}</div></div>
      <div class="kpi"><div class="kpi-label">Away avg GF</div><div class="kpi-value">${fmtNum(sa.avg_gf)}</div></div>
      <div class="kpi"><div class="kpi-label">Home win%</div><div class="kpi-value yellow">${sh.n ? fmtPct(sh.w / sh.n) : '--'}</div></div>`;

    const dates = rec.map(r => shortDate(r.date));
    Plotly.newPlot('tm-form', [
      { x: dates, y: rec.map(r => r.pts), type: 'scatter', mode: 'lines+markers', name: 'Points', line: { color: '#3b82f6' }, fill: 'tozeroy', fillcolor: 'rgba(59,130,246,.12)' },
      { x: dates, y: rec.map(r => r.xg_f), type: 'scatter', mode: 'lines', name: 'xG for', line: { color: '#22c55e', dash: 'dot' } },
      { x: dates, y: rec.map(r => r.xg_a), type: 'scatter', mode: 'lines', name: 'xG against', line: { color: '#ef4444', dash: 'dot' } },
    ], plotBase({ title: { text: `${name} — last 20`, font: { size: 14, color: '#e2e8f0' } }, margin: { l: 50, r: 30, t: 40, b: 60 }, xaxis: { gridcolor: '#263258', nticks: 12 } }), { responsive: true });

    const venues = [['home', sh, 'var(--green)'], ['away', sa, 'var(--orange)']];
    Plotly.newPlot('tm-splits', [{
      x: venues.flatMap(([v, s]) => [v + ' W', v + ' D', v + ' L']),
      y: venues.flatMap(([v, s]) => [s.w || 0, s.d || 0, s.l || 0]),
      type: 'bar', marker: { color: ['#22c55e', '#eab308', '#ef4444', '#22c55e', '#eab308', '#ef4444'] },
      text: venues.flatMap(([v, s]) => [s.w || 0, s.d || 0, s.l || 0]), textposition: 'outside',
    }], plotBase({ title: { text: 'W / D / L by venue', font: { size: 14, color: '#e2e8f0' } }, margin: { l: 50, r: 30, t: 40, b: 50 }, yaxis: { gridcolor: '#263258', title: 'Matches' } }), { responsive: true });

    const gh = res.goals_hist || [];
    Plotly.newPlot('tm-goals', [{
      x: gh.map(g => g.total_goals), y: gh.map(g => g.len), type: 'bar', marker: { color: '#8b5cf6' },
    }], plotBase({ title: { text: 'Total goals in team matches', font: { size: 14, color: '#e2e8f0' } }, margin: { l: 50, r: 30, t: 40, b: 50 }, xaxis: { gridcolor: '#263258', title: 'Total goals', dtick: 1 } }), { responsive: true });

    const att = (res.attendance || []).filter(a => a.avg_attendance != null);
    if (att.length) {
      Plotly.newPlot('tm-att', [{
        x: att.map(a => a.season), y: att.map(a => a.avg_attendance), type: 'scatter', mode: 'lines+markers', line: { color: '#3b82f6' }, fill: 'tozeroy', fillcolor: 'rgba(59,130,246,.1)',
      }], plotBase({ title: { text: 'Avg attendance', font: { size: 14, color: '#e2e8f0' } }, margin: { l: 60, r: 30, t: 40, b: 50 }, yaxis: { gridcolor: '#263258', title: 'Attendance' } }), { responsive: true });
    } else { Plotly.purge('tm-att'); $('#tm-att').innerHTML = '<div class="empty">Attendance not captured for this team.</div>'; }

    $('#tm-recent').innerHTML = rec.map(r => `<tr>
      <td>${shortDate(r.date)}</td><td>${esc(r.home_team)}</td><td>${esc(r.away_team)}</td>
      <td><b>${r.gf != null ? r.gf.toFixed(0) : '-'} - ${r.ga != null ? r.ga.toFixed(0) : '-'}</b></td>
      <td>${r.xg_f != null ? fmtNum(r.xg_f) : '--'} - ${r.xg_a != null ? fmtNum(r.xg_a) : '--'}</td>
      <td><span class="badge badge-${r.result === 'H' ? 'h' : r.result === 'D' ? 'd' : 'a'}">${r.result}</span> <span class="badge badge-gray">${r.venue}</span></td></tr>`).join('');
  } catch (e) { toast('Failed to load team data: ' + e.message); }
}

/* ---------- Referees ---------- */
async function loadReferees() {
  showLoading('#referees-content');
  try {
    const res = await fetchJSON('/api/referees?min_matches=30');
    const refs = res.referees || [];
    const h = document.createElement('div');
    h.innerHTML = `
      <div class="section-title">Referee Analytics</div>
      <div class="grid-2">
        <div class="card"><h3>Goals per match by referee</h3><div id="ref-goals" class="chart-container"></div></div>
        <div class="card"><h3>Home win bias</h3><div id="ref-bias" class="chart-container"></div></div>
      </div>
      <div class="card"><div class="sim-table"><table><thead><tr>
        <th>Referee</th><th>Matches</th><th>Avg Goals</th><th>Avg Home</th><th>Avg Away</th>
        <th>Home Win%</th><th>Draw%</th><th>Avg Att</th>
      </tr></thead><tbody id="ref-body"></tbody></table></div></div>`;
    $('#referees-content').innerHTML = '';
    $('#referees-content').appendChild(h);

    const top = refs.slice(0, 25);
    const names = top.map(r => r.Referee);
    Plotly.newPlot('ref-goals', [
      { x: names, y: top.map(r => r.avg_goals), type: 'bar', name: 'Total', marker: { color: '#8b5cf6' } },
      { x: names, y: top.map(r => r.avg_home_goals), type: 'bar', name: 'Home', marker: { color: '#22c55e' } },
      { x: names, y: top.map(r => r.avg_away_goals), type: 'bar', name: 'Away', marker: { color: '#ef4444' } },
    ], plotBase({ title: { text: 'Top 25 by matches', font: { size: 14, color: '#e2e8f0' } }, barmode: 'group', margin: { l: 50, r: 30, t: 40, b: 90 }, xaxis: { gridcolor: '#263258', tickangle: 45 } }), { responsive: true });

    const avg = refs.reduce((a, r) => a + r.home_win_rate, 0) / (refs.length || 1);
    Plotly.newPlot('ref-bias', [{
      x: names, y: top.map(r => r.home_win_rate), type: 'bar', marker: {
        color: top.map(r => r.home_win_rate > avg ? '#ef4444' : '#3b82f6'),
      },
    }], plotBase({
      title: { text: `Home win rate (league avg ${fmtPct(avg)})`, font: { size: 14, color: '#e2e8f0' } },
      margin: { l: 50, r: 30, t: 40, b: 90 }, xaxis: { gridcolor: '#263258', tickangle: 45 }, yaxis: { gridcolor: '#263258', tickformat: ',.0%' },
    }), { responsive: true });

    $('#ref-body').innerHTML = refs.map(r => `<tr>
      <td><b>${esc(r.Referee)}</b></td><td>${r.n}</td><td>${fmtNum(r.avg_goals)}</td>
      <td>${fmtNum(r.avg_home_goals)}</td><td>${fmtNum(r.avg_away_goals)}</td>
      <td>${fmtPct(r.home_win_rate)}</td><td>${fmtPct(r.draw_rate)}</td>
      <td>${r.avg_attendance != null ? Math.round(r.avg_attendance).toLocaleString() : '--'}</td></tr>`).join('');
  } catch (e) { toast('Failed to load referees: ' + e.message); }
}

/* ---------- Features ---------- */
async function loadFeatures() {
  showLoading('#features-content');
  try {
    const res = await fetchJSON('/api/features');
    const feats = res.features || [];
    if (!feats.length) { $('#features-content').innerHTML = '<div class="empty">' + (res.message || 'No feature data.') + '</div>'; return; }
    $('#features-content').innerHTML = `
      <div class="section-title">Feature Importance (XGBoost)</div>
      <div class="card"><div id="feat-chart" class="chart-container" style="min-height:${Math.max(420, feats.length * 24)}px"></div></div>`;
    Plotly.newPlot('feat-chart', [{
      x: feats.map(f => f.importance), y: feats.map(f => f.name),
      type: 'bar', orientation: 'h', marker: { color: '#8b5cf6' },
      text: feats.map(f => f.importance.toFixed(4)), textposition: 'outside',
    }], plotBase({ title: { text: 'Top features by importance', font: { size: 16, color: '#e2e8f0' } }, height: Math.max(420, feats.length * 24), margin: { l: 230, r: 60, t: 40, b: 40 }, yaxis: { autorange: 'reversed', gridcolor: 'transparent' }, xaxis: { gridcolor: '#263258', title: 'Importance' } }), { responsive: true });
  } catch (e) { toast('Failed to load features: ' + e.message); }
}

/* ---------- init ---------- */
$$('.nav button').forEach(b => b.addEventListener('click', () => navTo(b.dataset.tab)));
render();
