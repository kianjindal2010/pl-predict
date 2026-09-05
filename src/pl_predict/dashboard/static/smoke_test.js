const fs = require('fs');
const path = require('path');

const knownIds = new Set();
function makeEl(tag) {
  const el = {
    tagName: tag, innerHTML: '', _children: [], style: {}, dataset: {}, value: '',
    textContent: '', scrollIntoView() {}, appendChild(c) { this._children.push(c); return c; },
    addEventListener(type, fn) { (this._ls = this._ls || {})[type] = (this._ls[type] || []).concat([fn]); },
    get listeners() { return this._ls || {}; },
    remove() {}, querySelector() { return makeEl('div'); }, querySelectorAll() { return []; },
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, toggle(c, v) { v ? this._s.add(c) : this._s.delete(c); }, contains(c) { return this._s.has(c); } },
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._inner || ''; },
    set(v) {
      this._inner = v;
      if (typeof v === 'string') {
        const re = /id="([^"]+)"/g; let m;
        while ((m = re.exec(v))) knownIds.add(m[1]);
      }
    },
    configurable: true,
  });
  return el;
}

const toasts = [];
const bodyEl = makeEl('body');
const origAppend = bodyEl.appendChild.bind(bodyEl);
bodyEl.appendChild = (c) => {
  const cn = c && c.className ? String(c.className) : '';
  if (cn === 'toast' || cn === 'toast info') toasts.push(String(c.textContent || ''));
  return origAppend(c);
};

const NAV_TABS = ['overview', 'predict', 'season', 'performance', 'fpl', 'players', 'shots', 'teams', 'referees', 'features'];
const navButtons = NAV_TABS.map(t => { const b = makeEl('button'); b.dataset.tab = t; return b; });

global.document = {
  querySelector(sel) {
    if (typeof sel !== 'string') return makeEl('div');
    const m = /^#(.+)$/.exec(sel);
    if (m) {
      if (!knownIds.has(m[1])) return null;
      if (!elementCache.has(m[1])) elementCache.set(m[1], makeEl('div'));
      return elementCache.get(m[1]);
    }
    return makeEl('div');
  },
  querySelectorAll(sel) {
    if (sel === '.nav button') return navButtons.slice();
    return [];
  },
  body: bodyEl,
  createElement: (t) => makeEl(t),
};
const elementCache = new Map();

global.window = { addEventListener() {} };
global.location = { hash: '' };
const BASE = 'http://127.0.0.1:8000';
const realFetch = global.fetch.bind(global);
global.fetch = (url, opts) => realFetch(BASE + url, opts);
global.Plotly = { newPlot() {}, purge() {} };
global.URLSearchParams = URLSearchParams;

const containerIds = (() => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf-8');
  const re = /id="([^"]+)"/g; const ids = []; let m;
  while ((m = re.exec(html))) ids.push(m[1]);
  return ids;
})();
containerIds.forEach(id => knownIds.add(id));

const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf-8');
eval(src + '\n;globalThis.__render = render; globalThis.__TABS = TABS; globalThis.openPlayer = openPlayer;');

const missingClick = navButtons.filter(b => !(b.listeners.click && b.listeners.click.length));
if (missingClick.length) {
  console.log('FAIL nav buttons missing click listeners:', missingClick.map(b => b.dataset.tab).join(', '));
  process.exit(1);
}
console.log('OK   nav buttons have click listeners');

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function renderTab(hash, waitForId) {
  toasts.length = 0;
  knownIds.clear();
  containerIds.forEach(id => knownIds.add(id));
  location.hash = hash;
  await __render();
  if (waitForId) {
    for (let i = 0; i < 100 && !knownIds.has(waitForId); i++) await sleep(100);
  }
  await sleep(1500);   // give async loaders time to finish
  if (toasts.length) throw new Error('toasts: ' + toasts.join(' | '));
}

const tabs = ['overview', 'predict', 'season', 'performance', 'fpl', 'players', 'shots', 'teams', 'referees', 'features'];
(async () => {
  // wait for the server's startup cache to finish (WARM_DONE flag)
  for (let i = 0; i < 120; i++) {
    try {
      const r = await realFetch(BASE + '/api/health');
      const j = await r.json();
      if (j.ready) break;
    } catch (e) {}
    await sleep(500);
  }
  let failed = 0;
  for (const t of tabs) {
    try { await renderTab('#/' + t); console.log('OK   ' + t); }
    catch (e) { failed++; console.log('FAIL ' + t + ': ' + e.message); }
  }
  try { await renderTab('#/predict?home=Arsenal&away=Chelsea'); console.log('OK   predict?home=Arsenal&away=Chelsea'); }
  catch (e) { failed++; console.log('FAIL predict-params: ' + e.message); }
  try { await renderTab('#/teams/Man%20City'); console.log('OK   teams/Man City'); }
  catch (e) { failed++; console.log('FAIL teams-link: ' + e.message); }
  try {
    await renderTab('#/players', 'pl-detail');
    await openPlayer('8260', 'Erling Haaland');
    await sleep(1000);
    if (toasts.length) throw new Error('toasts: ' + toasts.join(' | '));
    console.log('OK   openPlayer(8260)');
  } catch (e) { failed++; console.log('FAIL openPlayer: ' + e.message); }
  console.log(failed ? '--- ' + failed + ' FAILURES' : '--- ALL TABS PASS');
  process.exit(failed ? 1 : 0);
})();
