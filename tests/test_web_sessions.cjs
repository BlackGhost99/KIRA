/* Parcours DOM : pas de clés réelles ni de réseau ; ce test ne vérifie pas le rendu visuel. */
const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let i = 0; i < 8; i++) await tick(); }
const status = { version: 'test', database: 'postgres', providers: [{name: 'gemini', configured: true, cost_class: 'free', model: 'test'}],
  budget: {used: 150, paid_used: 10, limit: 100}, routing_mode: 'QUALITY', veille: {}, devices: {}, jobs: {running: [], last: {}} };

function harness(initialAccess = false, canRefresh = true) {
  const dom = new JSDOM(html, {url: 'https://kira.test/', runScripts: 'outside-only', pretendToBeVisual: true});
  const w = dom.window, calls = [], errors = [];
  let valid = initialAccess, recent = false, effects = 0, mode = 'QUALITY';
  w.addEventListener('error', e => errors.push(e.message));
  w.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
  w.HTMLElement.prototype.scrollTo = () => {};
  w.HTMLDialogElement.prototype.showModal = function() { this.open = true; };
  w.HTMLDialogElement.prototype.close = function() { this.open = false; };
  w.localStorage.setItem('kira_token', 'obsolete-secret');
  w.fetch = async (url, opts = {}) => {
    calls.push({url, opts});
    const route = new URL(url, 'https://kira.test').pathname;
    let code = 200, data = {};
    if (route === '/api/auth/login') { valid = true; data = {name: 'Brice'}; }
    else if (route === '/api/auth/refresh') { code = canRefresh ? 200 : 401; valid = canRefresh; }
    else if (!valid) code = 401;
    else if (route === '/api/auth/me') data = {name: 'Brice'};
    else if (route === '/api/auth/logout') valid = false;
    else if (route === '/api/auth/reauth') {
      recent = JSON.parse(opts.body).password === 'correct'; code = recent ? 200 : 401;
      data = {error: 'Mot de passe incorrect.'};
    }
    else if (route === '/api/auth/sessions') data = {sessions: [{id:'mine', current:true}, {id:'other', current:false, user_agent:'Autre navigateur'}]};
    else if (route === '/api/auth/sessions/other') { code = recent ? 200 : 428; if (recent) effects++; }
    else if (route === '/api/status') data = {...status, routing_mode: mode};
    else if (route === '/api/settings/routing') { mode = JSON.parse(opts.body).mode; data = {mode}; }
    else if (route === '/api/conversations') data = {conversations: []};
    else if (route === '/api/briefing') data = {greeting: 'Bonjour', suggestions: [], veille: {pending: 0, top: []}};
    else if (route === '/api/actions/pending') data = {actions: [], paused: false};
    else if (route === '/api/devices') data = {devices: [], categories: [], never_auto: [], paused:false};
    else if (route === '/api/evolution') data = {proposals: [], jobs: {running: []}};
    else if (route === '/api/memory') data = {items: [], counts: {}, semantic: {configured:false}};
    return {status:code, ok:code >= 200 && code < 300, json: async () => data};
  };
  w.eval(script);
  return {dom, w, calls, errors, expire: () => {valid = false;}, effects: () => effects};
}

(async () => {
  let ctx = harness();
  try {
    await settle();
    assert.equal(ctx.w.document.querySelector('#app').hidden, false);
    assert.equal(ctx.calls.filter(c => c.url === '/api/auth/refresh').length, 1);
    assert.equal(ctx.w.localStorage.getItem('kira_token'), null);
    assert(ctx.calls.every(c => c.opts.credentials === 'same-origin'));
    assert(ctx.calls.every(c => !c.opts.headers?.Authorization));
    ctx.expire();
    ctx.w.document.querySelector('[data-view=more]').click();
    ctx.w.document.querySelector('[data-view=memory]').click();
    await settle();
    assert.equal(ctx.calls.filter(c => c.url === '/api/auth/refresh').length, 2, 'renouvellement unique pour les requêtes simultanées');
    ctx.w.document.querySelector('[data-view=more]').click(); await settle();
    const select = ctx.w.document.querySelector('[aria-label="Mode des fournisseurs"]');
    select.value = 'ECONOMY'; select.dispatchEvent(new ctx.w.Event('change')); await settle();
    assert(ctx.calls.some(c => c.url === '/api/settings/routing' && JSON.parse(c.opts.body).mode === 'ECONOMY'));
    let revoke = [...ctx.w.document.querySelectorAll('#settings-body .kv')].find(e => e.textContent.includes('Autre navigateur')).querySelector('button');
    revoke.click(); revoke.click(); await settle();
    const dialog = ctx.w.document.querySelector('#auth-reauth');
    assert(dialog.open);
    assert.equal(ctx.effects(), 0);
    const form = dialog.querySelector('form');
    dialog.querySelector('input').value = 'wrong'; form.dispatchEvent(new ctx.w.Event('submit', {cancelable: true})); await settle();
    assert(dialog.open); assert.match(dialog.textContent, /incorrect/); assert.equal(ctx.effects(), 0);
    dialog.querySelector('input').value = 'correct'; form.dispatchEvent(new ctx.w.Event('submit', {cancelable: true})); await settle();
    assert.equal(dialog.open, false); assert.equal(dialog.querySelector('input').value, ''); assert.equal(ctx.effects(), 1);
    const logout = [...ctx.w.document.querySelectorAll('#settings-body button')].find(b => b.textContent.includes('Se déconnecter'));
    logout.click(); await settle();
    assert(ctx.calls.some(c => c.url === '/api/auth/logout' && c.opts.method === 'POST'));
    assert.equal(ctx.w.document.querySelector('#login').hidden, false);
    assert.equal(ctx.errors.length, 0, ctx.errors.join('\n'));
  } finally { ctx.dom.window.close(); }
  ctx = harness(false, false);
  try {
    await settle();
    assert.equal(ctx.w.document.querySelector('#login').hidden, false);
    ctx.w.document.querySelector('#login-password').value = 'correct';
    ctx.w.document.querySelector('#login-form').dispatchEvent(new ctx.w.Event('submit', {cancelable: true})); await settle();
    assert.equal(ctx.w.document.querySelector('#app').hidden, false);
    assert.equal(ctx.w.document.querySelector('#login-password').value, '');
    assert.equal(ctx.w.localStorage.getItem('kira_token'), null);
    assert.equal(ctx.errors.length, 0, ctx.errors.join('\n'));
  } finally { ctx.dom.window.close(); }
  console.log('Interface sessions OK : restauration, renouvellement partagé, confirmation, révocation, connexion et déconnexion.');
})().catch(error => { console.error(error); process.exitCode = 1; });
