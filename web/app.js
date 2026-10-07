/* Interface de KIRA. Aucune dépendance externe : marked, DOMPurify et KaTeX sont embarqués dans /vendor. */
"use strict";
(() => {
  // ================= utilitaires =================
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const SVG_NS = "http://www.w3.org/2000/svg";

  const store = {
    mem: {},
    get(k) { try { return localStorage.getItem(k); } catch (e) { return this.mem[k] ?? null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { this.mem[k] = v; } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { delete this.mem[k]; } },
  };

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
      else if (k === "value" || k === "checked") el[k] = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    const add = (kid) => {
      if (kid == null || kid === false) return;
      if (Array.isArray(kid)) kid.forEach(add);
      else el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    };
    kids.forEach(add);
    return el;
  }

  function icon(name, cls = "") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", ("ic " + cls).trim());
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  function parseDate(iso) {
    if (!iso) return null;
    const s = /([zZ]|[+-]\d\d:?\d\d)$/.test(iso) ? iso : iso + "Z";
    const d = new Date(s);
    return isNaN(d) ? null : d;
  }
  const dayFmt = new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "short" });
  function relTime(iso) {
    const d = parseDate(iso);
    if (!d) return "";
    const mins = Math.round((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return "à l'instant";
    if (mins < 60) return `il y a ${mins} min`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `il y a ${hours} h`;
    const days = Math.round(hours / 24);
    if (days === 1) return "hier";
    if (days < 7) return `il y a ${days} jours`;
    return dayFmt.format(d);
  }
  const plural = (n, one, many) => `${n} ${n > 1 ? many : one}`;

  function toast(message, bad = false) {
    const el = h("div", { class: "toast" + (bad ? " bad" : ""), role: "status" }, message);
    const box = $("#toasts");
    while (box.children.length >= 2) box.firstChild.remove();
    box.append(el);
    setTimeout(() => el.remove(), bad ? 6000 : 3600);
  }

  /** Bouton à double confirmation : un premier appui arme, un second exécute (pas de boîte de dialogue). */
  function armed(btn, label, action) {
    let timer = null;
    const original = Array.from(btn.childNodes);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (btn.dataset.armed) {
        clearTimeout(timer);
        delete btn.dataset.armed;
        action();
        return;
      }
      btn.dataset.armed = "1";
      btn.classList.add("confirm");
      btn.replaceChildren(label);
      timer = setTimeout(() => {
        delete btn.dataset.armed;
        btn.classList.remove("confirm");
        btn.replaceChildren(...original);
      }, 3000);
    });
    return btn;
  }

  // ================= état et API =================
  const state = {
    authenticated: false,
    name: "",
    view: "chat",
    convId: null,
    busy: false,
    deep: false,
    lastMsgId: 0,
    briefing: null,
    status: null,
    memKind: "",
    veilleTab: "pending",
    installEvt: null,
    bound: false,
  };

  store.del("kira_token"); // migration : aucun secret dans le stockage JavaScript

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  let refreshPromise = null;
  let reauthPromise = null;

  async function renewSession() {
    if (!refreshPromise) {
      const rotate = async () => {
        const ctl = new AbortController();
        const timeout = setTimeout(() => ctl.abort(), 90000);
        try {
        // Un autre onglet peut avoir renouvelé les cookies pendant l'attente du verrou.
        const current = await fetch("/api/auth/me", { credentials: "same-origin", signal: ctl.signal });
        if (current.ok) return true;
        const res = await fetch("/api/auth/refresh", { method: "POST", credentials: "same-origin", signal: ctl.signal, headers: { "X-KIRA-CSRF": "1" } });
        return res.ok;
        } finally { clearTimeout(timeout); }
      };
      refreshPromise = (navigator.locks ? navigator.locks.request("kira-session-refresh", rotate) : rotate())
        .finally(() => { refreshPromise = null; });
    }
    return refreshPromise;
  }

  function confirmPassword() {
    if (reauthPromise) return reauthPromise;
    reauthPromise = new Promise((resolve) => {
      const dialog = $("#auth-reauth");
      const form = $("form", dialog);
      const input = $("input", dialog);
      const error = $(".reauth-error", dialog);
      const finish = (ok) => {
        input.value = "";
        form.removeEventListener("submit", submit);
        dialog.removeEventListener("cancel", cancel);
        $(".reauth-cancel", dialog).removeEventListener("click", cancel);
        dialog.close();
        resolve(ok);
      };
      const cancel = (e) => { e.preventDefault(); finish(false); };
      const submit = async (e) => {
        e.preventDefault();
        const button = $("button[type=submit]", form);
        button.disabled = true;
        try {
          const res = await authedFetch("/api/auth/reauth", { method: "POST", body: JSON.stringify({ password: input.value }) });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { error.textContent = data.error || "Confirmation impossible."; input.value = ""; input.focus(); }
          else finish(true);
        } catch (err) { error.textContent = "Impossible de joindre KIRA."; }
        finally { button.disabled = false; }
      };
      error.textContent = "";
      form.addEventListener("submit", submit);
      dialog.addEventListener("cancel", cancel);
      $(".reauth-cancel", dialog).addEventListener("click", cancel);
      dialog.showModal();
      input.focus();
    }).finally(() => { reauthPromise = null; });
    return reauthPromise;
  }

  async function authedFetch(path, opts = {}) {
    const options = { ...opts, credentials: "same-origin", headers: { "X-KIRA-CSRF": "1", ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}), ...opts.headers } };
    let res = await fetch(path, options);
    const authAction = ["/api/auth/login", "/api/auth/refresh", "/api/auth/logout", "/api/auth/reauth"].includes(path);
    if (res.status === 401 && !authAction) {
      if (await renewSession()) res = await fetch(path, options);
      if (res.status === 401) { logout("Ta session a expiré. Reconnecte-toi."); throw new ApiError("Session expirée.", 401); }
    }
    if (res.status === 428 && !authAction) {
      if (!await confirmPassword()) throw new ApiError("Action annulée.", 428);
      res = await fetch(path, options);
    }
    return res;
  }

  async function api(path, opts = {}) {
    const { method = "GET", body, raw = false, timeout = 90000 } = opts;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeout);
    let res;
    try {
      res = await authedFetch(path, {
        method,
        signal: ctl.signal,
        headers: {
          ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError("Impossible de joindre KIRA. Vérifie ta connexion.", 0);
    } finally {
      clearTimeout(timer);
    }
    if (res.status === 401 && path !== "/api/auth/login") {
      logout("Ta session a expiré. Reconnecte-toi.");
      throw new ApiError("Session expirée.", 401);
    }
    if (raw) {
      if (!res.ok) throw new ApiError("Erreur " + res.status, res.status);
      return res;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(data.error || "Erreur " + res.status, res.status);
    return data;
  }
  const guard = (fn) => async (...args) => {
    try { return await fn(...args); } catch (e) { if (e.status !== 401) toast(e.message || "Une erreur est survenue.", true); }
  };

  // ================= Markdown + formules (sûr) =================
  const PUA = (n) => String.fromCharCode(n);
  const CODE_RE = new RegExp(PUA(0xe000) + "(\\d+)" + PUA(0xe001), "g");
  const MATH_RE = new RegExp(PUA(0xe002) + "(\\d+)" + PUA(0xe003), "g");

  function protect(src) {
    const codes = [], maths = [];
    const stash = (tex, display) => { maths.push({ tex, display }); return PUA(0xe002) + (maths.length - 1) + PUA(0xe003); };
    let t = src.replace(/(```|~~~)[\s\S]*?(?:\1|$)|`[^`\n]+`/g, (m) => { codes.push(m); return PUA(0xe000) + (codes.length - 1) + PUA(0xe001); });
    t = t
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => stash(m, true))
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => stash(m, true))
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => stash(m, false))
      .replace(/(?<![\\$\w])\$(?=[^\s$])([^$\n]*?[^\s$])\$(?![\d$\w])/g, (_, m) => stash(m, false));
    t = t.replace(CODE_RE, (_, i) => codes[+i]);
    return { text: t, maths };
  }

  let purifyReady = false;
  function setupPurify() {
    if (purifyReady) return;
    purifyReady = true;
    DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
  }

  function renderMath(tex, display) {
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", trust: false, output: "htmlAndMathml", maxExpand: 1000, maxSize: 30 });
    } catch (e) {
      return null;
    }
  }

  // Les graphiques et les captures d'écran sont privés : on les charge avec les cookies de session, puis on les montre via un lien local.
  const blobCache = new Map();
  function protectedImage(path) {
    if (!blobCache.has(path)) {
      blobCache.set(path, api(path, { raw: true }).then((r) => r.blob()).then((b) => URL.createObjectURL(b)).catch((e) => { blobCache.delete(path); throw e; }));
    }
    return blobCache.get(path);
  }
  function hydrateImages(root) {
    $$("img[data-file]", root).forEach((img) => {
      protectedImage(img.dataset.file).then((u) => { img.src = u; }).catch(() => img.replaceWith(document.createTextNode("[image indisponible]")));
    });
  }
  function clearImages() { blobCache.forEach((p) => p.then((u) => URL.revokeObjectURL(u)).catch(() => {})); blobCache.clear(); }

  function renderMarkdown(src) {
    const wrap = h("div", { class: "prose" });
    try {
      setupPurify();
      const { text, maths } = protect(String(src || ""));
      const html = marked.parse(text, { gfm: true, breaks: false, async: false });
      const frag = DOMPurify.sanitize(html, {
        RETURN_DOM_FRAGMENT: true,
        ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|\/api\/files\/)/i,
        FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select", "iframe", "object", "embed", "svg", "math"],
        FORBID_ATTR: ["style"],
      });
      // images : uniquement celles que KIRA a produites lui-même
      $$("img", frag).forEach((img) => {
        if (!/^\/api\/files\/[0-9a-f]{32}$/.test(img.getAttribute("src") || "")) {
          img.replaceWith(document.createTextNode(img.getAttribute("alt") ? `[image : ${img.getAttribute("alt")}]` : "[image]"));
        } else {
          img.dataset.file = img.getAttribute("src"); // chargée avec ta session (jamais publique), voir hydrateImages
          img.removeAttribute("src");
        }
      });
      // formules
      const walker = document.createTreeWalker(frag, NodeFilter.SHOW_TEXT);
      const targets = [];
      for (let n = walker.nextNode(); n; n = walker.nextNode()) if (n.nodeValue.includes(PUA(0xe002))) targets.push(n);
      for (const node of targets) {
        const out = document.createDocumentFragment();
        let last = 0;
        const s = node.nodeValue;
        s.replace(MATH_RE, (m, i, offset) => {
          out.append(s.slice(last, offset));
          const item = maths[+i];
          const rendered = item && renderMath(item.tex, item.display);
          if (rendered) {
            const tpl = document.createElement("template");
            tpl.innerHTML = rendered; // sortie de KaTeX (trust: false) : sûre
            out.append(tpl.content);
          } else {
            out.append((item && item.display ? "$$" : "$") + (item ? item.tex : "") + (item && item.display ? "$$" : "$"));
          }
          last = offset + m.length;
          return m;
        });
        out.append(s.slice(last));
        node.replaceWith(out);
      }
      // tableaux défilants, blocs de code copiables
      $$("table", frag).forEach((t) => { const w = h("div", { class: "table-wrap" }); t.replaceWith(w); w.append(t); });
      $$("pre", frag).forEach((pre) => {
        const box = h("div", { class: "codeblock" });
        pre.replaceWith(box);
        box.append(pre, h("button", { type: "button", class: "copy-code" }, "Copier"));
      });
      wrap.append(frag);
      hydrateImages(wrap);
    } catch (e) {
      wrap.textContent = String(src || "");
      wrap.style.whiteSpace = "pre-wrap";
    }
    return wrap;
  }

  // ================= écrans d'entrée =================
  function show(el, on) { el.hidden = !on; }

  function showLogin(message) {
    show($("#splash"), false);
    show($("#app"), false);
    show($("#login"), true);
    $("#login-error").textContent = message || "";
    $("#login-password").value = "";
    setTimeout(() => $("#login-password").focus(), 50);
  }

  function logout(message) {
    if ($("#auth-reauth").open) $("#auth-reauth").dispatchEvent(new Event("cancel", { cancelable: true }));
    clearImages();
    $("#tray").replaceChildren();
    show($("#tray"), false);
    store.del("kira_token");
    state.authenticated = false;
    state.convId = null;
    state.briefing = null;
    showLogin(message || "");
  }

  async function boot() {
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
    window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); state.installEvt = e; });
    $("#login-form").addEventListener("submit", onLogin);

    const wake = setTimeout(() => {
      $("#splash-text").textContent = "KIRA se réveille… l'hébergement gratuit peut mettre jusqu'à une minute.";
    }, 3500);
    try {
      const me = await api("/api/auth/me");
      state.authenticated = true;
      state.name = me.name;
      enterApp();
    } catch (e) {
      if (e.status === 401) return;
      $("#splash-text").textContent = e.message;
      if (!$("#splash-retry")) {
        $(".gate-card", $("#splash")).append(h("button", { id: "splash-retry", class: "btn", on: { click: () => location.reload() } }, "Réessayer"));
      }
    } finally {
      clearTimeout(wake);
    }
  }

  async function onLogin(e) {
    e.preventDefault();
    const btn = $("button[type=submit]", e.target);
    btn.disabled = true;
    $("#login-error").textContent = "";
    try {
      const res = await authedFetch("/api/auth/login", { method: "POST", body: JSON.stringify({ password: $("#login-password").value }) });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Connexion impossible.");
      state.authenticated = true;
      state.name = data.name;
      $("#login-password").value = "";
      enterApp();
    } catch (err) {
      $("#login-error").textContent = err instanceof TypeError ? "Impossible de joindre KIRA. Vérifie ta connexion." : err.message;
    } finally {
      btn.disabled = false;
    }
  }

  function enterApp() {
    show($("#splash"), false);
    show($("#login"), false);
    show($("#app"), true);
    if (!state.bound) bindUi();
    showView("chat");
    newChat();
    loadHistory();
    loadBriefing();
    refreshStatus().then(() => schedulePending(0));
  }

  // ================= navigation =================
  function showView(name) {
    state.view = name;
    for (const v of ["chat", "memory", "veille", "more"]) show($("#view-" + v), v === name);
    $$(".nav-btn").forEach((b) => { if (b.dataset.view === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    closeDrawer();
    if (name === "memory") loadMemory();
    if (name === "veille") loadVeille();
    if (name === "more") loadMore();
  }

  function openDrawer() { $("#history").classList.add("open"); show($("#scrim"), true); }
  function closeDrawer() { $("#history").classList.remove("open"); show($("#scrim"), false); }

  async function refreshStatus() {
    try {
      state.status = await api("/api/status");
      const s = state.status;
      show($("#badge-veille"), (s.veille.pending || 0) > 0);
      show($("#badge-more"), (s.proposals_draft || 0) > 0 || ((s.devices || {}).pending || 0) > 0);
    } catch (e) { /* silencieux */ }
  }

  // ================= discussion =================
  function newChat() {
    state.convId = null;
    state.lastMsgId = 0;
    $("#messages").replaceChildren();
    show($("#welcome"), true);
    setTitle("Nouvelle conversation");
    renderWelcome();
    markCurrent();
    closeDrawer();
    $("#thread").scrollTo({ top: 0, behavior: "instant" });
  }

  function setTitle(t) { $("#chat-title").textContent = t; }

  async function loadBriefing() {
    try { state.briefing = await api("/api/briefing"); } catch (e) { return; }
    if (!state.convId) renderWelcome();
  }

  function renderWelcome() {
    const b = state.briefing;
    $("#welcome-greeting").textContent = b ? b.greeting : `Bonjour ${state.name || ""}`.trim();
    const learned = $("#welcome-learned");
    const sug = $("#welcome-suggestions");
    learned.replaceChildren();
    sug.replaceChildren();
    if (!b) { show(learned, false); return; }
    const n = b.veille.pending;
    if (n > 0) {
      show(learned, true);
      learned.append(h("p", {}, n === 1 ? "J'ai lu un article qui pourrait t'intéresser." : `J'ai lu ${n} articles qui pourraient t'intéresser.`));
      for (const it of b.veille.top.slice(0, 2)) {
        learned.append(h("button", { type: "button", class: "l-item", on: { click: () => showView("veille") } },
          h("span", { class: "l-title" }, it.title), it.summary ? h("span", { class: "l-sum" }, it.summary) : null));
      }
      learned.append(h("button", { type: "button", class: "linkish", on: { click: () => showView("veille") } }, "Voir ce que j'ai appris"));
    } else {
      show(learned, false);
    }
    for (const text of b.suggestions) {
      sug.append(h("li", {}, h("button", { type: "button", on: { click: () => onSuggestion(text) } }, h("span", {}, text), icon("chevron"))));
    }
  }

  async function onSuggestion(text) {
    if (text.startsWith("On reprend : ")) {
      try {
        const list = await api("/api/conversations?limit=1");
        if (list.conversations[0]) return openConversation(list.conversations[0].id);
      } catch (e) { return; }
    }
    send(text);
  }

  async function loadHistory() {
    let list;
    try { list = (await api("/api/conversations")).conversations; } catch (e) { return; }
    const ul = $("#history-list");
    ul.replaceChildren();
    if (!list.length) { ul.append(h("li", { class: "empty-note" }, "Nos conversations apparaîtront ici.")); return; }
    for (const c of list) {
      const del = armed(h("button", { type: "button", class: "icon-btn hist-del danger", "aria-label": "Supprimer la conversation" }, icon("trash")), "Supprimer ?", guard(async () => {
        await api("/api/conversations/" + c.id, { method: "DELETE" });
        if (state.convId === c.id) newChat();
        loadHistory();
      }));
      ul.append(h("li", { class: "hist-row", "data-id": c.id },
        h("button", { type: "button", class: "hist-open", on: { click: () => openConversation(c.id) } },
          h("span", { class: "hist-title" }, c.title), h("span", { class: "hist-date" }, relTime(c.updated_at))),
        del));
    }
    markCurrent();
  }

  function markCurrent() {
    $$(".hist-row").forEach((r) => r.classList.toggle("current", r.dataset.id === state.convId));
  }

  const openConversation = guard(async (id) => {
    const data = await api("/api/conversations/" + id);
    state.convId = id;
    state.lastMsgId = 0;
    $("#messages").replaceChildren();
    show($("#welcome"), false);
    setTitle(data.conversation.title);
    for (const m of data.messages) appendMessage(m);
    markCurrent();
    showView("chat");
    $("#thread").scrollTo({ top: $("#thread").scrollHeight, behavior: "instant" });
  });

  function appendMessage(m) {
    state.lastMsgId = Math.max(state.lastMsgId, m.id || 0);
    const el = m.role === "user" ? h("div", { class: "msg user" }, m.content) : kiraMessage(m);
    $("#messages").append(el);
    return el;
  }

  const TOOL_NOTE = {
    python: "calculé avec Python",
    web_search: "cherché sur le web",
    fetch_url: "lu une page",
    remember: "retenu quelque chose de toi",
    recall: "retrouvé nos échanges",
    devices: "regardé tes machines",
    device_action: "agi sur une de tes machines",
    device_result: "relu le résultat d'une demande",
  };

  function kiraMessage(m) {
    const meta = m.meta || {};
    const el = h("article", { class: "msg kira", "data-id": m.id });
    const tools = [...new Set(meta.tools || [])].map((t) => TOOL_NOTE[t]).filter(Boolean);
    if (tools.length) el.append(h("div", { class: "notes" }, tools.map((t) => h("span", {}, "J'ai " + t))));
    el.append(renderMarkdown(m.content));

    const up = h("button", { type: "button", class: "icon-btn", "aria-label": "Utile", "aria-pressed": "false" }, icon("thumb"));
    const down = h("button", { type: "button", class: "icon-btn", "aria-label": "Pas utile", "aria-pressed": "false" }, icon("thumb", "flip"));
    const copy = h("button", { type: "button", class: "icon-btn", "aria-label": "Copier la réponse" }, icon("copy"));
    const noteBox = h("div", { class: "fb-note", hidden: true });
    const setFb = (v) => {
      up.setAttribute("aria-pressed", String(v === 1));
      down.setAttribute("aria-pressed", String(v === -1));
      up.classList.toggle("on", v === 1);
      down.classList.toggle("on", v === -1);
    };
    setFb(m.feedback || 0);
    const vote = (v) => guard(async () => {
      const current = up.classList.contains("on") ? 1 : down.classList.contains("on") ? -1 : 0;
      const next = current === v ? 0 : v;
      await api(`/api/messages/${m.id}/feedback`, { method: "POST", body: { value: next } });
      setFb(next);
      noteBox.hidden = next !== -1;
      if (next === -1) $("input", noteBox).focus();
    });
    up.addEventListener("click", vote(1));
    down.addEventListener("click", vote(-1));
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(m.content); toast("Réponse copiée."); } catch (e) { toast("Copie impossible sur cet appareil.", true); }
    });
    const noteInput = h("input", { type: "text", maxlength: "500", placeholder: "Qu'est-ce qui n'allait pas ? (facultatif)", "aria-label": "Ce qui n'allait pas" });
    const sendNote = guard(async () => {
      const v = noteInput.value.trim();
      if (!v) { noteBox.hidden = true; return; }
      await api(`/api/messages/${m.id}/feedback`, { method: "POST", body: { value: -1, note: v } });
      noteBox.hidden = true;
      toast("Merci, j'en tiens compte pour m'améliorer.");
    });
    noteBox.append(noteInput, h("button", { type: "button", class: "btn small", on: { click: sendNote } }, "Envoyer"));
    noteInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); sendNote(); } });

    const model = meta.model ? meta.model.replace(/^claude-/, "") : "";
    el.append(h("div", { class: "actions-row" }, up, down, copy, model ? h("span", { class: "meta" }, model + (meta.tier === "deep" ? " · approfondie" : "")) : null), noteBox);
    return el;
  }

  function pendingBlock(text) {
    return h("div", { class: "pending", role: "status" }, h("span", { class: "dots" }, h("i"), h("i"), h("i")), h("span", { class: "pending-text" }, text));
  }

  async function streamChat(body, onEvent) {
    let res;
    try {
      res = await authedFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError("Impossible de joindre KIRA. Vérifie ta connexion.", 0);
    }
    if (res.status === 401) { logout("Ta session a expiré. Reconnecte-toi."); throw new ApiError("Session expirée.", 401); }
    if (!res.ok) throw new ApiError((await res.json().catch(() => ({}))).error || "Erreur " + res.status, res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let ended = false;
    for (;;) {
      let chunk;
      try { chunk = await reader.read(); } catch (e) { throw new ApiError("Connexion interrompue.", 0); }
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
          if (ev.type === "end") ended = true;
          onEvent(ev);
        }
      }
    }
    if (!ended) throw new ApiError("Connexion interrompue.", 0);
  }

  async function recover(convId, afterId, block) {
    block.querySelector(".pending-text").textContent = "Connexion perdue. Je vérifie si ma réponse est arrivée…";
    for (let i = 0; i < 50; i++) {
      await sleep(4000);
      if (state.convId !== convId) return true;
      try {
        const d = await api("/api/conversations/" + convId);
        const last = d.messages[d.messages.length - 1];
        if (last && last.role === "assistant" && last.id > afterId) {
          block.replaceWith(appendMessage(last));
          return true;
        }
      } catch (e) { if (e.status === 401) return true; }
    }
    return false;
  }

  async function send(text) {
    text = (text || "").trim();
    if (!text || state.busy) return;
    state.busy = true;
    const tier = state.deep ? "deep" : "default";
    setDeep(false);
    const input = $("#input");
    input.value = "";
    autosize();
    updateSend();
    if (window.matchMedia("(pointer: coarse)").matches) input.blur(); // téléphone : le clavier se range, la réponse se lit en plein écran
    show($("#welcome"), false);
    const before = state.lastMsgId;
    $("#messages").append(h("div", { class: "msg user" }, text));
    const block = pendingBlock("KIRA réfléchit…");
    $("#messages").append(block);
    const thread = $("#thread");
    thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });

    let userSaved = before;
    let failure = null;
    let answered = false;
    try {
      await streamChat({ message: text, conversation_id: state.convId, tier }, (ev) => {
        if (ev.type === "conversation") {
          state.convId = ev.id;
          setTitle(ev.title);
        } else if (ev.type === "user_saved") {
          userSaved = ev.id;
          state.lastMsgId = Math.max(state.lastMsgId, ev.id);
        } else if (ev.type === "status") {
          block.querySelector(".pending-text").textContent = ev.text + "…";
          if (ev.tool === "device_action") schedulePending(1200); // un accord va peut-être être demandé
        } else if (ev.type === "message") {
          answered = true;
          const el = kiraMessage({ id: ev.id, content: ev.content, meta: { provider: ev.provider, model: ev.model, tools: ev.tools, tier: ev.tier }, feedback: 0 });
          state.lastMsgId = Math.max(state.lastMsgId, ev.id);
          block.replaceWith(el);
          el.scrollIntoView({ block: "start", behavior: "smooth" });
        } else if (ev.type === "error") {
          failure = ev.message;
        }
      });
    } catch (e) {
      if (e.status === 401) { state.busy = false; return; }
      if (!answered && !failure) {
        const ok = state.convId ? await recover(state.convId, userSaved, block) : false;
        answered = ok;
        if (!ok) failure = e.message;
      }
    }
    if (!answered) {
      const err = errorBlock(failure || "Je n'ai pas pu répondre.", text);
      if (block.isConnected) block.replaceWith(err); else $("#messages").append(err);
      err.scrollIntoView({ block: "end", behavior: "smooth" });
    }
    state.busy = false;
    updateSend();
    loadHistory();
    refreshStatus();
  }

  function errorBlock(message, text) {
    const el = h("div", { class: "msg error", role: "alert" }, h("p", {}, message),
      h("div", {}, h("button", { type: "button", class: "btn small", on: { click: () => { el.remove(); send(text); } } }, "Réessayer")));
    return el;
  }

  function setDeep(on) {
    state.deep = on;
    $("#deep").setAttribute("aria-pressed", String(on));
  }

  function autosize() {
    const t = $("#input");
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 168) + "px";
  }
  function updateSend() { $("#send").disabled = state.busy || !$("#input").value.trim(); }

  const seqs = {};
  const fresh = (key) => { const n = (seqs[key] = (seqs[key] || 0) + 1); return () => seqs[key] === n; };

  // ================= mémoire =================
  const KIND_LABEL = { profile: "Sur toi", fact: "Fait", knowledge: "Appris", note: "Note", identity: "Identité de KIRA", episodic: "Événement", procedural: "Méthode", project: "Projet", goal: "Objectif", experience: "Expérience", relationship: "Relation" };
  const KIND_TABS = [["", "Tout"], ...Object.entries(KIND_LABEL)];
  $("#memory-status").addEventListener("change", () => loadMemory());

  const loadMemory = guard(async () => {
    const q = $("#memory-q").value.trim();
    const params = new URLSearchParams();
    if (state.memKind) params.set("kind", state.memKind);
    if (q) params.set("q", q);
    params.set("status", $("#memory-status").value);
    const isFresh = fresh("memory");
    const data = await api("/api/memory?" + params);
    if (!isFresh()) return;
    const semantic = data.semantic || {};
    $("#memory-search-status").textContent = semantic.configured && semantic.pgvector
      ? `Recherche par sens configurée · ${semantic.indexed || 0} souvenirs indexés · mots en secours`
      : "Recherche par mots · la recherche par sens attend son activation cloud";
    const total = Object.values(data.counts).reduce((a, b) => a + b, 0);
    $("#memory-kinds").replaceChildren(...KIND_TABS.map(([k, label]) => {
      const n = k ? data.counts[k] || 0 : total;
      return h("button", { type: "button", role: "tab", "aria-selected": String(state.memKind === k), on: { click: () => { state.memKind = k; loadMemory(); } } }, label, h("span", { class: "n" }, String(n)));
    }));
    $("#memory-list").replaceChildren(...(data.items.length
      ? data.items.map(memoryItem)
      : [h("p", { class: "empty" }, q ? "Rien ne correspond à cette recherche." : "Je ne sais encore presque rien de toi. Parle-moi de ton niveau, de tes objectifs, ou ajoute un souvenir toi-même.")]));
  });

  function memoryItem(it) {
    const card = h("article", { class: "item", "data-id": it.id });
    const view = () => {
      const tools = h("div", { class: "tools tight" },
        h("button", { type: "button", class: "btn ghost small", on: { click: guard(async () => {
          const data = await api(`/api/memory/${it.id}/history`);
          const versions = h("div", { class: "memory-history" }, data.items.map(v => h("div", { class: "item" },
            h("div", { class: "meta" }, `Version ${v.version} · ${relTime(v.updated_at)} · ${v.status}`), h("p", {}, v.content))));
          card.replaceChildren(versions, h("button", { type: "button", class: "btn small", on: { click: view } }, "Retour"));
        }) } }, "Historique"),
        it.status !== "superseded" ? h("button", { type: "button", class: "btn ghost small", on: { click: guard(async () => {
          Object.assign(it, await api(`/api/memory/${it.id}`, { method: "PATCH", body: { status: it.status === "archived" ? "active" : "archived" } }));
          await loadMemory();
        }) } }, it.status === "archived" ? "Réactiver" : "Archiver") : null,
        h("span", { class: "spacer" }),
        h("button", { type: "button", class: "icon-btn", "aria-label": "Modifier", on: { click: edit } }, icon("edit")),
        armed(h("button", { type: "button", class: "icon-btn danger", "aria-label": "Effacer" }, icon("trash")), "Effacer ?", guard(async () => {
          await api("/api/memory/" + it.id, { method: "DELETE" });
          card.classList.add("leaving");
          setTimeout(() => { card.remove(); loadMemory(); refreshStatus(); }, 220);
        })));
      const meta = h("div", { class: "meta" }, h("span", { class: "kind" }, KIND_LABEL[it.kind] || it.kind),
        h("span", {}, `v${it.version || 1}`),
        it.superseded_by ? h("span", {}, `Remplacé par le souvenir #${it.superseded_by}`) : null,
        it.tags ? h("span", {}, it.tags) : null, h("span", {}, relTime(it.created_at)),
        /^https?:\/\//.test(it.source || "") ? h("a", { href: it.source, target: "_blank", rel: "noopener noreferrer" }, "source") : (it.source ? h("span", {}, it.source) : null));
      card.replaceChildren(meta, h("div", { class: "body" }, it.content), tools);
    };
    const edit = () => {
      const ta = h("textarea", { rows: "4", maxlength: "4000", "aria-label": "Contenu du souvenir" });
      ta.value = it.content;
      const kind = h("select", { "aria-label": "Type" }, Object.entries(KIND_LABEL).map(([k, v]) => h("option", { value: k }, v)));
      kind.value = it.kind;
      const save = h("button", { type: "button", class: "btn primary small", on: { click: guard(async () => {
        const v = ta.value.trim();
        if (!v) { toast("Un souvenir ne peut pas être vide.", true); return; }
        const updated = await api("/api/memory/" + it.id, { method: "PATCH", body: { content: v, kind: kind.value } });
        Object.assign(it, updated);
        view();
        toast("C'est corrigé.");
      }) } }, "Enregistrer");
      const replace = it.status === "active" ? h("button", { type: "button", class: "btn small", on: { click: guard(async () => {
        const v = ta.value.trim();
        if (!v) { toast("Un souvenir ne peut pas être vide.", true); return; }
        await api(`/api/memory/${it.id}/supersede`, { method: "POST", body: { content: v, kind: kind.value } });
        await loadMemory();
        toast("Nouvelle information retenue. L'ancienne reste dans l'historique.");
      }) } }, "Remplacer l'information") : null;
      card.replaceChildren(ta, h("div", { class: "row" }, h("label", { class: "field grow" }, h("span", {}, "Type"), kind),
        h("div", { class: "actions" }, h("button", { type: "button", class: "btn ghost small", on: { click: view } }, "Annuler"), replace, save)));
      ta.focus();
    };
    view();
    return card;
  }

  // ================= veille =================
  const SOURCE_TRUST = [["0.9", "Très fiable"], ["0.6", "Correcte"], ["0.3", "À vérifier"]];
  const trustLabel = (s) => (s >= 0.75 ? "très fiable" : s >= 0.45 ? "fiabilité correcte" : "à vérifier");

  const loadVeille = guard(async () => {
    const tab = state.veilleTab;
    const isFresh = fresh("veille");
    const counts = (await api("/api/veille/items?status=pending&limit=1")).counts;
    if (!isFresh()) return;
    $("#veille-tabs").replaceChildren(
      ...[["pending", "À relire", counts.pending || 0], ["validated", "Gardés", counts.validated || 0], ["sources", "Sources", null]].map(([k, label, n]) =>
        h("button", { type: "button", role: "tab", "aria-selected": String(tab === k), on: { click: () => { state.veilleTab = k; loadVeille(); } } }, label, n === null ? null : h("span", { class: "n" }, String(n)))));
    show($("#veille-all"), tab === "pending" && (counts.pending || 0) > 0);
    $("#veille-sub").textContent = (counts.pending || 0)
      ? `${plural(counts.pending, "chose attend", "choses attendent")} ton avis. Ce que tu gardes entre dans ma mémoire.`
      : "Je lis des sources fiables pour toi. Tu décides de ce qui entre dans ma mémoire.";
    let nodes;
    if (tab === "sources") {
      nodes = await sourceNodes();
    } else {
      const data = await api(`/api/veille/items?status=${tab}&limit=80`);
      if (data.items.length) {
        nodes = data.items.map((it) => veilleItem(it, tab === "pending"));
      } else {
        const empty = h("p", { class: "empty" }, tab === "pending"
          ? "Rien à relire pour l'instant. Je continue de surveiller mes sources."
          : "Tu n'as encore rien gardé. Ce que tu valides apparaîtra ici.");
        if (tab === "pending") empty.append(h("br"), h("button", { type: "button", class: "btn", on: { click: runVeille } }, "Lancer une veille maintenant"));
        nodes = [empty];
      }
    }
    if (!isFresh()) return;
    $("#veille-body").replaceChildren(...nodes);
  });

  function veilleItem(it, actionable) {
    const card = h("article", { class: "item" });
    const done = (path, okMsg) => guard(async () => {
      await api(`/api/veille/items/${it.id}/${path}`, { method: "POST" });
      card.classList.add("leaving");
      toast(okMsg);
      setTimeout(() => { card.remove(); loadVeille(); refreshStatus(); loadBriefing(); }, 230);
    });
    card.append(
      h("div", { class: "meta" }, it.topic ? h("span", { class: "kind" }, it.topic) : null,
        it.source_name ? h("span", {}, it.source_name) : null, h("span", { class: "trust" }, trustLabel(it.score || 0))),
      h("h3", {}, /^https?:\/\//.test(it.url || "") ? h("a", { href: it.url, target: "_blank", rel: "noopener noreferrer" }, it.title) : it.title),
      h("div", { class: "body" }, it.summary || ""));
    if (actionable) {
      card.append(h("div", { class: "tools" },
        h("button", { type: "button", class: "btn primary small", on: { click: done("validate", "Gardé : je m'en souviendrai.") } }, icon("check"), "Garder"),
        h("button", { type: "button", class: "btn ghost small", on: { click: done("reject", "Écarté.") } }, "Écarter")));
    }
    return card;
  }

  async function sourceNodes() {
    const nodes = [];
    const form = h("form", { class: "card form" },
      h("label", { class: "field" }, h("span", {}, "Adresse d'une source (flux RSS ou page)"), h("input", { type: "url", name: "url", placeholder: "https://…", required: true })),
      h("div", { class: "row" },
        h("label", { class: "field grow" }, h("span", {}, "Nom (facultatif)"), h("input", { type: "text", name: "name", maxlength: "120" })),
        h("label", { class: "field" }, h("span", {}, "Fiabilité"), h("select", { name: "score" }, SOURCE_TRUST.map(([v, l]) => h("option", { value: v }, l)))),
        h("button", { class: "btn primary", type: "submit" }, "Ajouter")));
    form.addEventListener("submit", guard(async (e) => {
      e.preventDefault();
      const f = new FormData(form);
      await api("/api/veille/sources", { method: "POST", body: { url: f.get("url"), name: f.get("name"), score: Number(f.get("score")), kind: "rss" } });
      toast("Source ajoutée.");
      loadVeille();
    }));
    nodes.push(form);
    const data = await api("/api/veille/sources");
    for (const s of data.sources) {
      const sw = h("button", { type: "button", class: "switch", role: "switch", "aria-checked": String(!!s.active), "aria-label": "Surveiller cette source" });
      sw.addEventListener("click", guard(async () => {
        const next = sw.getAttribute("aria-checked") !== "true";
        await api("/api/veille/sources/" + s.id, { method: "PATCH", body: { active: next ? 1 : 0 } });
        sw.setAttribute("aria-checked", String(next));
      }));
      const trust = h("select", { "aria-label": "Fiabilité", class: "trust-select" }, SOURCE_TRUST.map(([v, l]) => h("option", { value: v }, l)));
      trust.value = s.score >= 0.75 ? "0.9" : s.score >= 0.45 ? "0.6" : "0.3";
      trust.style.width = "auto";
      trust.style.minHeight = "34px";
      trust.addEventListener("change", guard(async () => {
        await api("/api/veille/sources/" + s.id, { method: "PATCH", body: { score: Number(trust.value) } });
        toast("Fiabilité mise à jour.");
      }));
      const del = armed(h("button", { type: "button", class: "btn ghost small danger" }, icon("trash"), "Retirer"), "Retirer ?", guard(async () => {
        await api("/api/veille/sources/" + s.id, { method: "DELETE" });
        loadVeille();
      }));
      nodes.push(h("article", { class: "item src-row" },
        h("div", {}, h("h3", {}, s.name || s.url), h("div", { class: "src-url" }, s.url),
          s.last_error ? h("div", { class: "src-err" }, "Dernière lecture en échec : " + s.last_error) : (s.last_run ? h("div", { class: "src-url" }, "Lue " + relTime(s.last_run)) : null)),
        sw, h("div", { class: "tools" }, trust, h("span", { class: "spacer" }), del)));
    }
    return nodes;
  }

  async function waitJob(name, onDone) {
    for (let i = 0; i < 100; i++) {
      await sleep(3000);
      let st;
      try { st = await api("/api/status"); } catch (e) { if (e.status === 401) return; continue; }
      if (!st.jobs.running[name]) { state.status = st; onDone(st.jobs.last[name], st); return; }
    }
    onDone(null, null);
  }

  const runVeille = guard(async () => {
    const btn = $("#veille-run");
    btn.disabled = true;
    const label = $("span", btn);
    label.textContent = "Lecture en cours…";
    const ic = $("svg", btn);
    ic.classList.add("spin");
    const res = await api("/api/veille/run", { method: "POST" });
    if (!res.started) toast(res.message);
    await waitJob("veille", (last) => {
      if (last && last.ok === false) toast("La veille a rencontré un problème : " + (last.error || "inconnu"), true);
      else if (last && last.result) toast(last.result.new ? `Veille terminée : ${plural(last.result.new, "nouveauté", "nouveautés")}.` : "Veille terminée : rien de nouveau.");
    });
    btn.disabled = false;
    label.textContent = "Lancer une veille";
    ic.classList.remove("spin");
    refreshStatus();
    loadBriefing();
    if (state.view === "veille") loadVeille();
  });

  // ================= plus : évolution et réglages =================
  const loadMore = guard(async () => {
    await refreshStatus();
    renderSettings();
    loadDevices();
    loadEvolution();
  });

  const STATUS_PILL = (p) => {
    if (p.status === "pr_opened") return h("span", { class: "pill ok" }, "Envoyée sur GitHub");
    if (p.status === "rejected") return h("span", { class: "pill" }, "Écartée");
    if (p.tests_ok === false) return h("span", { class: "pill bad" }, "Ne passe pas mes vérifications");
    return h("span", { class: "pill spark" }, "À valider");
  };

  function diffView(text) {
    const pre = h("pre", {});
    for (const line of (text || "").split("\n")) {
      const cls = line.startsWith("+++") || line.startsWith("---") ? "" : line.startsWith("+") ? "diff-add" : line.startsWith("-") ? "diff-del" : line.startsWith("@@") ? "diff-hunk" : "";
      pre.append(h("span", { class: cls }, line + "\n"));
    }
    return pre;
  }

  const loadEvolution = guard(async () => {
    const data = await api("/api/evolution");
    const running = !!data.jobs.running.evolution;
    $("#evo-go").disabled = running;
    const stateEl = $("#evo-state");
    const last = data.jobs.last.evolution;
    if (running) { stateEl.hidden = false; stateEl.textContent = "J'analyse mon code et ce que je sais de toi… (1 à 3 minutes)"; stateEl.className = "hint"; watchEvolution(); }
    else if (last && last.ok === false) { stateEl.hidden = false; stateEl.textContent = "Mon analyse n'a pas abouti : " + last.error; stateEl.className = "hint warn"; }
    else if (last && last.ok && last.result && last.result.message) { stateEl.hidden = false; stateEl.textContent = last.result.message; stateEl.className = "hint"; }
    else stateEl.hidden = true;
    const list = $("#evo-list");
    list.replaceChildren();
    if (!data.proposals.length) {
      list.append(h("p", { class: "empty" }, "Aucune idée pour l'instant. Dis-moi ce qui te gêne dans nos échanges, ou lance une analyse."));
      return;
    }
    const githubOk = !!(state.status && state.status.github);
    for (const p of data.proposals) {
      const card = h("article", { class: "item proposal" }, STATUS_PILL(p), h("h3", {}, p.title), h("div", { class: "body" }, p.rationale));
      if (p.status === "pr_opened" && p.pr_url) {
        card.append(h("p", { class: "hint" }, "Elle ne sera appliquée que si tu la fusionnes toi-même sur GitHub."),
          h("div", { class: "tools" }, h("a", { class: "btn small", href: p.pr_url, target: "_blank", rel: "noopener noreferrer" }, icon("external"), "Voir sur GitHub")));
      }
      if (p.error && p.status === "draft") card.append(h("p", { class: "hint warn" }, p.error));
      if (p.status === "draft") {
        if (p.tests_ok !== false && !githubOk) card.append(h("p", { class: "hint warn" }, "GitHub n'est pas connecté : ajoute GITHUB_TOKEN dans les réglages du serveur pour pouvoir accepter."));
        const tools = h("div", { class: "tools" });
        if (p.tests_ok !== false) {
          tools.append(h("button", { type: "button", class: "btn primary small", disabled: !githubOk, on: { click: guard(async (e) => {
            const btn = e.currentTarget; btn.disabled = true; btn.textContent = "Envoi…";
            try { await api(`/api/evolution/${p.id}/approve`, { method: "POST", timeout: 120000 }); toast("Envoyée sur GitHub : à toi de la relire."); }
            finally { loadEvolution(); refreshStatus(); }
          }) } }, icon("check"), "Accepter"));
        }
        tools.append(h("button", { type: "button", class: "btn ghost small", on: { click: guard(async () => { await api(`/api/evolution/${p.id}/reject`, { method: "POST" }); loadEvolution(); refreshStatus(); }) } }, "Écarter"));
        card.append(tools);
      }
      if (p.diff) {
        const checks = p.tests_ok === true ? "Mes vérifications automatiques passent." : p.tests_ok === false ? "Mes vérifications automatiques échouent." : "";
        card.append(h("details", { class: "tech" }, h("summary", {}, "Détails techniques"),
          h("p", { class: "hint" }, `Fichiers touchés : ${(p.files || []).join(", ") || "—"}. ${checks}`), diffView(p.diff),
          p.tests_ok === false && p.tests_output ? h("pre", {}, p.tests_output.slice(-4000)) : null));
      }
      list.append(card);
    }
  });

  let evoWatching = false;
  async function watchEvolution() {
    if (evoWatching) return;
    evoWatching = true;
    await waitJob("evolution", () => {});
    evoWatching = false;
    refreshStatus();
    if (state.view === "more") loadEvolution();
  }

  const proposeEvolution = guard(async (e) => {
    e.preventDefault();
    const goal = $("#evo-goal").value.trim();
    const res = await api("/api/evolution/propose", { method: "POST", body: goal ? { goal } : {} });
    toast(res.message);
    $("#evo-goal").value = "";
    loadEvolution();
  });

  function kv(label, value) {
    return h("div", { class: "kv" }, h("span", { class: "k" }, label), h("span", { class: "v" }, value));
  }

  function renderSettings() {
    const s = state.status;
    const root = $("#settings-body");
    root.replaceChildren();
    if (!s) { root.append(h("p", { class: "hint" }, "Impossible de lire l'état du serveur.")); return; }

    root.append(h("h3", { class: "sub-h" }, "Les IA que j'utilise"),
      h("p", { class: "hint" }, "Si l'un tombe en panne, je passe automatiquement au suivant."));
    for (const p of s.providers) {
      const led = h("i", { class: "led " + (p.configured ? (p.cooldown_s ? "warn" : "ok") : "") });
      root.append(kv(p.name, [led, p.configured ? (p.disabled ? "configuration à corriger" : p.cooldown_s ? `en pause (${p.cooldown_s} s)` : `${p.model} · ${p.cost_class === "free" ? "gratuit" : p.cost_class === "paid" ? "payant" : "hébergé par toi"}`) : "non configuré"]));
    }
    const modeSelect = h("select", { "aria-label": "Mode des fournisseurs", on: { change: guard(async (e) => {
      await api("/api/settings/routing", { method: "POST", body: { mode: e.target.value } });
      await refreshStatus();
    }) } }, [["QUALITY", "Qualité"], ["ECONOMY", "Économie"], ["PRIVATE", "Fournisseurs autorisés"]].map(([value, label]) => h("option", { value }, label)));
    modeSelect.value = s.routing_mode;
    root.append(kv("Mode", modeSelect), h("p", { class: "hint" }, "Économie privilégie les comptes gratuits. Fournisseurs autorisés limite les réponses aux services que tu as approuvés dans la configuration."));
    const pct = Math.min(100, Math.round((s.budget.paid_used / Math.max(1, s.budget.limit)) * 100));
    root.append(h("h3", { class: "sub-h" }, "Aujourd'hui"),
      h("div", { class: "kv" }, h("span", { class: "k" }, "Budget payant utilisé"), h("span", { class: "v" }, `${s.budget.paid_used.toLocaleString("fr-FR")} / ${s.budget.limit.toLocaleString("fr-FR")} jetons`)),
      h("div", { class: "meter" + (pct > 80 ? " high" : ""), role: "progressbar", "aria-valuenow": String(pct), "aria-valuemin": "0", "aria-valuemax": "100" }, h("i", { style: `width:${pct}%` })));

    root.append(h("h3", { class: "sub-h" }, "Mon installation"),
      kv("Recherche web", s.web_search === "tavily" ? "Tavily" : "DuckDuckGo"),
      kv("Mémoire", s.database === "postgres" ? "Base durable (PostgreSQL)" : "Fichier local (SQLite)"),
      kv("GitHub", s.github ? "Connecté" : "Non connecté"),
      kv("Version", s.version));
    if (s.database !== "postgres") {
      root.append(h("p", { class: "hint warn", style: "margin-top:10px" }, "Sur un hébergement gratuit, un fichier local s'efface à chaque redémarrage. Configure DATABASE_URL (Supabase) pour que je n'oublie rien."));
    }

    root.append(h("h3", { class: "sub-h" }, "Toi et tes données"));
    const actions = h("div", { class: "stack" });
    actions.append(h("button", { type: "button", class: "btn", on: { click: exportData } }, icon("download"), "Exporter mes données"));
    if (!window.matchMedia("(display-mode: standalone)").matches) {
      actions.append(h("button", { type: "button", class: "btn", on: { click: installApp } }, icon("download"), "Installer sur l'écran d'accueil"));
    }
    actions.append(h("button", { type: "button", class: "btn ghost", on: { click: guard(async () => { await api("/api/auth/logout", { method: "POST" }); logout(""); }) } }, icon("logout"), "Se déconnecter"));
    root.append(actions);
    const sessionsRoot = h("div", {}, h("p", { class: "hint" }, "Chargement des sessions…"));
    root.append(h("h3", { class: "sub-h" }, "Sessions connectées"), sessionsRoot);
    api("/api/auth/sessions").then((data) => {
      sessionsRoot.replaceChildren(...data.sessions.map((session) => {
        const button = h("button", { type: "button", class: "btn ghost small" }, "Révoquer");
        armed(button, "Confirmer", guard(async () => {
          await api("/api/auth/sessions/" + encodeURIComponent(session.id), { method: "DELETE" });
          if (session.current) logout(""); else renderSettings();
        }));
        return kv(session.current ? "Cet appareil" : session.user_agent || "Autre appareil", button);
      }));
    }).catch(() => { sessionsRoot.replaceChildren(h("p", { class: "hint" }, "Sessions indisponibles.")); });

    const journal = h("div", {});
    const journalBtn = h("button", { type: "button", class: "btn ghost small", style: "margin-top:14px", on: { click: guard(async () => {
      if (journal.childElementCount) { journal.replaceChildren(); journalBtn.textContent = "Voir mon journal d'activité"; return; }
      const d = await api("/api/audit?limit=60");
      journalBtn.textContent = "Masquer le journal";
      journal.append(h("p", { class: "hint" }, "Tout ce que je fais est inscrit ici."), h("div", { class: "audit" }, d.entries.map((en) =>
        h("div", {}, h("b", {}, en.action), " · ", en.actor, " · ", relTime(en.ts), Object.keys(en.details || {}).length ? " · " + JSON.stringify(en.details).slice(0, 140) : ""))));
    }) } }, "Voir mon journal d'activité");
    root.append(journalBtn, journal);
  }

  const exportData = guard(async () => {
    const res = await api("/api/export", { raw: true });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: "kira-export.json" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  });

  async function installApp() {
    if (state.installEvt) {
      state.installEvt.prompt();
      await state.installEvt.userChoice.catch(() => {});
      state.installEvt = null;
    } else {
      toast("Dans le menu de ton navigateur, choisis « Installer l'application » ou « Ajouter à l'écran d'accueil ».");
    }
  }

  // ================= le cœur : machines et accords =================
  const PLATFORM = { windows: "Windows", linux: "Linux", mac: "Mac", android: "Android" };
  const MODE_LABEL = { auto: "Libre", ask: "Je valide", deny: "Interdit" };
  const ACT_LABEL = { done: "fait", failed: "échec", denied: "refusé", expired: "expiré", cancelled: "annulé", pending: "attend ton accord", queued: "envoyé à la machine", running: "en cours" };
  const ACT_PILL = { done: "ok", failed: "bad", denied: "bad", expired: "", cancelled: "", pending: "spark", queued: "spark", running: "spark" };
  const SENSITIVE = ["exec", "screen", "input"];
  const SHORT = { monitor: "État de la machine", input: "Souris et clavier" };
  let devData = null;

  function thumb(path) {
    const img = h("img", { class: "shot", alt: "Capture d'écran", title: "Ouvrir en grand" });
    protectedImage(path).then((u) => { img.src = u; img.addEventListener("click", () => window.open(u, "_blank", "noopener")); }).catch(() => img.replaceWith(h("span", { class: "hint" }, "[capture indisponible]")));
    return img;
  }

  // ---- plateau d'accords : visible sur tous les écrans, avec la commande exacte
  let pendingTimer = null;
  function schedulePending(delay) {
    clearTimeout(pendingTimer);
    pendingTimer = setTimeout(pollPending, delay == null ? (state.busy ? 2000 : 6000) : delay);
  }

  async function pollPending() {
    if (!state.authenticated) return;
    const d = state.status && state.status.devices;
    if (!document.hidden && d && d.total > 0) {
      try {
        const res = await api("/api/actions/pending", { timeout: 15000 });
        renderTray(res.actions);
        if (state.busy && res.actions.length) {
          const t = $(".pending-text");
          if (t) t.textContent = "J'attends ton accord pour agir sur ta machine…";
        }
      } catch (e) { /* silencieux : on réessaiera */ }
    }
    if (state.authenticated) schedulePending();
  }

  function renderTray(actions) {
    const box = $("#tray");
    const key = actions.map((a) => a.id).join(",");
    show(box, actions.length > 0);
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.replaceChildren(...actions.slice(0, 2).map(trayCard));
    // le plateau ne doit jamais cacher la fin de la conversation
    document.documentElement.style.setProperty("--tray-h", actions.length ? "230px" : "0px");
    if (actions.length) { const pend = $(".pending"); if (pend) pend.scrollIntoView({ block: "center", behavior: "smooth" }); }
    if (actions.length > 2) box.append(h("button", { type: "button", class: "btn small ghost tray-more", on: { click: () => showView("more") } }, `+ ${plural(actions.length - 2, "autre demande", "autres demandes")}`));
    show($("#badge-more"), actions.length > 0 || (state.status && state.status.proposals_draft > 0));
  }

  function trayCard(a) {
    const buttons = [];
    let card = null;
    const decide = (verb) => async () => {
      buttons.forEach((b) => (b.disabled = true));
      try {
        await api(`/api/actions/${a.id}/${verb}`, { method: "POST" });
        toast(verb === "approve" ? "Autorisé. J'envoie à la machine." : "Refusé. Rien n'est fait.");
        card.remove(); // tout de suite : pas d'attente du prochain relevé
        if (!$("#tray .tray-card")) { show($("#tray"), false); $("#tray").dataset.key = ""; document.documentElement.style.setProperty("--tray-h", "0px"); }
      } catch (e) {
        if (e.status !== 401) toast(e.message, true);
      }
      pollPending();
      refreshStatus();
    };
    const no = h("button", { type: "button", class: "btn ghost", on: { click: decide("deny") } }, "Refuser");
    const yes = h("button", { type: "button", class: "btn primary", on: { click: decide("approve") } }, icon("check"), "Autoriser");
    buttons.push(no, yes);
    card = h("article", { class: "tray-card", role: "alertdialog", "aria-label": "KIRA demande ton accord" },
      h("div", { class: "who" }, h("span", { class: "pill spark" }, "Demande d'accord"), h("span", {}, a.device_name), a.risk === "high" ? h("span", { class: "pill bad" }, "Sensible") : null),
      h("h3", {}, a.summary),
      a.detail ? h("pre", {}, a.detail) : null,
      a.reason ? h("p", { class: "why" }, a.reason) : null,
      h("div", { class: "btns" }, no, yes));
    return card;
  }

  // ---- écran « Mes machines »
  const loadDevices = guard(async () => {
    const ok = fresh("devices");
    const data = await api("/api/devices");
    if (!ok()) return;
    devData = data;
    renderDevices();
  });

  function renderDevices() {
    const data = devData;
    const list = $("#dev-list");
    const nodes = [];
    show($("#dev-global"), data.devices.length > 0);
    const sw = $("#dev-pause-all");
    sw.setAttribute("aria-checked", String(!!data.paused));
    if (!data.devices.length) {
      nodes.push(h("p", { class: "empty" }, "Aucune machine reliée. Ajoute ton PC ou ton téléphone : je pourrai y lire des fichiers, lancer des programmes et même voir l'écran, toujours selon tes règles."));
    } else {
      nodes.push(h("p", { class: "legend" }, h("span", {}, h("b", { class: "c-auto" }, "Libre"), " : je le fais seule"), h("span", {}, h("b", { class: "c-ask" }, "Je valide"), " : tu autorises chaque fois"), h("span", {}, h("b", { class: "c-deny" }, "Interdit"), " : jamais")));
      data.devices.forEach((d) => nodes.push(deviceCard(d)));
    }
    list.replaceChildren(...nodes);
  }

  function policyRow(d, cat) {
    const label = SHORT[cat] || devData.categories[cat];
    const enabled = d.enabled.includes(cat);
    const row = h("div", { class: "dev-row" + (enabled ? "" : " is-off") }, h("span", { class: "lab" }, label));
    if (!enabled) {
      row.append(h("span", { class: "pill" }, "Coupé sur la machine"), h("span", { class: "off" }, `Pour l'activer, sur la machine : python kira_core.py allow ${cat}`));
      return row;
    }
    const never = devData.never_auto.includes(cat);
    const group = h("div", { class: "modes", role: "radiogroup", "aria-label": label });
    const sync = () => $$("button", group).forEach((b) => b.setAttribute("aria-checked", String(d.policy[cat] === b.dataset.mode)));
    for (const mode of ["auto", "ask", "deny"]) {
      const b = h("button", { type: "button", role: "radio", "data-mode": mode, "aria-checked": String(d.policy[cat] === mode), disabled: mode === "auto" && never, title: mode === "auto" && never ? "Supprimer demande toujours ton accord" : null }, MODE_LABEL[mode]);
      b.addEventListener("click", guard(async () => {
        if (d.policy[cat] === mode) return;
        const before = d.policy[cat];
        d.policy[cat] = mode;
        sync();
        try {
          const out = await api(`/api/devices/${d.id}`, { method: "PATCH", body: { policy: { [cat]: mode } } });
          Object.assign(d, out);
        } catch (e) { d.policy[cat] = before; sync(); throw e; }
        if (mode === "auto" && SENSITIVE.includes(cat)) toast("« Libre » : j'agis sans te demander. Réserve-le à une machine de confiance. Si j'ai lu Internet pendant la conversation, je te redemande quand même.");
      }));
      group.append(b);
    }
    row.append(group);
    const missing = (cat === "screen" && d.available.screenshot === false) ? "Outil de capture manquant : pip install mss pillow (ou scrot / grim sous Linux)."
      : (cat === "input" && d.available.input === false) ? "Outil de pilotage manquant : pip install pyautogui (ou xdotool sous Linux)." : "";
    if (missing) row.append(h("span", { class: "off" }, missing));
    return row;
  }

  function deviceCard(d) {
    const card = h("article", { class: "item dev" });
    const title = h("h3", {}, d.name);
    const rename = h("button", { type: "button", class: "icon-btn", "aria-label": "Renommer cette machine" }, icon("edit"));
    rename.addEventListener("click", () => {
      const input = h("input", { type: "text", class: "name-input", maxlength: "60", value: d.name, "aria-label": "Nom de la machine" });
      title.replaceWith(input);
      input.focus(); input.select();
      let done = false;
      const finish = async (save) => {
        if (done) return;
        done = true;
        const v = input.value.trim();
        input.replaceWith(title);
        if (save && v && v !== d.name) {
          try { const out = await api(`/api/devices/${d.id}`, { method: "PATCH", body: { name: v } }); title.textContent = out.name; d.name = out.name; } catch (e) { if (e.status !== 401) toast(e.message, true); }
        }
      };
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); });
      input.addEventListener("blur", () => finish(true));
    });
    const stateTxt = d.paused ? "en pause" : d.local_pause ? "en pause sur la machine" : d.online ? "en ligne" : `hors ligne${d.last_seen ? " · vue " + relTime(d.last_seen) : ""}`;
    const led = h("i", { class: "led " + (d.paused || d.local_pause ? "warn" : d.online ? "ok" : "") });
    card.append(h("div", { class: "dev-head" }, title, rename, h("span", { class: "pill" }, PLATFORM[d.platform] || d.platform || "?"), h("span", { class: "dev-state" }, led, stateTxt)));

    const grid = h("div", { class: "dev-grid" });
    Object.keys(devData.categories).forEach((cat) => grid.append(policyRow(d, cat)));
    card.append(grid);

    const facts = [];
    if (d.roots.length) facts.push("Dossiers où j'ai le droit d'agir : " + d.roots.join(", ") + ".");
    else facts.push("Aucun dossier autorisé sur la machine : je ne peux pas toucher aux fichiers.");
    if (d.info && d.info.hostname) facts.push(`${d.info.os || ""} · ${d.info.hostname}`.trim() + (d.agent_version ? ` · cœur ${d.agent_version}` : ""));
    facts.push("Mes mots de passe, clés et fichiers .env restent toujours inaccessibles, même dans ces dossiers.");
    card.append(h("p", { class: "dev-facts" }, facts.join(" ")));

    const activity = h("div", { class: "acts", hidden: true });
    const pauseBtn = h("button", { type: "button", class: "btn small" }, icon(d.paused ? "play" : "pause"), d.paused ? "Reprendre" : "Mettre en pause");
    pauseBtn.addEventListener("click", guard(async () => { await api(`/api/devices/${d.id}`, { method: "PATCH", body: { paused: !d.paused } }); loadDevices(); refreshStatus(); }));
    const actBtn = h("button", { type: "button", class: "btn small ghost" }, icon("activity"), "Activité");
    actBtn.addEventListener("click", guard(async () => {
      if (!activity.hidden) { activity.hidden = true; return; }
      const res = await api(`/api/devices/${d.id}/actions?limit=15`);
      activity.replaceChildren(...(res.actions.length ? res.actions.map(actionRow) : [h("p", { class: "hint" }, "Rien pour l'instant.")]));
      activity.hidden = false;
    }));
    const del = armed(h("button", { type: "button", class: "btn small ghost danger" }, icon("trash"), "Retirer"), "Confirmer le retrait", guard(async () => {
      await api(`/api/devices/${d.id}`, { method: "DELETE" });
      toast("Machine retirée. Elle n'a plus aucun accès. Sur la machine : python kira_core.py unpair (et uninstall).");
      loadDevices(); refreshStatus();
    }));
    card.append(h("div", { class: "dev-tools" }, pauseBtn, actBtn, del), activity);
    return card;
  }

  function actionRow(a) {
    const row = h("div", { class: "act" },
      h("div", { class: "act-top" }, h("span", { class: "pill " + (ACT_PILL[a.status] || "") }, ACT_LABEL[a.status] || a.status), h("span", {}, a.category_label), h("span", {}, relTime(a.created_at))),
      h("div", { class: "act-sum" }, a.summary));
    if (a.detail) row.append(h("details", { class: "tech" }, h("summary", {}, "Ce qui a été envoyé"), h("pre", {}, a.detail)));
    if (a.result) row.append(h("details", { class: "tech" }, h("summary", {}, "Réponse de la machine"), h("pre", {}, a.result)));
    (a.files || []).forEach((f) => row.append(thumb(f)));
    return row;
  }

  // ---- ajouter une machine : code à usage unique + commandes à copier
  const pair = { code: "", os: "linux", full: false, sha: "", known: new Set(), timer: null };

  function detectOs() {
    const ua = navigator.userAgent || "";
    if (/android/i.test(ua)) return "android";
    if (/windows/i.test(ua)) return "windows";
    return "linux";
  }

  function pairCommands() {
    const host = location.origin;
    const full = pair.full ? " --full" : "";
    if (pair.os === "windows") {
      return ["# PowerShell (Python 3 requis : python.org)", `curl.exe -L -o kira_core.py ${host}/core/kira_core.py`, `python kira_core.py pair --server ${host} --code ${pair.code}${full}`, "python kira_core.py install"].join("\n");
    }
    const py = pair.os === "android" ? "python" : "python3";
    return [pair.os === "android" ? "# Dans Termux : pkg install python curl" : "# Dans un terminal (Linux ou Mac)", `curl -fsSL ${host}/core/kira_core.py -o kira_core.py`,
      `${py} kira_core.py pair --server ${host} --code ${pair.code}${full}`, pair.os === "android" ? `${py} kira_core.py run` : `${py} kira_core.py install`].join("\n");
  }

  function renderPair() {
    const box = $("#dev-pair");
    const tabs = h("div", { class: "os-tabs", role: "tablist", "aria-label": "Type de machine" }, [["windows", "Windows"], ["linux", "Linux / Mac"], ["android", "Android"]].map(([k, label]) =>
      h("button", { type: "button", role: "tab", "aria-selected": String(pair.os === k), on: { click: () => { pair.os = k; renderPair(); } } }, label)));
    const copy = h("button", { type: "button", class: "copy-code", on: { click: async (e) => { try { await navigator.clipboard.writeText(pairCommands()); e.currentTarget.textContent = "Copié"; setTimeout(() => (e.currentTarget.textContent = "Copier"), 1500); } catch (err) { toast("Copie impossible : sélectionne le texte.", true); } } } }, "Copier");
    const full = h("input", { type: "checkbox", id: "pair-full" });
    full.checked = pair.full;
    full.addEventListener("change", () => { pair.full = full.checked; renderPair(); });
    box.replaceChildren(
      h("h3", { class: "sub-h", style: "margin:0" }, "Relier une machine"),
      h("p", { class: "hint" }, "1. Choisis le type de machine. 2. Colle ces commandes dessus. Le code ne sert qu'une fois et expire dans 10 minutes."),
      h("div", { class: "pair-code", "aria-label": "Code d'appairage" }, pair.code),
      tabs,
      h("div", { class: "cmd" }, h("pre", {}, pairCommands()), copy),
      h("label", { class: "check" }, full, h("span", {}, "Autoriser aussi les commandes, la vue de l'écran et la souris/clavier sur cette machine. Sinon ils restent coupés, et tu pourras les activer plus tard sur la machine elle-même.")),
      h("p", { class: "hint" }, "L'appli attend la machine : dès qu'elle est reliée, elle apparaît ici. Au début, tout ce qui modifie quelque chose demande ton accord."),
      pair.sha ? h("details", { class: "tech" }, h("summary", {}, "Vérifier le programme"), h("p", { class: "hint" }, "Empreinte SHA-256 de kira_core.py (compare avec sha256sum kira_core.py ou Get-FileHash) :"), h("pre", {}, pair.sha)) : null,
      h("div", { class: "actions", style: "margin-left:0" }, h("button", { type: "button", class: "btn ghost small", on: { click: startPair } }, icon("refresh"), "Nouveau code"), h("button", { type: "button", class: "btn ghost small", on: { click: closePair } }, "Fermer")));
  }

  function closePair() {
    clearInterval(pair.timer);
    pair.timer = null;
    show($("#dev-pair"), false);
  }

  const startPair = guard(async () => {
    const res = await api("/api/devices/pair-code", { method: "POST" });
    pair.code = res.code;
    pair.sha = res.agent_sha256 || "";
    pair.os = pair.os || detectOs();
    pair.known = new Set(((devData && devData.devices) || []).map((d) => d.id));
    renderPair();
    show($("#dev-pair"), true);
    $("#dev-pair").scrollIntoView({ block: "nearest", behavior: "smooth" });
    clearInterval(pair.timer);
    const until = Date.now() + (res.expires_in || 600) * 1000;
    pair.timer = setInterval(async () => {
      if (Date.now() > until || !state.authenticated) return closePair();
      try {
        const data = await api("/api/devices");
        const added = data.devices.find((d) => !pair.known.has(d.id));
        if (added) { closePair(); toast(`« ${added.name} » est reliée.`); devData = data; renderDevices(); refreshStatus(); }
      } catch (e) { /* on réessaie */ }
    }, 3000);
  });

  // ================= liaison de l'interface =================
  function bindUi() {
    state.bound = true;
    $$(".nav-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
    $("#history-open").addEventListener("click", openDrawer);
    $("#history-close").addEventListener("click", closeDrawer);
    $("#scrim").addEventListener("click", closeDrawer);
    $("#new-chat").addEventListener("click", () => { showView("chat"); newChat(); });
    $("#new-chat-rail").addEventListener("click", () => { showView("chat"); newChat(); });

    // titre : toucher pour renommer
    const title = $("#chat-title");
    const rename = () => {
      if (!state.convId) return;
      const input = h("input", { type: "text", class: "bar-title-input", maxlength: "120", value: title.textContent, "aria-label": "Titre de la conversation" });
      title.replaceWith(input);
      input.focus();
      input.select();
      let done = false;
      const finish = async (save) => {
        if (done) return;
        done = true;
        const v = input.value.trim();
        input.replaceWith(title);
        if (save && v && v !== title.textContent) {
          try { await api("/api/conversations/" + state.convId, { method: "PATCH", body: { title: v } }); title.textContent = v; loadHistory(); } catch (e) { if (e.status !== 401) toast(e.message, true); }
        }
      };
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); });
      input.addEventListener("blur", () => finish(true));
    };
    title.addEventListener("click", rename);
    title.addEventListener("keydown", (e) => { if (e.key === "Enter") rename(); });

    // saisie
    const input = $("#input");
    input.addEventListener("input", () => { autosize(); updateSend(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !window.matchMedia("(pointer: coarse)").matches) {
        e.preventDefault();
        $("#composer").requestSubmit();
      }
    });
    input.addEventListener("focus", () => document.body.classList.add("typing"));
    input.addEventListener("blur", () => document.body.classList.remove("typing"));
    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
    $$(".composer-row button").forEach((b) => b.addEventListener("pointerdown", (e) => { if (document.activeElement === input) e.preventDefault(); }));
    $("#deep").addEventListener("click", () => {
      setDeep(!state.deep);
      if (state.deep) toast("Réflexion approfondie pour ton prochain message : plus lent, plus coûteux.");
    });

    // copie des blocs de code
    $("#thread").addEventListener("click", async (e) => {
      const btn = e.target.closest(".copy-code");
      if (!btn) return;
      try { await navigator.clipboard.writeText($("pre", btn.parentElement).innerText); btn.textContent = "Copié"; setTimeout(() => (btn.textContent = "Copier"), 1500); } catch (err) { toast("Copie impossible.", true); }
    });

    // mémoire
    let t = null;
    $("#memory-q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(loadMemory, 250); });
    $("#memory-add-btn").addEventListener("click", () => { const f = $("#memory-add"); f.hidden = !f.hidden; if (!f.hidden) $("#memory-add-text").focus(); });
    $("#memory-add-cancel").addEventListener("click", () => { $("#memory-add").hidden = true; });
    $("#memory-add").addEventListener("submit", guard(async (e) => {
      e.preventDefault();
      const content = $("#memory-add-text").value.trim();
      if (!content) return;
      await api("/api/memory", { method: "POST", body: { kind: $("#memory-add-kind").value, content } });
      $("#memory-add-text").value = "";
      $("#memory-add").hidden = true;
      toast("Retenu.");
      loadMemory();
    }));

    // veille
    $("#veille-run").addEventListener("click", runVeille);
    $("#veille-all").addEventListener("click", guard(async () => {
      const r = await api("/api/veille/validate-all", { method: "POST", body: { min_score: 0.7 } });
      toast(r.validated ? `${plural(r.validated, "élément gardé", "éléments gardés")}.` : "Rien d'assez fiable à garder automatiquement.");
      loadVeille(); refreshStatus(); loadBriefing();
    }));

    // évolution
    $("#evo-form").addEventListener("submit", proposeEvolution);

    // machines
    pair.os = detectOs();
    $("#dev-add").addEventListener("click", startPair);
    $("#dev-pause-all").addEventListener("click", guard(async () => {
      const now = $("#dev-pause-all").getAttribute("aria-checked") !== "true";
      await api("/api/devices/pause-all", { method: "POST", body: { paused: now } });
      toast(now ? "Toutes les machines sont en pause : plus rien n'agit." : "Les machines reprennent.");
      loadDevices(); refreshStatus();
    }));

    // retour au premier plan : actualise l'état
    document.addEventListener("visibilitychange", () => { if (!document.hidden && state.authenticated) { refreshStatus(); schedulePending(300); if (!state.convId) loadBriefing(); } });
  }

  boot();
})();
