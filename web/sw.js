/* Service worker de KIRA : ouvre l'application même sans réseau. Les données (/api/) ne sont JAMAIS mises en cache. */
const VERSION = "kira-shell-v3-sessions";
const SHELL = [
  "/", "/style.css", "/app.js", "/manifest.webmanifest",
  "/vendor/marked.umd.js", "/vendor/purify.min.js", "/vendor/katex.min.js", "/vendor/katex.min.css",
  "/icons/icon-192.png", "/icons/favicon.png",
  "/vendor/fonts/lora.woff", "/vendor/fonts/lora-italic.woff",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname === "/healthz") return; // réseau uniquement

  const immutable = url.pathname.startsWith("/vendor/") || url.pathname.startsWith("/icons/");
  if (immutable) {
    // polices, bibliothèques, icônes : d'abord le cache
    event.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
        return res;
      }))
    );
    return;
  }
  // le reste (page, script, styles) : d'abord le réseau, le cache sert de secours hors ligne
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || (req.mode === "navigate" ? caches.match("/") : Response.error())))
  );
});
