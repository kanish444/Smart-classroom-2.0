// SmartClass Vision AI - Minimal Safe PWA Service Worker
self.addEventListener("install", () => {
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(self.clients.claim());
});
