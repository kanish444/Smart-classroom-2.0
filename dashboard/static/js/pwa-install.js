// SmartClass Vision AI - PWA Install Controller
(function() {
  let deferredPrompt = null;

  function isStandalone() {
    return (
      (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) ||
      (window.navigator && window.navigator.standalone === true)
    );
  }

  function initPWAInstall() {
    const installBtn = document.getElementById("btnInstallApp");
    if (!installBtn) return;

    // Do not display if already running as standalone PWA
    if (isStandalone()) {
      installBtn.style.display = "none";
      return;
    }

    // 1. Capture beforeinstallprompt
    window.addEventListener("beforeinstallprompt", (e) => {
      // Prevent browser's automatic mini-infobar prompt
      e.preventDefault();
      // Stash deferred event for user click
      deferredPrompt = e;

      // Reveal the Install SmartClass button
      installBtn.style.display = "inline-flex";
    });

    // 2. User click triggers the saved install prompt
    installBtn.addEventListener("click", async () => {
      if (!deferredPrompt) return;

      try {
        await deferredPrompt.prompt();
        const choice = await deferredPrompt.userChoice;
        if (choice && choice.outcome === "accepted") {
          installBtn.style.display = "none";
        }
      } catch (err) {
        console.warn("PWA install prompt error:", err);
      } finally {
        deferredPrompt = null;
      }
    });

    // 3. Respond to appinstalled
    window.addEventListener("appinstalled", () => {
      deferredPrompt = null;
      installBtn.style.display = "none";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPWAInstall);
  } else {
    initPWAInstall();
  }
})();
