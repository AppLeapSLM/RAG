/**
 * Cloudflare Worker — Smart front door for AppLeap RAG.
 *
 * Routes: test.appleap.ai/*
 *
 * 1. Tries to reach the VM's FastAPI server.
 * 2. If reachable → proxies the request through.
 * 3. If unreachable → redirects to the Cloud Function to wake the VM.
 */

const VM_ORIGIN = "https://vper0v5y49cm1z-8000.proxy.runpod.net";
const WAKE_URL = "https://us-central1-project-c422bce9-652e-45ce-8b5.cloudfunctions.net/appleap-wake";
const HEALTH_TIMEOUT_MS = 4000;

export default {
  async fetch(request) {
    // Quick health check to see if VM is up
    const isUp = await checkHealth();

    if (isUp) {
      // Proxy the request to the VM
      const url = new URL(request.url);
      const vmUrl = VM_ORIGIN + url.pathname + url.search;

      const vmRequest = new Request(vmUrl, {
        method: request.method,
        headers: request.headers,
        body: request.body,
        redirect: "follow",
      });

      try {
        const response = await fetch(vmRequest);
        // Clone response with CORS headers for the proxied domain
        const newHeaders = new Headers(response.headers);
        newHeaders.set("Access-Control-Allow-Origin", "*");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      } catch {
        // VM was up a moment ago but request failed — show starting page
        return wakeAndWait();
      }
    }

    return wakeAndWait();
  },
};

async function checkHealth() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);

    const resp = await fetch(`${VM_ORIGIN}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    return resp.ok;
  } catch {
    return false;
  }
}

async function wakeAndWait() {
  // Fire-and-forget call to Cloud Function to start the VM
  try {
    fetch(WAKE_URL).catch(() => {});
  } catch {}

  const html = `<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>AppLeap RAG — Starting</title>
<style>
  body { font-family: -apple-system, sans-serif; display: flex;
         justify-content: center; align-items: center; height: 100vh;
         margin: 0; background: #f5f5f5; color: #1a1a1a; }
  .card { background: #fff; padding: 48px; border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
          max-width: 480px; }
  h1 { font-size: 22px; margin-bottom: 16px; }
  p { font-size: 15px; color: #555; line-height: 1.6; }
  .spinner { display: inline-block; width: 24px; height: 24px;
             border: 3px solid #ddd; border-top-color: #111;
             border-radius: 50%; animation: spin 0.8s linear infinite;
             margin-bottom: 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head><body>
<div class="card">
  <div class="spinner"></div>
  <h1>Starting up the GPU...</h1>
  <p>The server is booting. This usually takes about 60 seconds.<br>
  This page will auto-refresh.</p>
</div>
<script>setTimeout(() => location.reload(), 20000);</script>
</body></html>`;

  return new Response(html, {
    status: 503,
    headers: { "Content-Type": "text/html", "Retry-After": "30" },
  });
}
