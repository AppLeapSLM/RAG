/**
 * Cloudflare Worker — Proxy for AppLeap RAG.
 *
 * Routes: test.appleap.ai/*
 * Proxies all requests to the GCP VM (always running).
 */

const VM_ORIGIN = "http://vm.appleap.ai:8000";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const vmUrl = VM_ORIGIN + url.pathname + url.search;

    try {
      const headers = new Headers(request.headers);
      headers.set("Host", "vm.appleap.ai");

      const vmRequest = new Request(vmUrl, {
        method: request.method,
        headers: headers,
        body: request.body,
        redirect: "follow",
      });

      const response = await fetch(vmRequest);
      const newHeaders = new Headers(response.headers);
      newHeaders.set("Access-Control-Allow-Origin", "*");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
      });
    } catch {
      return new Response(
        "<h1>Server is temporarily unavailable</h1><p>Please try again in a moment.</p>",
        { status: 503, headers: { "Content-Type": "text/html" } },
      );
    }
  },
};
