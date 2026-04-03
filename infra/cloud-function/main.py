"""Cloud Function: Start the AppLeap RunPod pod if it's stopped, return status page."""

import json
import os
import urllib.request

import functions_framework

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
POD_ID = os.environ.get("RUNPOD_POD_ID", "vper0v5y49cm1z")
POD_URL = f"https://{POD_ID}-8000.proxy.runpod.net"
GRAPHQL_URL = "https://api.runpod.io/graphql"


def _runpod_query(query):
    """Execute a RunPod GraphQL query."""
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


@functions_framework.http
def handle(request):
    result = _runpod_query(
        f'query {{ pod(input: {{podId: "{POD_ID}"}}) {{ id desiredStatus runtime {{ uptimeInSeconds }} }} }}'
    )

    pod = result.get("data", {}).get("pod")
    if not pod:
        return _page("Pod not found", "Check RunPod dashboard.")

    desired = pod.get("desiredStatus", "")
    runtime = pod.get("runtime")
    is_running = runtime is not None

    if is_running:
        return _page(
            "Pod is running",
            f'Redirecting... <script>window.location.href="{POD_URL}";</script>',
        )

    if desired == "RUNNING":
        # Already starting
        return _page(
            "Starting up...",
            "The GPU pod is booting. This usually takes about 30 seconds."
            '<br><br>This page will auto-refresh.'
            '<script>setTimeout(()=>location.reload(), 15000);</script>',
        )

    # Pod is stopped — start it
    _runpod_query(
        f'mutation {{ podResume(input: {{podId: "{POD_ID}", gpuCount: 1}}) {{ id desiredStatus }} }}'
    )
    return _page(
        "Starting up...",
        "The GPU pod is booting. This usually takes about 30 seconds."
        '<br><br>This page will auto-refresh.'
        '<script>setTimeout(()=>location.reload(), 15000);</script>',
    )


def _page(title, body):
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>AppLeap RAG — {title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; display: flex;
         justify-content: center; align-items: center; height: 100vh;
         margin: 0; background: #f5f5f5; color: #1a1a1a; }}
  .card {{ background: #fff; padding: 48px; border-radius: 12px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
           max-width: 480px; }}
  h1 {{ font-size: 22px; margin-bottom: 16px; }}
  p {{ font-size: 15px; color: #555; line-height: 1.6; }}
</style>
</head><body>
<div class="card">
  <h1>{title}</h1>
  <p>{body}</p>
</div>
</body></html>"""
    return (html, 200, {"Content-Type": "text/html"})
