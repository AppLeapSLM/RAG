# Nango self-hosted deployment runbook

Self-hosted Nango sits between SaaS providers (GitHub, Slack, Notion, ...)
and the AppLeap RAG webhook receiver. Providers POST webhooks to Nango;
Nango normalizes, dedups against its own state, and forwards to our
`POST /webhooks/nango` endpoint.

```
SaaS provider → Nango (self-hosted on VM) → AppLeap RAG /webhooks/nango → jobs table → workers
```

This runbook covers a fresh deployment on the GCP VM `appleap-dev`.

## Prerequisites

- Docker + Docker Compose installed on the VM.
  ```
  sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
  sudo usermod -aG docker abhinav
  ```
- A Cloudflare Tunnel running on the VM (already in use for `test.appleap.ai`).
- Two new DNS records under `appleap.ai`:
  - `nango.appleap.ai` → CNAME to the existing tunnel target (for inbound provider webhooks → Nango)
  - `test.appleap.ai/webhooks/nango` → already covered by the existing tunnel route to the FastAPI app (for outbound Nango → AppLeap)

## Step 1 — Configure secrets

```bash
cd deploy/nango
cp .env.example .env
# Generate values:
echo "NANGO_ENCRYPTION_KEY=$(openssl rand -base64 32)" >> .env
echo "NANGO_DB_PASSWORD=$(openssl rand -hex 24)" >> .env
# Set NANGO_SERVER_URL=https://nango.appleap.ai (already in .env.example — uncomment/edit)

# Generate the Nango → AppLeap webhook signing secret separately. Pick any
# strong random string; you'll set this in two places:
#   - In Nango's UI under "Webhook Settings" as the signing secret
#   - As APPLEAP_NANGO_SIGNING_SECRET in the FastAPI service's systemd drop-in
openssl rand -hex 32
```

`.env` is gitignored. Treat all three values as production secrets.

## Step 2 — Set the FastAPI webhook signing secret

Create a systemd drop-in alongside the existing admin-token drop-in:

```bash
sudo tee /etc/systemd/system/appleap.service.d/nango-webhook.conf >/dev/null <<EOF
[Service]
Environment="APPLEAP_NANGO_SIGNING_SECRET=<paste the openssl rand -hex 32 from above>"
EOF
sudo chmod 0640 /etc/systemd/system/appleap.service.d/nango-webhook.conf
sudo systemctl daemon-reload
sudo systemctl restart appleap
```

Without this, the webhook receiver runs in dev mode (signature checks bypassed). Do not skip in production.

## Step 3 — Bring up Nango

```bash
cd deploy/nango
docker compose --env-file .env up -d
docker compose logs -f nango-server   # watch for "Nango server is up"
```

The server binds to `127.0.0.1:3003` — not exposed to the internet directly. Cloudflare Tunnel handles public ingress.

## Step 4 — Add Cloudflare Tunnel route

In the Cloudflare dashboard (or via `cloudflared tunnel route` CLI), add:

```
Hostname: nango.appleap.ai
Service:  http://localhost:3003
```

Verify:

```bash
curl -I https://nango.appleap.ai/health
# expect: HTTP/2 200
```

## Step 5 — Configure Nango's outbound webhook

Open `https://nango.appleap.ai` in a browser. In Settings → Webhook Settings:

- **Webhook URL:** `https://test.appleap.ai/webhooks/nango`
- **Webhook secret:** the value you generated in Step 2 (must match `APPLEAP_NANGO_SIGNING_SECRET`)
- Enable "Send webhooks for sync events" and "Send webhooks for auth events"

Save. Nango will POST a test webhook on save — check the FastAPI logs:

```bash
sudo journalctl -u appleap -f
# expect: nango_webhook accepted event_id=... provider=... jobs=N
```

## Step 6 — Run the worker as a systemd service

```bash
# Install the unit file
sudo cp deploy/systemd/appleap-worker.service /etc/systemd/system/

# Mirror the same secrets the API has (worker also needs them — it talks to
# Nango on outbound calls in Phase 4 and reads APPLEAP_NANGO_SIGNING_SECRET
# for verifying any internal handoffs).
sudo mkdir -p /etc/systemd/system/appleap-worker.service.d
sudo tee /etc/systemd/system/appleap-worker.service.d/secrets.conf >/dev/null <<EOF
[Service]
Environment="APPLEAP_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/appleap_rag"
Environment="APPLEAP_NANGO_SIGNING_SECRET=<paste same value as Step 2>"
EOF
sudo chmod 0640 /etc/systemd/system/appleap-worker.service.d/secrets.conf

sudo systemctl daemon-reload
sudo systemctl enable --now appleap-worker
sudo journalctl -u appleap-worker -f
# expect: worker starting id=appleap-dev:<pid>
```

## Step 7 — End-to-end smoke test

With no real provider connected yet, you can synth a webhook directly:

```bash
SECRET=$(sudo grep -oP 'APPLEAP_NANGO_SIGNING_SECRET=\K[^"]+' /etc/systemd/system/appleap.service.d/nango-webhook.conf)
BODY='{"providerConfigKey":"smoke","connectionId":"test","model":"docs","records":[{"id":"smoke-1","action":"upsert"}]}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $2}')
curl -i -X POST https://test.appleap.ai/webhooks/nango \
    -H "Content-Type: application/json" \
    -H "X-Nango-Signature: $SIG" \
    -d "$BODY"
# expect: 200 {"status":"accepted","event_id":"...","jobs":1}
```

Then check the worker log:
```bash
sudo journalctl -u appleap-worker -n 20
# expect:
#   job_start id=N action=upsert
#   upsert (placeholder) provider=smoke source_id=smoke-1 — connector handler not yet registered (Phase 4)
#   job_done id=N
```

If the placeholder log appears, Phases 1–3 are wired correctly end-to-end. The first real connector lands in Phase 4.

## Troubleshooting

- **401 from /webhooks/nango**: the signing secret in Nango's UI doesn't match `APPLEAP_NANGO_SIGNING_SECRET`. Re-paste both ends.
- **`nango-db` won't start**: usually leftover volume permissions. `docker compose down -v` and try again. (Wipes Nango's internal data — only safe before any real connections are configured.)
- **Webhook accepted but no job rows**: the Nango payload didn't have a `records`/`data` array. Check the logged event_id and inspect the raw body via Nango's "Webhook Logs" UI.
- **Stuck `locked_at`**: a worker crashed mid-job. The sweeper inside `backend.workers.run` clears these every minute.
- **`processed_events` table growth**: trim rows older than the longest provider retry window (~7 days). Add a cron later; not urgent.
