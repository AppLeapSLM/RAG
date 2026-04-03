#!/bin/bash
# Shuts down the VM if no HTTP requests in the last 30 minutes.
# Install: crontab -e → */5 * * * * /opt/appleap/idle-shutdown.sh
#
# How it works:
#   - The FastAPI server touches /tmp/appleap-last-request on every request (via middleware).
#   - This script checks the age of that file.
#   - If older than IDLE_MINUTES (or missing), it shuts down.

IDLE_MINUTES=15
STAMP_FILE="/tmp/appleap-last-request"

if [ ! -f "$STAMP_FILE" ]; then
    # No stamp file yet. Check if the server is running at all.
    if curl -s --max-time 3 http://localhost:8000/health > /dev/null 2>&1; then
        # Server is up but no requests yet — touch the stamp and wait.
        touch "$STAMP_FILE"
    fi
    # Either way, don't shut down — give the server time to receive traffic.
    exit 0
fi

LAST_MODIFIED=$(stat -c %Y "$STAMP_FILE")
NOW=$(date +%s)
AGE_MINUTES=$(( (NOW - LAST_MODIFIED) / 60 ))

if [ "$AGE_MINUTES" -ge "$IDLE_MINUTES" ]; then
    logger "appleap: Idle for ${AGE_MINUTES} minutes. Shutting down."
    sudo shutdown -h now
fi
