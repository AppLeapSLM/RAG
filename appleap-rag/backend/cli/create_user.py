"""Bootstrap CLI: create an AppLeap user via the admin-gated endpoint.

The first admin user is the only chicken-and-egg case in the auth design:
without an existing admin Bearer token, you can't call /auth/users via the
JWT path. This CLI uses the legacy X-Admin-Token (from
APPLEAP_ADMIN_TOKEN, same as the corpus ingest CLI) to create users
through the same /auth/users endpoint.

Usage:
    export APPLEAP_ADMIN_TOKEN=<token-from-systemd-drop-in>
    python -m backend.cli.create_user --email you@acme.com --role admin
    python -m backend.cli.create_user --email teammate@acme.com --role user --api-url https://test.appleap.ai

The password is read from stdin (hidden via getpass) — never passed on the
command line, so it doesn't end up in shell history or process listings.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

import httpx


def main() -> int:
    p = argparse.ArgumentParser(description="Create an AppLeap user")
    p.add_argument("--email", required=True, help="User's email (lowercased server-side)")
    p.add_argument(
        "--role",
        default="user",
        choices=["admin", "user"],
        help="Role to assign (default: user)",
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("APPLEAP_API_URL", "http://localhost:8000"),
        help="Base URL of the AppLeap API",
    )
    args = p.parse_args()

    token = os.environ.get("APPLEAP_ADMIN_TOKEN")
    if not token:
        print(
            "ERROR: APPLEAP_ADMIN_TOKEN is not set. The bootstrap CLI uses the legacy\n"
            "admin secret (the same one /ingest/file accepts) to create the first user.",
            file=sys.stderr,
        )
        return 1

    password = getpass.getpass(f"Password for {args.email}: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("ERROR: Passwords do not match.", file=sys.stderr)
        return 1
    if len(password) < 12:
        print("ERROR: Password must be at least 12 characters.", file=sys.stderr)
        return 1

    url = args.api_url.rstrip("/") + "/auth/users"
    try:
        resp = httpx.post(
            url,
            json={"email": args.email, "password": password, "role": args.role},
            headers={"X-Admin-Token": token},
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        print(f"ERROR: HTTP request failed: {e}", file=sys.stderr)
        return 2

    if resp.status_code == 200:
        body = resp.json()
        print(
            f"Created user id={body['id']} email={body['email']} "
            f"role={body['role']} active={body['active']}"
        )
        return 0

    print(f"ERROR: server returned {resp.status_code}: {resp.text}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
