"""HS256 JWT issuance + validation.

Stateless tokens — no DB session table. The trade-off: tokens can't be revoked
before their `exp`. `jwt_lifetime_hours` (default 24) bounds the worst case.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.config import settings


class TokenError(Exception):
    """Token failed to decode/validate. Caller should respond 401."""


def encode_token(*, user_id: str, email: str, role: str) -> tuple[str, int]:
    """Issue a JWT for the given user. Returns (token, expires_in_seconds)."""
    if not settings.jwt_secret:
        raise RuntimeError("APPLEAP_JWT_SECRET is not set")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=settings.jwt_lifetime_hours)
    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    expires_in = int((exp - now).total_seconds())
    return token, expires_in


def decode_token(token: str) -> dict[str, Any]:
    """Verify signature + expiry; return the claims. Raises TokenError on any
    failure mode (signature mismatch, expired, malformed, missing fields).
    """
    if not settings.jwt_secret:
        raise TokenError("Authentication is not configured on this server")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "email", "role", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise TokenError("Token expired")
    except jwt.InvalidTokenError as e:
        raise TokenError(f"Invalid token: {e}")

    return payload
