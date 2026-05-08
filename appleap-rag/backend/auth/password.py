"""Password hashing via bcrypt.

`bcrypt` truncates passwords longer than 72 bytes — UI-side validation should
cap inputs there. We don't pre-hash because that would invalidate any
external bcrypt verification later (recovery, migration to another stack).
"""

from __future__ import annotations

import bcrypt

from backend.config import settings

# Pre-computed bcrypt hash of an arbitrary string. Used in /auth/login when the
# email isn't found, so the response time matches a real (failed) password
# verification — defeats email-enumeration via timing.
# Generated once at import; the value never matches anything real.
DUMMY_HASH: str = bcrypt.hashpw(
    b"this-is-not-a-real-password-do-not-use",
    bcrypt.gensalt(rounds=settings.bcrypt_rounds),
).decode("utf-8")


def hash_password(plain: str) -> str:
    """Return a bcrypt hash for `plain`. Result encodes the salt + cost."""
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt(rounds=settings.bcrypt_rounds),
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check against a stored bcrypt hash. Returns False on any
    failure (malformed hash, mismatched length, wrong password) — never raises.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
