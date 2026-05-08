"""FastAPI dependencies for authentication and conversation ownership."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.tokens import TokenError, decode_token
from backend.db.connection import get_session
from backend.db.models import Conversation


@dataclass
class AuthedUser:
    """The user identity attached to an authenticated request. Sourced from
    the JWT — not loaded from the DB on every request.

    Note: a user disabled via `users.active=false` keeps a working token
    until it expires (≤ jwt_lifetime_hours). Acceptable trade-off for the
    stateless-token simplicity at our scale.
    """
    id: str
    email: str
    role: str


def _extract_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return authorization.split(" ", 1)[1].strip()


async def require_user(
    authorization: str | None = Header(default=None),
) -> AuthedUser:
    """Require any authenticated user. Use as: `Depends(require_user)`."""
    token = _extract_bearer(authorization)
    try:
        claims = decode_token(token)
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthedUser(id=claims["sub"], email=claims["email"], role=claims["role"])


async def require_admin(
    user: AuthedUser = Depends(require_user),
) -> AuthedUser:
    """Require an authenticated user with role='admin'."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


async def get_user_conversation(
    conversation_id: str,
    user: AuthedUser,
    session: AsyncSession,
) -> Conversation:
    """Load a Conversation only if it exists AND is owned by `user`.

    Returns 404 (not 403) on either miss — refusing to disclose whether the
    conversation exists at all if it isn't the caller's.
    """
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.owner_user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv
