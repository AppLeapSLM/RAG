"""Authentication: bcrypt passwords + HS256 JWT tokens + FastAPI dependencies.

Public surface:
- hash_password / verify_password — password handling.
- DUMMY_HASH — used to keep failed-email logins constant-time.
- encode_token / decode_token — JWT issuance + validation.
- require_user / require_admin — FastAPI dependencies for gating endpoints.
- get_user_conversation — load a Conversation only if the caller owns it.
"""

from backend.auth.password import (
    DUMMY_HASH,
    hash_password,
    verify_password,
)
from backend.auth.tokens import encode_token, decode_token, TokenError
from backend.auth.dependencies import (
    AuthedUser,
    get_user_conversation,
    require_admin,
    require_user,
)

__all__ = [
    "DUMMY_HASH",
    "hash_password",
    "verify_password",
    "encode_token",
    "decode_token",
    "TokenError",
    "AuthedUser",
    "get_user_conversation",
    "require_admin",
    "require_user",
]
