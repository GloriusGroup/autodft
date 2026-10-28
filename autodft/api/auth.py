"""Session cookies for the dashboard.

There is exactly one credential in AutoDFT: a personal API key. Scripts
send it as ``X-AutoDFT-API-Key`` (or ``Authorization: Bearer``); browsers
exchange it once at ``/login`` for a session cookie, which is what this
module mints and checks. There is no shared password: a secret that every
user knows identifies nobody, and it was admin.

The cookie is stateless. It encodes the expiry, the username it was
issued for, a fingerprint of that user's key at the time, and an
HMAC-SHA256 signature keyed by ``Settings.session_secret()`` — a random
value persisted under the data path, not something anyone types.
Restarting the controller does not invalidate sessions; rotating a user's
key invalidates theirs, and deleting ``.session_secret`` invalidates all
of them.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

COOKIE_NAME = "autodft_auth"


def issue_token(
    secret: str, lifetime_seconds: int, username: str, key_fingerprint: str,
) -> str:
    """Mint a session token for *username*, valid for *lifetime_seconds*.

    The token carries the username, so the cookie says *who* is signed in
    and not merely *that* someone is, and the fingerprint of the key it
    was issued under, so rotating that key ends the session. Both are
    inside the signed payload and cannot be edited without the secret.
    """
    expires_at = int(time.time()) + int(lifetime_seconds)
    payload = f"{expires_at}.{username}.{key_fingerprint}"
    return f"{payload}.{_sign(secret, payload)}"


def verify_token(token: str, secret: str) -> Optional[tuple[str, str]]:
    """``(username, key_fingerprint)`` of *token*, or None if it does not verify."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    expires_str, username, fingerprint, sig = parts

    try:
        expires_at = int(expires_str)
    except (ValueError, TypeError):
        return None
    if expires_at < int(time.time()):
        return None

    # Compare as bytes: compare_digest refuses str operands containing
    # non-ASCII, and the cookie value is attacker-controlled, so a single
    # high byte otherwise raised TypeError out of the auth middleware and
    # turned every request into a logged 500.
    if not hmac.compare_digest(
        sig.encode("utf-8", "replace"),
        _sign(secret, f"{expires_str}.{username}.{fingerprint}").encode("utf-8"),
    ):
        return None
    return username, fingerprint


def _sign(secret: str, payload) -> str:
    return hmac.new(
        str(secret).encode("utf-8"),
        str(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
