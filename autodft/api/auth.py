"""Cookie + header authentication for the dashboard and the JSON API.

The same shared secret protects two routes of entry:

* **Dashboard / browser** — login form sets an HMAC-signed session
  cookie. Subsequent requests carry the cookie until it expires.
* **Scripts** — send the password via the ``X-AutoDFT-Password`` header
  on every request. No cookie needed.

The cookie is stateless: it encodes the expiry time and an HMAC-SHA256
signature keyed by the configured password. Restarting the controller
does not invalidate sessions. Changing the password invalidates every
cookie immediately (the signatures stop verifying).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import Request

from autodft.config import Settings

COOKIE_NAME = "autodft_auth"
HEADER_NAME = "X-AutoDFT-Password"


def issue_token(password: str, lifetime_seconds: int, username: str = "") -> str:
    """Mint a session token valid until ``now + lifetime_seconds``.

    The token carries the username it was issued for, so the cookie says
    *who* is logged in and not merely *that* someone is. It stays
    stateless: the username is inside the signed payload, so it cannot be
    edited without the dashboard password.
    """
    expires_at = int(time.time()) + int(lifetime_seconds)
    payload = f"{expires_at}.{username}"
    return f"{payload}.{_sign(password, payload)}"


def verify_token(token: str, password: str) -> Optional[str]:
    """The username *token* was issued for, or None if it does not verify.

    Returns the empty string for a valid pre-accounts token, which carried
    no username. Callers distinguish "no session" (None) from "a session
    with no user attached" ("") -- the latter resolves to admin, so
    sessions opened before the upgrade keep working until they expire.
    """
    if not token or "." not in token:
        return None
    parts = token.split(".")
    if len(parts) == 2:
        expires_str, sig = parts        # legacy: {expires}.{sig}
        username = ""
    elif len(parts) == 3:
        expires_str, username, sig = parts
    else:
        return None

    try:
        expires_at = int(expires_str)
    except (ValueError, TypeError):
        return None
    if expires_at < int(time.time()):
        return None

    payload = expires_str if len(parts) == 2 else f"{expires_str}.{username}"
    # Bytes for the same reason as the header comparison in
    # is_authenticated(): the cookie value is attacker-controlled.
    if not hmac.compare_digest(
        sig.encode("utf-8", "replace"), _sign(password, payload).encode("utf-8")
    ):
        return None
    return username


def _sign(password: str, payload) -> str:
    return hmac.new(
        password.encode("utf-8"),
        str(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def is_authenticated(request: Request, settings: Settings) -> bool:
    """True iff the request carries a valid cookie OR a matching header."""
    password = settings.security.dashboard_password

    # Header path — scripts / curl / Python urllib.
    # Compare as bytes: compare_digest refuses str operands containing
    # non-ASCII, and both the header and the cookie are attacker-controlled,
    # so a single high byte otherwise raised TypeError out of the auth
    # middleware and turned every request into a logged 500.
    header_val = request.headers.get(HEADER_NAME)
    if header_val and hmac.compare_digest(
        header_val.encode("utf-8", "replace"), password.encode("utf-8")
    ):
        return True

    # Cookie path — browser flow via /login
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie and verify_token(cookie, password) is not None:
        return True

    return False
