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

from fastapi import Request

from autodft.config import Settings

COOKIE_NAME = "autodft_auth"
HEADER_NAME = "X-AutoDFT-Password"


def issue_token(password: str, lifetime_seconds: int) -> str:
    """Mint a session token that's valid until ``now + lifetime_seconds``."""
    expires_at = int(time.time()) + int(lifetime_seconds)
    sig = _sign(password, expires_at)
    return f"{expires_at}.{sig}"


def verify_token(token: str, password: str) -> bool:
    """Return True iff *token* was issued for *password* and hasn't expired."""
    if not token or "." not in token:
        return False
    try:
        expires_str, sig = token.split(".", 1)
        expires_at = int(expires_str)
    except (ValueError, AttributeError):
        return False
    if expires_at < int(time.time()):
        return False
    expected = _sign(password, expires_at)
    # Bytes for the same reason as the header comparison in
    # is_authenticated(): the cookie value is attacker-controlled.
    return hmac.compare_digest(
        sig.encode("utf-8", "replace"), expected.encode("utf-8")
    )


def _sign(password: str, expires_at: int) -> str:
    return hmac.new(
        password.encode("utf-8"),
        str(expires_at).encode("utf-8"),
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
    if cookie and verify_token(cookie, password):
        return True

    return False
