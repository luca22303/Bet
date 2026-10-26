"""A read-only client for your own Kickbase squad, market and league.

Kickbase is a different kind of integration from everything else in this
project. Every other source feeds the point-in-time store for backtesting a
forecast; this one has no backtesting angle at all -- there is no "point in
time" for your own live fantasy squad, only "what does it look like right
now". That is why it lives in its own package, outside `bet.ingest`, and why
it never writes to the DuckDB store.

Kickbase publishes no API documentation. What is here is built from the
shape several independent community projects have reverse-engineered over
the years, not from anything verified against a live response -- outbound
access to kickbase.com is blocked from the environment this was written in,
the same as every other source this project treats as unverified. Unlike
those, this one also needs a login, which changes the right level of
caution: a wrong guess at a public page just returns no data, a wrong guess
at authentication risks tripping anti-automation defences on a real
account. So this client tries exactly one endpoint shape per call rather
than several in sequence (contrast `kicker.py`'s multi-strategy fallback,
safe only because every attempt there is a free, anonymous GET) and raises
with the full response body on failure instead of silently trying another
guess -- the fix is "tell me what Kickbase actually sent back", not "retry
louder". Kickbase's own public app has been rewritten at least once
(the "2023 relaunch") in ways that changed its API meaningfully, so even a
previously-correct guess can go stale.

This client never executes a trade. It only reads your squad, the market
and your league table; every buy/sell decision stays a recommendation for
you to act on in the app yourself, the same boundary `bet`'s own forecast
keeps between a prediction and a bet actually placed.

Credentials come from environment variables (KICKBASE_EMAIL,
KICKBASE_PASSWORD) or explicit constructor arguments for testing -- never a
command-line flag, which leaks into shell history, and never a config file,
which risks an accidental commit.
"""

from __future__ import annotations

import os

import requests

BASE_URL = "https://api.kickbase.com"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")


class KickbaseError(RuntimeError):
    """Base class, so callers can catch every Kickbase-specific failure at once."""


class KickbaseAuthError(KickbaseError):
    """Missing credentials, or the login call itself did not work."""


class KickbaseAPIError(KickbaseError):
    """A call after login failed. Carries the raw response for diagnosis."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _first(payload: dict, *keys, default=None):
    """The first present, non-None key -- the same tolerance `sofascore.py`
    uses, since this API's exact field names are a guess, not a certainty."""
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


class KickbaseClient:
    """One login, then plain authenticated GETs. Nothing here mutates anything."""

    def __init__(self, email: str | None = None, password: str | None = None,
                session: requests.Session | None = None, base_url: str = BASE_URL):
        self.email = email or os.environ.get("KICKBASE_EMAIL")
        self.password = password or os.environ.get("KICKBASE_PASSWORD")
        if not self.email or not self.password:
            raise KickbaseAuthError(
                "No credentials. Set KICKBASE_EMAIL and KICKBASE_PASSWORD as "
                "environment variables before running this -- never pass them as "
                "a command-line argument (shell history) or put them in a file "
                "that could be committed.")
        self.base_url = base_url
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.token: str | None = None
        self.user_id: str | None = None

    def login(self) -> dict:
        """One best-guess login call. Raises with the full response on failure
        rather than trying a second shape -- see the module docstring on why
        this one endpoint does not get the multi-strategy treatment the
        anonymous scrapers in this project do.
        """
        url = f"{self.base_url}/v4/user/login"
        try:
            response = self.session.post(
                url, json={"em": self.email, "password": self.password}, timeout=15)
        except requests.RequestException as exc:
            raise KickbaseAuthError(f"could not reach {url}: {exc}") from exc

        if response.status_code != 200:
            raise KickbaseAuthError(
                f"login to {url} returned {response.status_code}: "
                f"{response.text[:500]!r} -- this almost certainly means the "
                "endpoint or payload shape has changed since this was written; "
                "paste this error back so the guess can be corrected")

        try:
            payload = response.json()
        except ValueError as exc:
            raise KickbaseAuthError(
                f"login response from {url} was not JSON: {response.text[:300]!r}") from exc

        token = _first(payload, "token", "tkn", "accessToken", "access_token")
        if not token:
            raise KickbaseAuthError(
                f"login to {url} returned 200 but no recognisable token field -- "
                f"top-level keys were {sorted(payload.keys())}; paste this back "
                "so the field name can be corrected")

        self.token = token
        self.session.headers["Authorization"] = f"Bearer {token}"
        user = _first(payload, "user", "u", default={})
        self.user_id = _first(user, "id", "userId") if isinstance(user, dict) else None
        return payload

    def _get(self, path: str) -> dict:
        if not self.token:
            raise KickbaseAuthError("call login() first")
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, timeout=15)
        except requests.RequestException as exc:
            raise KickbaseAPIError(f"could not reach {url}: {exc}") from exc
        if response.status_code != 200:
            raise KickbaseAPIError(
                f"{url} returned {response.status_code}", status_code=response.status_code,
                body=response.text[:500])
        try:
            return response.json()
        except ValueError as exc:
            raise KickbaseAPIError(
                f"{url} did not return JSON: {response.text[:300]!r}") from exc

    def leagues(self) -> list[dict]:
        """Every league your account is a member of."""
        payload = self._get("/v4/leagues/selection")
        return _first(payload, "leagues", "items", default=[]) or []

    def squad(self, league_id: str) -> list[dict]:
        """Your own players in one league: id, name, position, current value."""
        payload = self._get(f"/v4/leagues/{league_id}/squad")
        return _first(payload, "players", "it", "squad", default=[]) or []

    def market(self, league_id: str) -> list[dict]:
        """Players currently listed for transfer in one league."""
        payload = self._get(f"/v4/leagues/{league_id}/market")
        return _first(payload, "players", "it", "market", default=[]) or []

    def player_performance(self, league_id: str, player_id: str) -> dict:
        """A player's recent match-by-match Kickbase points and market value
        history -- the basis for a consistency read, not just a season total."""
        return self._get(f"/v4/leagues/{league_id}/players/{player_id}/performance")

    def table(self, league_id: str) -> list[dict]:
        """The league standings -- where you actually stand."""
        payload = self._get(f"/v4/leagues/{league_id}/table")
        return _first(payload, "us", "users", "table", default=[]) or []
