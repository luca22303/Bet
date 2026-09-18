"""Automatic team-news collection from several independent sources.

Diversity is the point, not convenience. Any single source has a house style, a
publication rhythm and blind spots: a club's own channel never reports friction
it would rather bury, an aggregator lags, a tabloid over-reports a knock as a
crisis. Reading several and letting the extraction layer's judge pass arbitrate
is what makes the resulting facts worth putting in front of a model.

Everything here is RSS or plain HTML, parsed with the standard library. No new
dependency, and the feeds are public and cheap to poll.

What this does not do is decide anything. It fetches documents and hands them to
`bet.extract`, where a local model turns them into entries and a second pass
verifies each one against the actual squad. A fetcher that also interpreted would
put unverified claims into the store, which is the failure the judge exists to
prevent.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import pandas as pd

from bet.ingest.base import IngestResult, Source
from bet.teams import UnknownTeamError, resolve

# Public RSS feeds, deliberately drawn from different kinds of publisher.
# `team` is set where a feed covers one club, and left None for league-wide
# feeds whose articles must be attributed by matching club names in the text.
FEEDS: list[dict] = [
    {"name": "kicker_bundesliga",
     "url": "https://newsfeed.kicker.de/news/bundesliga",
     "team": None, "language": "de"},
    {"name": "bundesliga_official",
     "url": "https://www.bundesliga.com/en/bundesliga/news/rss",
     "team": None, "language": "en"},
    {"name": "sportschau_fussball",
     "url": "https://www.sportschau.de/fussball/index~rss2.xml",
     "team": None, "language": "de"},
    {"name": "guardian_bundesliga",
     "url": "https://www.theguardian.com/football/bundesliga/rss",
     "team": None, "language": "en"},
]

# Words that mark an article as plausibly about availability. A crude gate, but
# it keeps the local model off match reports and transfer gossip, which is most
# of any football feed and none of what this pipeline wants.
RELEVANCE_TERMS = {
    "injur", "injured", "fitness", "doubt", "ruled out", "return", "suspend",
    "ban", "miss", "absence", "absent", "sidelined", "knock", "strain",
    "surgery", "recover",
    # German equivalents: German-language feeds are a large share of the useful
    # sources for this league.
    "verletz", "ausfall", "fällt aus", "fallt aus", "fraglich", "gesperrt",
    "sperre", "angeschlagen", "comeback", "rückkehr", "ruckkehr", "operation",
    "reha", "muskel", "bänder", "bander",
}


@dataclass
class Article:
    source: str
    title: str
    url: str
    published_at: datetime
    summary: str = ""
    language: str = "en"
    team_id: str | None = None

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.summary}".strip()

    def is_relevant(self) -> bool:
        lowered = self.text.lower()
        return any(term in lowered for term in RELEVANCE_TERMS)


def strip_tags(markup: str) -> str:
    """Plain text from an RSS description, which is usually escaped HTML."""
    if not markup:
        return ""
    text = html.unescape(markup)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(value: str | None) -> datetime | None:
    """RSS dates arrive in several shapes; try each and give up quietly."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value.strip(), fmt)
            return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            continue
    return None


def parse_feed(xml_text: str, source_name: str, *, language: str = "en") -> list[Article]:
    """Parse an RSS 2.0 or Atom feed into articles.

    Both shapes appear across these publishers, so both are handled rather than
    assuming one and silently returning nothing for the other.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []

    articles: list[Article] = []
    atom = "{http://www.w3.org/2005/Atom}"

    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in {"item", "entry"}:
            continue

        def field_text(*names: str) -> str:
            for name in names:
                node = item.find(name) if not name.startswith("{") else item.find(name)
                if node is None:
                    node = item.find(f"{atom}{name}")
                if node is not None and node.text:
                    return node.text.strip()
            return ""

        title = field_text("title")
        if not title:
            continue

        link = field_text("link", "guid")
        if not link:
            link_node = item.find(f"{atom}link")
            link = link_node.get("href", "") if link_node is not None else ""

        published = _parse_date(field_text("pubDate", "published", "updated"))
        articles.append(Article(
            source=source_name,
            title=strip_tags(title),
            url=link,
            published_at=published or datetime.utcnow(),
            summary=strip_tags(field_text("description", "summary", "content")),
            language=language,
        ))

    return articles


def attribute_team(article: Article, candidates: list[str] | None = None) -> str | None:
    """Work out which club a league-wide article is about.

    Returns None when two or more clubs are named with no clear subject. A
    preview mentioning both sides is not team news about either, and guessing
    would attach one club's injury list to its opponent.
    """
    from bet.teams import _LOOKUP, _normalise, known_team_ids

    text = _normalise(article.text)
    matched: set[str] = set()
    for alias, team_id in _LOOKUP.items():
        if len(alias) < 4:
            continue           # too short to match safely inside prose
        if alias in text:
            matched.add(team_id)

    if candidates:
        matched &= set(candidates)
    if len(matched) == 1:
        return matched.pop()

    if len(matched) > 1:
        # Prefer the club named in the headline, which is usually the subject.
        headline = _normalise(article.title)
        in_title = {tid for alias, tid in _LOOKUP.items()
                    if len(alias) >= 4 and alias in headline and tid in matched}
        if len(in_title) == 1:
            return in_title.pop()
    return None


class NewsSource(Source):
    """Fetches feeds and returns articles; it does not interpret them."""

    name = "news"

    def ingest(self, feeds: list[dict] | None = None, *, since_days: int = 5,
               relevant_only: bool = True, cache: bool = False) -> IngestResult:
        """Not a store write: returns articles for the extraction layer.

        `cache` defaults to False because a feed is a moving window and a cached
        copy of last week's is worse than no fetch at all.
        """
        result = IngestResult(source=self.name)
        feeds = feeds if feeds is not None else FEEDS
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        collected: list[Article] = []

        for feed in feeds:
            try:
                text, _ = self.fetch(feed["url"], suffix=".xml", cache=cache)
            except Exception as exc:
                result.errors.append(f"{feed['name']}: {exc}")
                continue
            result.documents_fetched += 1

            articles = parse_feed(text, feed["name"], language=feed.get("language", "en"))
            if not articles:
                result.errors.append(f"{feed['name']}: feed parsed but contained no items")
                continue

            for article in articles:
                if article.published_at < cutoff:
                    continue
                if relevant_only and not article.is_relevant():
                    continue
                article.team_id = feed.get("team") or attribute_team(article)
                collected.append(article)

        self.articles = collected
        result.rows_written["articles"] = len(collected)
        return result

    def fetch_articles(self, feeds: list[dict] | None = None, **kwargs) -> list[Article]:
        self.ingest(feeds, **kwargs)
        return getattr(self, "articles", [])


def group_by_team(articles: list[Article]) -> dict[str, list[Article]]:
    """Bundle articles per club, dropping those that could not be attributed."""
    grouped: dict[str, list[Article]] = {}
    for article in articles:
        if article.team_id:
            grouped.setdefault(article.team_id, []).append(article)
    return grouped
