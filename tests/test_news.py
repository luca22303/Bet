"""News collection.

No network. Feeds are parsed from fixtures so the parsing, the relevance gate
and team attribution are all testable offline.
"""

from datetime import datetime, timedelta

import pytest

from bet.ingest.news import (
    FEEDS,
    Article,
    attribute_team,
    group_by_team,
    parse_feed,
    strip_tags,
)

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Kane faces late fitness test</title><link>http://x/1</link>
<pubDate>Fri, 27 Mar 2026 10:00:00 +0100</pubDate>
<description>&lt;p&gt;Bayern Munich striker Harry Kane is a doubt after a knock.&lt;/p&gt;</description></item>
<item><title>Transfer roundup</title><link>http://x/2</link>
<pubDate>Fri, 27 Mar 2026 09:00:00 +0100</pubDate>
<description>Summer rumours from around the league.</description></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Dortmund injury latest</title><link href="http://y/1"/>
<updated>2026-03-27T10:00:00Z</updated>
<summary>Zwei Spieler verletzt und fraglich fuer das Spiel.</summary></entry></feed>"""


def test_rss_is_parsed():
    articles = parse_feed(RSS, "test")
    assert len(articles) == 2
    assert articles[0].title == "Kane faces late fitness test"
    assert articles[0].url == "http://x/1"
    assert articles[0].published_at.year == 2026


def test_atom_is_parsed():
    """Both feed shapes appear across these publishers."""
    articles = parse_feed(ATOM, "test")
    assert len(articles) == 1
    assert articles[0].url == "http://y/1"


def test_malformed_feed_returns_nothing_rather_than_raising():
    assert parse_feed("not xml at all", "test") == []


def test_html_is_stripped_from_descriptions():
    # RSS descriptions arrive as escaped HTML, so unescaping comes first and
    # tag-stripping second.
    assert strip_tags("&lt;p&gt;Hello &amp;amp; goodbye&lt;/p&gt;") == "Hello &amp; goodbye"
    assert strip_tags("<p>Hello<br/>world</p>") == "Hello world"
    assert strip_tags("") == ""


def test_relevance_gate_filters_non_injury_news():
    """Keeps the local model off match reports and transfer gossip."""
    articles = parse_feed(RSS, "test")
    assert articles[0].is_relevant()
    assert not articles[1].is_relevant()


def test_german_terms_are_recognised():
    """German-language feeds are a large share of the useful sources."""
    assert parse_feed(ATOM, "test")[0].is_relevant()


def test_team_attribution_from_article_text():
    articles = parse_feed(RSS, "test")
    assert attribute_team(articles[0]) == "bayern_munich"


def test_ambiguous_articles_are_not_attributed():
    """A preview naming both sides is not team news about either.

    Guessing would attach one club's injury list to its opponent.
    """
    preview = Article("t", "Bayern Munich vs Borussia Dortmund preview",
                      "u", datetime.utcnow(), "Both sides have injury concerns")
    assert attribute_team(preview) is None


def test_headline_breaks_a_tie():
    article = Article("t", "Bayern Munich injury latest", "u", datetime.utcnow(),
                      "They face Borussia Dortmund on Saturday")
    assert attribute_team(article) == "bayern_munich"


def test_attribution_can_be_restricted_to_candidates():
    article = Article("t", "Bayern Munich injury latest", "u", datetime.utcnow(), "")
    assert attribute_team(article, candidates=["sc_freiburg"]) is None


def test_grouping_drops_unattributed_articles():
    articles = [
        Article("t", "Bayern Munich injury", "u1", datetime.utcnow(), ""),
        Article("t", "Something unattributable", "u2", datetime.utcnow(), ""),
    ]
    for article in articles:
        article.team_id = attribute_team(article)
    grouped = group_by_team(articles)
    assert set(grouped) == {"bayern_munich"}


def test_feed_list_is_diverse():
    """Several kinds of publisher, not several mirrors of one.

    A club channel buries what it would rather not report, an aggregator lags,
    a tabloid over-reports a knock. Reading across them is the point.
    """
    assert len(FEEDS) >= 3
    assert len({f["url"].split("/")[2] for f in FEEDS}) == len(FEEDS)
    assert {f.get("language") for f in FEEDS} >= {"de", "en"}


def test_article_text_combines_title_and_summary():
    article = Article("t", "Title", "u", datetime.utcnow(), "Summary")
    assert "Title" in article.text and "Summary" in article.text
