"""
Pull README.md + docs/*.md from every repo (public and private) owned by
your GitHub account into data/raw/github/ — the first step of the
ingestion pipeline (Phase 1 of PLAN.md), run before app/ingestion/loader
parses these files into chunks.

Usage:
    python scripts/fetch_github_sources.py

Requires in .env:
    GITHUB_TOKEN     - a fine-grained personal access token with
                        read-only "Contents" and "Metadata" repository
                        permissions. No write scopes needed.
    GITHUB_USERNAME  - optional, informational only (not required by
                        the GitHub API call itself, which is scoped to
                        "the authenticated user" via the token).

Safe to re-run: only changed files are re-downloaded (tracked via git
blob SHA in data/raw/github/_manifest.json).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core import get_settings, setup_logging
import logfire
from app.ingestion.fetchers.github_fetcher import GitHubFetcher

setup_logging()


def main() -> None:
    settings = get_settings()
    token = settings.github.token.get_secret_value() if settings.github.token else None
    username = settings.github.username

    if not token:
        raise SystemExit(
            "Missing GITHUB_TOKEN in .env. Create a fine-grained PAT at "
            "https://github.com/settings/personal-access-tokens with read-only "
            "'Contents' + 'Metadata' access, then add it to .env (see .env.example)."
        )

    with logfire.span("Fetch GitHub Sources"):
        fetcher = GitHubFetcher(token=token, username=username)
        summary = fetcher.run()
        logfire.info("Summary: {summary}", summary=summary)


if __name__ == "__main__":
    main()
