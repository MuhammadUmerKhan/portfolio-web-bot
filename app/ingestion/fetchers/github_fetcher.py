"""
GitHub source fetcher.

Pulls README.md (at any depth — repo root and every subdirectory) and any
*.md files under a docs/ folder (at any depth) from every repository
(public + private) owned by the authenticated GitHub account, and saves
them into data/raw/github/<repo_name>/... for the ingestion pipeline
(app/ingestion/loader) to pick up.

Design notes:
- Uses /user/repos (authenticated) rather than /users/{username}/repos,
    because the latter only ever returns public repos regardless of token.
- Idempotent: each file's git blob SHA is tracked in a manifest
    (data/raw/github/_manifest.json). Re-running only re-downloads files
    whose SHA changed since the last run, so this is safe to schedule
    (e.g. a nightly GitHub Actions cron) without re-fetching everything.
- Forks are skipped by default — the point of this corpus is your own
    work, not code you happen to have forked.
- Rate limits: authenticated requests get 5,000/hour, which is enormous
    headroom for a personal repo count. Verify current limits at
    https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting
    if this is ever run against an account with hundreds of repos.
"""

from __future__ import annotations

import base64
import json
import logfire
import time
from pathlib import Path
from typing import Optional
from app.core import get_settings

import requests

GITHUB_API = "https://api.github.com"
OUTPUT_ROOT = Path("data/raw/github")
MANIFEST_PATH = OUTPUT_ROOT / "_manifest.json"

# Directories to never descend into, even if they show up in a repo's tree
# (e.g. a vendored dependency or a committed venv) — a README inside one of
# these isn't your own work and would just add noise to the corpus.
EXCLUDED_DIR_NAMES = {
    "node_modules", "vendor", ".git", "dist", "build", "__pycache__",
    ".venv", "venv", "third_party", "external", ".github",
}


class GitHubFetcher:
    def __init__(self, token: Optional[str] = None, username: Optional[str] = None):
        settings = get_settings()
        final_token = token or (settings.github.token.get_secret_value() if settings.github.token else None)
        final_username = username or settings.github.username

        if not final_token:
            raise ValueError(
                "GITHUB_TOKEN is required (a fine-grained PAT with read-only "
                "'Contents' + 'Metadata' access to your repos is enough)."
            )
        self.token = final_token
        self.username = final_username  # informational only; not required for /user/repos
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.manifest: dict[str, str] = self._load_manifest()

    # ---------- manifest (idempotency) ----------

    def _load_manifest(self) -> dict[str, str]:
        if MANIFEST_PATH.exists():
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    # ---------- low-level request with rate-limit backoff ----------

    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        resp = None
        for attempt in range(3):
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 5)
                logfire.warning("Rate limited. Waiting {wait}s before retry.", wait=wait)
                time.sleep(min(wait, 120))
                continue
            return resp
        return resp

    # ---------- GitHub API calls ----------

    def list_repos(self) -> list[dict]:
        """List every repo the authenticated user owns (public + private), paginated."""
        repos: list[dict] = []
        page = 1
        while True:
            resp = self._get(
                f"{GITHUB_API}/user/repos",
                params={"affiliation": "owner", "per_page": 100, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            repos.extend(batch)
            page += 1
        logfire.info("Found {count} owned repos.", count=len(repos))
        return repos

    def list_markdown_targets(self, owner: str, repo: str, default_branch: str) -> list[str]:
        """Return paths for every README.md (at any depth) and every *.md file
        that lives under a docs/ folder (at any depth) in the repo.

        Catches monorepo-style layouts (backend/README.md, packages/ui/README.md,
        packages/ui/docs/setup.md, ...) rather than only the repo root.
        """
        resp = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if resp.status_code != 200:
            logfire.warning("Could not list tree for {owner}/{repo}: {status_code}", owner=owner, repo=repo, status_code=resp.status_code)
            return []
        tree = resp.json().get("tree", [])
        targets = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = entry["path"]
            parts = path.split("/")
            dir_parts = [p.lower() for p in parts[:-1]]
            if any(p in EXCLUDED_DIR_NAMES for p in dir_parts):
                continue
            basename = parts[-1].lower()
            if basename == "readme.md":
                targets.append(path)
            elif basename.endswith(".md") and "docs" in dir_parts:
                targets.append(path)
        return targets

    def fetch_file(self, owner: str, repo: str, path: str) -> Optional[dict]:
        resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")
        if resp.status_code != 200:
            logfire.warning("Could not fetch {owner}/{repo}/{path}: {status_code}", owner=owner, repo=repo, path=path, status_code=resp.status_code)
            return None
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return {"sha": data["sha"], "content": content}

    # ---------- orchestration ----------

    def run(self) -> dict:
        repos = self.list_repos()
        fetched, skipped, failed = 0, 0, 0

        for repo in repos:
            if repo.get("fork"):
                continue  # skip forks — corpus should be your own work only

            owner = repo["owner"]["login"]
            name = repo["name"]
            default_branch = repo.get("default_branch", "main")

            targets = self.list_markdown_targets(owner, name, default_branch)
            if not targets:
                continue

            repo_dir = OUTPUT_ROOT / name
            repo_dir.mkdir(parents=True, exist_ok=True)

            # repo-level metadata — consumed later by Phase 1 metadata
            # extraction and Phase 4's knowledge graph (project entity).
            repo_meta = {
                "name": name,
                "description": repo.get("description"),
                "url": repo.get("html_url"),
                "language": repo.get("language"),
                "topics": repo.get("topics", []),
                "pushed_at": repo.get("pushed_at"),
                "private": repo.get("private", False),
            }
            (repo_dir / "_meta.json").write_text(
                json.dumps(repo_meta, indent=2), encoding="utf-8"
            )

            for path in targets:
                manifest_key = f"{name}/{path}"
                file_info = self.fetch_file(owner, name, path)
                if file_info is None:
                    failed += 1
                    continue

                if self.manifest.get(manifest_key) == file_info["sha"]:
                    skipped += 1
                    continue  # unchanged since last run — idempotent skip

                local_path = repo_dir / path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(file_info["content"], encoding="utf-8")
                self.manifest[manifest_key] = file_info["sha"]
                fetched += 1
                logfire.info("Saved {manifest_key}", manifest_key=manifest_key)

        self._save_manifest()
        summary = {"fetched": fetched, "skipped_unchanged": skipped, "failed": failed}
        logfire.info(
            "Done. Fetched: {fetched}, unchanged (skipped): {skipped}, failed: {failed}",
            fetched=fetched,
            skipped=skipped,
            failed=failed,
        )
        return summary
