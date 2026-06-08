"""URL helpers shared by planner and executor."""

from urllib.parse import urlparse


def normalise_url(url: str) -> str:
    """
    Keep scheme + host + path. Drop query params and fragments.
    For SPAs these are often ephemeral (search terms, scroll position).
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
