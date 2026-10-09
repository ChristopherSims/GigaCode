"""API connection settings shared by local clients."""


def api_base_url(host="localhost"):
    """Return the API base URL."""
    return f"http://{host}:8000"
