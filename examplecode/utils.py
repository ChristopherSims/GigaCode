"""
General-purpose utility functions used across the project.
"""


def safe_get(mapping, key, default=None):
    """Return mapping[key] if it exists, else default."""
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return default


def chunk_list(items, chunk_size):
    """Split a list into sublists of at most chunk_size elements."""
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


def clamp_value(value, lower, upper):
    """Restrict value to the inclusive range [lower, upper]."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def format_bytes(size):
    """Convert a byte count to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
