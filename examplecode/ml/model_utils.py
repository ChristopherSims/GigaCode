"""
Machine learning model utilities for training and evaluation.
"""


def safe_get_param(config, key, default=None):
    """Return config[key] if present, else default."""
    try:
        return config[key]
    except (KeyError, TypeError):
        return default


def batch_iterator(data, batch_size):
    """Yield successive batches of data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def clip_gradient(grad, min_val, max_val):
    """Clamp a gradient value to a valid range."""
    if grad < min_val:
        return min_val
    if grad > max_val:
        return max_val
    return grad


def format_epoch_metrics(epoch, loss, accuracy):
    """Pretty-print training metrics for one epoch."""
    return f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {accuracy:.4f}"
