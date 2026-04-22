"""
Feature extraction helpers for machine learning pipelines.
"""


def extract_numeric_features(record, fields):
    """Pull numeric values from a record for selected fields."""
    features = []
    for field in fields:
        raw = record.get(field, 0.0)
        try:
            features.append(float(raw))
        except (ValueError, TypeError):
            features.append(0.0)
    return features


def scale_features(values):
    """Min-max scale a list of numbers to [0, 1]."""
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val if max_val != min_val else 1.0
    return [(v - min_val) / span for v in values]


def drop_missing_samples(dataset):
    """Remove samples that contain any missing values."""
    cleaned = []
    for sample in dataset:
        if all(v is not None and str(v).strip() != "" for v in sample.values()):
            cleaned.append(sample)
    return cleaned


def compute_feature_mean(dataset, feature_name):
    """Compute mean of a feature across all samples."""
    total = 0.0
    count = 0
    for sample in dataset:
        try:
            total += float(sample[feature_name])
            count += 1
        except (KeyError, ValueError):
            continue
    return total / count if count > 0 else 0.0
