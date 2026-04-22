"""
Data processing utilities for cleaning and transforming tabular data.
"""


def load_csv_rows(filepath):
    """Read a CSV file and return a list of row dictionaries."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        headers = [h.strip() for h in f.readline().split(",")]
        for line in f:
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    return rows


def normalize_numeric_column(values):
    """Normalize a list of numeric strings to floats in [0, 1]."""
    nums = [float(v) for v in values if v.strip() != ""]
    min_val = min(nums)
    max_val = max(nums)
    span = max_val - min_val if max_val != min_val else 1.0
    return [(v - min_val) / span for v in nums]


def remove_null_rows(data):
    """Drop rows where any value is empty or None."""
    cleaned = []
    for row in data:
        if all(v is not None and str(v).strip() != "" for v in row.values()):
            cleaned.append(row)
    return cleaned


def compute_column_mean(data, column):
    """Compute the arithmetic mean of a numeric column."""
    total = 0.0
    count = 0
    for row in data:
        try:
            total += float(row[column])
            count += 1
        except (KeyError, ValueError):
            continue
    return total / count if count > 0 else 0.0
