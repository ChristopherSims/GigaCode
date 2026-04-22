"""
Parsers for reading structured text and log files.
"""


def parse_key_value_pairs(text, delimiter="="):
    """Parse lines of 'key=value' into a dictionary."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if delimiter in line:
            key, value = line.split(delimiter, 1)
            result[key.strip()] = value.strip()
    return result


def parse_log_lines(lines):
    """Extract timestamp and level from simple log lines."""
    entries = []
    for line in lines:
        parts = line.strip().split(" ", 2)
        if len(parts) >= 3:
            entries.append({
                "timestamp": parts[0] + " " + parts[1],
                "level": parts[2].split(":", 1)[0],
                "message": parts[2].split(":", 1)[1].strip() if ":" in parts[2] else ""
            })
    return entries


def count_line_types(filepath):
    """Count blank, comment, and code lines in a file."""
    blank = comment = code = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                blank += 1
            elif stripped.startswith("#"):
                comment += 1
            else:
                code += 1
    return {"blank": blank, "comment": comment, "code": code}
