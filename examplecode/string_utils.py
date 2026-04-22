"""
String manipulation helpers for text processing tasks.
"""


def camel_to_snake(name):
    """Convert CamelCase string to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def truncate(text, max_length, suffix="..."):
    """Truncate text to max_length, appending suffix if shortened."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_words(text):
    """Count whitespace-separated words in a string."""
    return len(text.split())


def remove_punctuation(text):
    """Strip common punctuation characters from text."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    return "".join(ch for ch in text if ch not in punct)
