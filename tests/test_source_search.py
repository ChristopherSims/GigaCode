"""Definition ranking must not prefer benchmark or example implementations."""

import pytest

from gigacode.source_search import SourceSearchIndex


@pytest.fixture
def source_index():
    sources = {
        "rich/text.py": '''
class Text:
    """Text with associated style."""
    def remove_suffix(self, suffix: str) -> None:
        """Remove a suffix if it exists."""
        if self.plain.endswith(suffix):
            self.right_crop(len(suffix))

    def get_style_at_offset(self, offset):
        """Get the style of a character at given offset."""
        return self.styles[offset]

    def rstrip(self):
        """Strip whitespace from end of text."""
        self.plain = self.plain.rstrip()
''',
        "rich/control.py": '''
def strip_control_codes(text):
    """Remove control codes from text."""
    return text.translate({})
''',
        "rich/_null_file.py": '''
class NullFile:
    def readable(self):
        """Return whether the file is readable."""
        return False
''',
        "rich/_pick.py": '''
def pick_bool(*values):
    """Pick the first non-none bool or return the last value.
    Any number of boolean or None values.
    """
    for value in values:
        if value is not None:
            return value
    return bool(values[-1])
''',
        "rich/color.py": '''
def parse_rgb_hex(hex_color):
    """Parse six hex characters in to RGB triplet."""
    assert len(hex_color) == 6, "must be 6 characters"
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def blend_rgb(color1, color2, cross_fade=0.5):
    """Blend one RGB color in to another."""
    return tuple(int(a + (b - a) * cross_fade) for a, b in zip(color1, color2))
''',
        "rich/color_triplet.py": '''
class ColorTriplet:
    @property
    def hex(self):
        """A hex string representation of the RGB triplet."""
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"
''',
        "rich/filesize.py": '''
"""Functions for converting file sizes in to human readable strings."""
def _to_str(size, suffixes, base, precision=1, separator=" "):
    if size == 1:
        return "1 byte"
    for suffix in suffixes:
        if size < base:
            break
        size /= base
    return f"{size:.{precision}f}{separator}{suffix}"

def decimal(size, precision=1, separator=" "):
    """Convert a filesize to a string using a decimal base."""
    return _to_str(size, ["bytes", "kB"], 1000, precision, separator)
''',
        "rich/progress.py": '''
def filesize(size):
    """Format size with decimal filesize strings."""
    return decimal(size)

def blend_progress(weight):
    """Blend progress colors with a fractional weight."""
    return weight
''',
        "examples/color.py": '''
def blend_rgb(color1, color2, cross_fade=0.5):
    """Mix two RGB colors using a fractional blend weight."""
    return color1
''',
        "benchmarks/text.py": '''
def suffix_strip_end_method_Text():
    """Suffix strip end method Text."""
    pass
''',
    }
    return SourceSearchIndex({file: source.splitlines() for file, source in sources.items()})


@pytest.mark.parametrize("query,name,file", [
    ("suffix strip end method Text", "Text.remove_suffix", "rich/text.py"),
    ("small bool picker helper first non-None fallback booleans", "pick_bool", "rich/_pick.py"),
    ("mixes two RGB colors with fractional weight", "blend_rgb", "rich/color.py"),
    ("RGB blend helper with a mix-weight argument", "blend_rgb", "rich/color.py"),
    ("hex-to-RGB triplet parsing helper", "parse_rgb_hex", "rich/color.py"),
    ("filesize string formatting internal helper", "_to_str", "rich/filesize.py"),
    ("file size string formatting helper", "_to_str", "rich/filesize.py"),
    ("Method strips a substring off the end of stored text when it matches", "Text.remove_suffix", "rich/text.py"),
    ("file size human readable helper", "_to_str", "rich/filesize.py"),
])
def test_failed_benchmark_queries_select_definitions(source_index, query, name, file):
    match = source_index.search(query, 1)["matches"][0]
    assert (match["name"], match["file"]) == (name, file)
    assert match["definition_match"] and match["confidence"] == "high"


def test_no_evidence_does_not_guess(source_index):
    assert source_index.search("zzyxx worble")["matches"] == []


def test_readable_file_query_does_not_get_filesize_domain_boost(source_index):
    assert source_index.search("readable file", 1)["matches"][0]["name"] == "NullFile.readable"


@pytest.mark.parametrize("symbol", ["Text.remove_suffix", "rich.text.Text.remove_suffix"])
def test_qualified_method_lookup(source_index, symbol):
    match = source_index.qualified_matches(symbol, 1)["matches"][0]
    assert match["name"] == "Text.remove_suffix" and match["confidence"] == "exact"


def test_explicit_example_file_can_be_selected(source_index):
    match = source_index.search("mix RGB colors", 1, file="examples/color.py")["matches"][0]
    assert match["file"] == "examples/color.py"


def test_equal_candidates_are_explicitly_uncertain():
    lines = ['def target(value):', '    """Do the special conversion."""', "    return value"]
    index = SourceSearchIndex({"one.py": lines, "two.py": lines})
    assert index.search("special conversion")["matches"][0]["confidence"] == "uncertain"


def test_non_python_and_invalid_python_remain_windows():
    index = SourceSearchIndex({
        "source.js": ["function needle() { return 1; }"],
        "invalid.py": ["def needle(:"],
    })
    matches = index.search("needle")["matches"]
    assert matches and all(not match["definition_match"] for match in matches)


def test_method_preference_is_structural_not_just_a_dotted_name():
    index = SourceSearchIndex({
        "source.py": [
            "class Source:",
            "    def remove_prefix(self, value, prefix):",
            "        if value.startswith(prefix):",
            "            return value[len(prefix):]",
            "        return value",
            "def caller(value):",
            '    """Method removes a substring from the start when it matches."""',
            "    return value",
        ],
    })
    match = index.search("method removes substring at the start when it matches", 1)["matches"][0]
    assert match["name"] == "Source.remove_prefix"
