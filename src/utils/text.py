import re

# Patterns extracted from the kg.py file
SPLIT_PATTERN = r',|;|/|\n|<br\s*/?>|(?<=[a-z])(?=[A-Z])'

# Keys to skip while cleaning properties
SKIP_KEYS = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'image', 'yes', 'no', 'caption', 'alt'
}

def clean_value(val: str) -> str:
    """Clean a text value from an infobox or title by removing wiki markup, braces, and quotes."""
    if not val:
        return ""
    val = str(val)
    # Remove nested double braces {{...}} and single braces {...}
    val = re.sub(r'\{+(.*?)\}+', r'\1', val)
    # Remove wiki links [[Link|Text]] -> Text
    val = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', val)
    # Remove reference markers like [1] or [note 1]
    val = re.sub(r'\[.*?\]', '', val)
    # Limpieza de caracteres
    val = val.replace("'", "").replace('"', '').strip()
    return val

def normalize_infobox_value(v) -> str:
    """Convert lists or nulls to a flat string."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x is not None)
    return str(v)