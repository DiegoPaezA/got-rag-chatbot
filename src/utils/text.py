import re

# Patrones extraídos de tu archivo kg.py
SPLIT_PATTERN = r',|;|/|\n|<br\s*/?>|(?<=[a-z])(?=[A-Z])'

# Claves a ignorar en limpieza de propiedades
SKIP_KEYS = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'image', 'yes', 'no', 'caption', 'alt'
}

def clean_value(val: str) -> str:
    """
    Limpia un valor de texto proveniente del infobox o del título.
    Elimina wiki-markup ([[...]]), llaves y comillas.
    """
    if not val:
        return ""
    val = str(val)
    # Eliminar llaves anidadas dobles {{...}} y simples {...}
    val = re.sub(r'\{+(.*?)\}+', r'\1', val)
    # Eliminar enlaces wiki [[Link|Text]] -> Text
    val = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', val)
    # Eliminar referencias tipo [1], [note 1]
    val = re.sub(r'\[.*?\]', '', val)
    # Limpieza de caracteres
    val = val.replace("'", "").replace('"', '').strip()
    return val

def normalize_infobox_value(v) -> str:
    """Convierte listas o nulos a string plano."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x is not None)
    return str(v)