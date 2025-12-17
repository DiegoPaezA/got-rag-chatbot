import re
from typing import Any, Dict, List, Optional

from src.utils.text import clean_value, SPLIT_PATTERN


def clean_rel_key(key: Any) -> str:
    if key is None:
        return ""
    s = str(key).strip()
    if not s:
        return ""
    if "(" in s:
        s = s.split("(")[0].strip()
    return s


def merge_rel_sources(raw_props: Dict[str, Any], norm_rel: Dict[str, Any], rel_map: Dict[str, str]) -> Dict[str, Any]:
    rel_source: Dict[str, Any] = {}

    # normalized_relations wins
    for k, v in (norm_rel or {}).items():
        ck = clean_rel_key(k)
        if ck:
            rel_source[ck] = v

    # raw fallback only if not present and mapped in rel_map
    for k, v in (raw_props or {}).items():
        ck = clean_rel_key(k)
        if not ck or ck == "AKA":
            continue
        if ck in rel_source:
            continue
        if ck in rel_map:
            rel_source[ck] = v

    return rel_source


def parse_targets(val: Any, *, rel_type: Optional[str] = None) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]

    val_str = str(val)

    # Convert " and " to comma except for episode-like strings
    if rel_type not in {"DIED_IN_EPISODE"}:
        val_str = val_str.replace(" and ", ",")
        
    if rel_type in {"SEATED_AT", "LOCATED_IN"} and "," in val_str:
        return [t.strip() for t in val_str.split(",")]

    parts = [t.strip() for t in re.split(SPLIT_PATTERN, val_str) if t.strip()]
    return [clean_value(t) for t in parts if t]


def is_valid_target(target: str, source_id: str) -> bool:
    if not target:
        return False
    if target == source_id:
        return False
    if len(target) < 3:
        return False
    if "note" in target.lower():
        return False
    return True
