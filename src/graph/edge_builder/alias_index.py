from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from src.utils.text import clean_value, SPLIT_PATTERN
import re

logger = logging.getLogger(__name__)


@dataclass
class AliasIndex:
    honorifics: Set[str]
    placeholder_keys: Set[str]

    alias_unique: Dict[str, str] = None
    alias_multi: Dict[str, Set[str]] = None
    alias_max_len: int = 40

    def __post_init__(self) -> None:
        self.alias_unique = {}
        self.alias_multi = {}

    def norm(self, s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    def is_placeholder(self, s: str) -> bool:
        if not s:
            return True
        return self.norm(s) in self.placeholder_keys

    def strip_honorifics(self, s: str) -> str:
        s = str(s).strip()
        if not s:
            return s
        toks = s.split()
        removed = 0
        while toks and toks[0].lower() in self.honorifics and removed < 2:
            toks = toks[1:]
            removed += 1
        return " ".join(toks).strip()

    def add_alias_safely(self, alias_raw: str, canonical_id: str) -> None:
        key = self.norm(alias_raw)
        if not key:
            return

        if key in self.alias_multi:
            self.alias_multi[key].add(canonical_id)
            return

        if key in self.alias_unique:
            existing = self.alias_unique[key]
            if existing != canonical_id:
                self.alias_multi[key] = {existing, canonical_id}
                del self.alias_unique[key]
            return

        self.alias_unique[key] = canonical_id

    def build(self, nodes_map: Dict[str, Dict[str, Any]]) -> None:
        alias_raw: Dict[str, Set[str]] = {}

        def add(alias: str, nid: str) -> None:
            alias = str(alias).strip()
            if not alias:
                return

            variants = [alias]

            stripped = self.strip_honorifics(alias)
            if stripped and stripped != alias:
                variants.append(stripped)

            cv = clean_value(alias)
            if cv and cv != alias:
                variants.append(cv)

            for v in variants:
                k = self.norm(v)
                if k:
                    alias_raw.setdefault(k, set()).add(nid)

        for nid, node in nodes_map.items():
            props = node.get("properties", {}) or {}
            norm_rel = node.get("normalized_relations", {}) or {}

            add(nid, nid)

            if props.get("Title"):
                add(props["Title"], nid)

            aka_list: List[str] = []
            if isinstance(norm_rel.get("AKA"), list):
                aka_list = norm_rel["AKA"]
            elif props.get("AKA"):
                aka_list = [t.strip() for t in re.split(SPLIT_PATTERN, str(props["AKA"])) if t.strip()]

            for aka in aka_list:
                add(aka, nid)

            if node.get("type") == "House":
                if nid.lower().startswith("house "):
                    surname = nid[6:].strip()
                    if surname:
                        add(surname, nid)
                if props.get("Type"):
                    for token in re.split(SPLIT_PATTERN, str(props["Type"])):
                        tok = token.strip()
                        if tok:
                            add(tok, nid)

        # finalize
        for k, ids in alias_raw.items():
            if len(ids) == 1:
                self.alias_unique[k] = next(iter(ids))
            elif 1 < len(ids) <= 5:
                self.alias_multi[k] = ids

        all_keys = list(self.alias_unique.keys()) + list(self.alias_multi.keys())
        self.alias_max_len = max((len(k) for k in all_keys), default=40)

        logger.info(
            f"🧭 Alias map built: {len(self.alias_unique)} unique, {len(self.alias_multi)} small-ambiguous. "
            f"(max_key_len={self.alias_max_len})"
        )

    def resolve(
        self,
        *,
        raw_target: str,
        rel_type: str,
        nodes_map: Dict[str, Dict[str, Any]],
        expected_target_types: List[str],
    ) -> Tuple[str, str]:
        raw_target = str(raw_target).strip()
        if not raw_target:
            return raw_target, "none"

        if raw_target in nodes_map:
            return raw_target, "exact"

        stripped = self.strip_honorifics(raw_target)
        if stripped and stripped != raw_target and stripped in nodes_map:
            return stripped, "exact"

        key = self.norm(raw_target)
        if not key:
            return raw_target, "none"

        if key in self.alias_unique:
            cand = self.alias_unique[key]
            if expected_target_types and nodes_map.get(cand, {}).get("type") not in expected_target_types:
                pass
            else:
                return cand, "alias"

        if stripped and stripped != raw_target:
            key2 = self.norm(stripped)
            if key2 in self.alias_unique:
                cand = self.alias_unique[key2]
                if not expected_target_types or nodes_map.get(cand, {}).get("type") in expected_target_types:
                    return cand, "alias"

        if key in self.alias_multi:
            candidates = list(self.alias_multi[key])
            if expected_target_types:
                filtered = [cid for cid in candidates if nodes_map.get(cid, {}).get("type") in expected_target_types]
                if len(filtered) == 1:
                    return filtered[0], "alias"
                if len(filtered) > 1:
                    candidates = filtered

            if "House" in expected_target_types:
                house_pref = [cid for cid in candidates if cid.lower().startswith("house ")]
                if len(house_pref) == 1:
                    return house_pref[0], "alias"

            return raw_target, "ambiguous_unresolved"

        # house prefix heuristics
        if "House" in expected_target_types or rel_type in {"BELONGS_TO", "SWORN_TO", "VASSAL_OF", "OVERLORD_OF"}:
            base = stripped if stripped else raw_target
            for candidate in (f"House {base}", f"House of {base}"):
                if candidate in nodes_map and nodes_map[candidate].get("type") == "House":
                    return candidate, "house_prefix"

        return raw_target, "none"

    def segment_by_alias(
        self,
        *,
        raw_text: str,
        nodes_map: Dict[str, Dict[str, Any]],
        expected_target_types: List[str],
    ) -> List[str]:
        s = str(raw_text).strip()
        if not s:
            return []

        s_norm = self.norm(s)
        if len(s_norm) < 10:
            return []

        max_len = min(self.alias_max_len, 80)
        min_len = 5

        out: List[str] = []
        i = 0

        def accept(cand_id: str) -> bool:
            if not cand_id:
                return False
            if expected_target_types:
                t = nodes_map.get(cand_id, {}).get("type")
                if t not in expected_target_types:
                    return False
            if out and out[-1] == cand_id:
                return False
            return True

        while i < len(s_norm):
            found = False
            max_try = min(max_len, len(s_norm) - i)

            for L in range(max_try, min_len - 1, -1):
                sub = s_norm[i:i + L]

                if sub in self.alias_unique:
                    cand = self.alias_unique[sub]
                    if accept(cand):
                        out.append(cand)
                        i += L
                        found = True
                        break

                if sub in self.alias_multi:
                    cands = list(self.alias_multi[sub])
                    if expected_target_types:
                        cands = [cid for cid in cands if nodes_map.get(cid, {}).get("type") in expected_target_types]
                    if len(cands) == 1 and accept(cands[0]):
                        out.append(cands[0])
                        i += L
                        found = True
                        break

            if not found:
                i += 1

        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq
