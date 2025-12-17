import json
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyntheticRegistry:
    path: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    ids: Set[str] = field(default_factory=set)

    def record(self, node_obj: Dict[str, Any]) -> bool:
        nid = node_obj.get("id")
        if not nid or nid in self.ids:
            return False
        self.ids.add(nid)
        self.nodes.append(node_obj)
        return True

    def persist(self) -> None:
        if not self.nodes:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for n in self.nodes:
                    f.write(json.dumps(n, ensure_ascii=False) + "\n")
            logger.info(f"🧩 Synthetic nodes saved to: {self.path} ({len(self.nodes)})")
        except Exception as e:
            logger.warning(f"⚠️ Could not save synthetic nodes: {e}")


def ensure_synthetic_house(
    *,
    raw_house: str,
    nodes_map: Dict[str, Dict[str, Any]],
    alias_add_fn,  # callback: (alias_raw, canonical_id) -> None
    registry: SyntheticRegistry,
) -> str:
    h = str(raw_house).strip()
    if not h:
        return h

    synthetic_id = h if h.lower().startswith("house ") else f"House {h}"
    if synthetic_id in nodes_map:
        alias_add_fn(h, synthetic_id)
        return synthetic_id

    node_obj = {
        "id": synthetic_id,
        "type": "House",
        "properties": {"Title": synthetic_id, "synthetic": True},
        "normalized_relations": {},
    }
    nodes_map[synthetic_id] = node_obj
    registry.record(node_obj)
    alias_add_fn(h, synthetic_id)
    return synthetic_id


def ensure_synthetic_seasons(
    *,
    node: Dict[str, Any],
    nodes_map: Dict[str, Dict[str, Any]],
    registry: SyntheticRegistry,
) -> None:
    props = node.get("properties", {}) or {}
    season_val = props.get("Season") or props.get("Appearances")
    if not season_val:
        return

    for s_num in re.findall(r"\d+", str(season_val)):
        if len(s_num) not in {1, 2}:
            continue
        sid = f"Season {s_num}"
        if sid in nodes_map:
            continue
        node_obj = {
            "id": sid,
            "type": "Lore",
            "properties": {"Title": sid, "synthetic": True, "kind": "Season"},
            "normalized_relations": {},
        }
        nodes_map[sid] = node_obj
        registry.record(node_obj)


def ensure_synthetic_episode(
    *,
    raw_episode: str,
    nodes_map: Dict[str, Dict[str, Any]],
    alias_add_fn,
    registry: SyntheticRegistry,
) -> str:
    e = str(raw_episode).strip()
    if not e:
        return e

    synthetic_id = e
    if synthetic_id in nodes_map:
        alias_add_fn(e, synthetic_id)
        return synthetic_id

    node_obj = {
        "id": synthetic_id,
        "type": "Episode",
        "properties": {"Title": synthetic_id, "synthetic": True},
        "normalized_relations": {},
    }
    nodes_map[synthetic_id] = node_obj
    registry.record(node_obj)
    alias_add_fn(e, synthetic_id)
    return synthetic_id
