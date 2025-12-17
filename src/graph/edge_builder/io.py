import json
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def choose_nodes_path(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "nodes_cleaned.jsonl"),
        os.path.join(data_dir, "nodes_validated.jsonl"),
        os.path.join(data_dir, "nodes.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # default (will fail later with clear log)


def load_nodes(nodes_path: str) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                n = json.loads(line)
                nid = n.get("id")
                if nid:
                    nodes[nid] = n
            except Exception:
                continue
    return nodes


def save_edges(edges_path: str, edges: List[Dict[str, Any]]) -> None:
    with open(edges_path, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
