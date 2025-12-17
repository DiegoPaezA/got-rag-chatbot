import os
import re
import logging
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from .config import EdgeBuilderConfig, HONORIFICS, PLACEHOLDER, PLACEHOLDER_PATTERNS, SEGMENT_REL_TYPES, MANUAL_ALIASES
from .io import choose_nodes_path, load_nodes, save_edges
from .parsing import merge_rel_sources, parse_targets, is_valid_target
from .alias_index import AliasIndex
from .synthetic import (
    SyntheticRegistry,
    ensure_synthetic_house,
    ensure_synthetic_seasons,
    ensure_synthetic_episode,
)
from .audit import AuditCollector
from .stats import EdgeStats

logger = logging.getLogger(__name__)


class EdgeBuilder:
    """Build relationship edges from node properties with schema validation + robust target resolution (modular)."""

    def __init__(self, data_dir: str, config: EdgeBuilderConfig | None = None):
        self.data_dir = data_dir
        self.config = config or EdgeBuilderConfig()

        self.nodes_path = choose_nodes_path(data_dir)
        self.edges_path = os.path.join(data_dir, "edges.jsonl")
        self.synthetic_path = os.path.join(data_dir, "nodes_synthetic.jsonl")

        self.alias = AliasIndex(honorifics=HONORIFICS, placeholder_keys={self._norm_alias(x) for x in PLACEHOLDER})
        
    def _is_placeholder_target(self, s: str) -> bool:
            """Detecta si el target es basura/genérico usando Regex y Set."""
            if not s:
                return True
            raw = str(s).strip()
            if not raw:
                return True
            
            # 1. Chequeo rápido en Set (normalizado)
            key = self.alias.norm(raw)
            if key in PLACEHOLDER:
                return True
                
            # 2. Chequeo potente con Regex
            return any(p.match(raw) for p in PLACEHOLDER_PATTERNS)
    def run(self) -> None:
        if not os.path.exists(self.nodes_path):
            logger.error(f"❌ Nodes file not found at {self.nodes_path}")
            return

        logger.info(f"🔗 Building Edges from {self.nodes_path}...")
        nodes_map = load_nodes(self.nodes_path)

        # build alias maps from existing nodes
        self.alias.build(nodes_map)
        count_manual = 0
        for alias_val, canonical_id in MANUAL_ALIASES.items():
            # Validación de seguridad: solo agregar si el ID canónico existe
            if canonical_id in nodes_map:
                self.alias.add_alias_safely(alias_val, canonical_id)
                count_manual += 1
        
        if count_manual > 0:
            logger.info(f"   🔧 Injected {count_manual} manual aliases from config.")
        # ---------------------------------------------------------
        edges_set: Set[Tuple[str, str, str]] = set()
        stats = EdgeStats()
        audit = AuditCollector()
        registry = SyntheticRegistry(path=self.synthetic_path)

        # snapshot iteration (avoid dict-size change)
        for node_id, node in tqdm(list(nodes_map.items()), desc="Processing Nodes"):
            self._process_node(
                node_id=node_id,
                node=node,
                nodes_map=nodes_map,
                edges_set=edges_set,
                stats=stats,
                audit=audit,
                registry=registry,
            )

        edges_list = [{"source": s, "relation": r, "target": t} for (s, r, t) in edges_set]
        save_edges(self.edges_path, edges_list)

        # persist synthetic nodes (if any)
        registry.persist()
        stats.synthetic_nodes_created = len(registry.nodes)

        self._log_stats(edges_list, stats)
        audit.log_summary(total_missing=stats.skipped_missing_target)
        audit.save(self.data_dir, total_missing=stats.skipped_missing_target)

        logger.info(f"✅ Edge building finished. Output: {self.edges_path}")

    # --------------------------
    # Core processing
    # --------------------------

    def _process_node(
        self,
        *,
        node_id: str,
        node: Dict[str, Any],
        nodes_map: Dict[str, Dict[str, Any]],
        edges_set: Set[Tuple[str, str, str]],
        stats: EdgeStats,
        audit: AuditCollector,
        registry: SyntheticRegistry,
    ) -> None:
        node_type = node.get("type", "Lore")

        raw_props = node.get("properties", {}) or {}
        norm_rel = node.get("normalized_relations", {}) or {}

        rel_source = merge_rel_sources(raw_props, norm_rel, self.config.rel_map)

        for rel_key, val in rel_source.items():
            self._process_relation_field(
                source_id=node_id,
                source_type=node_type,
                rel_key=rel_key,
                val=val,
                nodes_map=nodes_map,
                edges_set=edges_set,
                stats=stats,
                audit=audit,
                registry=registry,
            )

        # Seasons: ensure synthetic nodes exist (if enabled) and then add edges
        if node_type == "Character":
            if self.config.enable_synthetic_seasons:
                ensure_synthetic_seasons(node=node, nodes_map=nodes_map, registry=registry)
            self._add_season_edges(node, edges_set, nodes_map, stats)

    def _process_relation_field(
        self,
        *,
        source_id: str,
        source_type: str,
        rel_key: str,
        val: Any,
        nodes_map: Dict[str, Dict[str, Any]],
        edges_set: Set[Tuple[str, str, str]],
        stats: EdgeStats,
        audit: AuditCollector,
        registry: SyntheticRegistry,
    ) -> None:
        rel_type = self.config.rel_map.get(rel_key)
        if not rel_type:
            return

        allowed_sources = self.config.schema_constraints.get(rel_type)
        if allowed_sources and source_type not in allowed_sources:
            stats.skipped_schema += 1
            return

        expected_target_types = self.config.target_constraints.get(rel_type, [])
        targets = parse_targets(val, rel_type=rel_type)
        raw_val_str = None if isinstance(val, list) else str(val)

        unresolved: List[Tuple[str, str]] = []

        # Pass 1 (normal resolution)
        for target in targets:
            if self._is_placeholder_target(target):
                    stats.skipped_placeholder += 1
                    continue
            resolved, reason = self.alias.resolve(
                raw_target=target,
                rel_type=rel_type,
                nodes_map=nodes_map,
                expected_target_types=expected_target_types,
            )
            self._bump_resolution_stats(stats, reason)

            if self.alias.is_placeholder(resolved):
                continue

            resolved = self._maybe_create_synthetic_target(
                rel_type=rel_type,
                expected_target_types=expected_target_types,
                resolved_target=resolved,
                nodes_map=nodes_map,
                registry=registry,
            )

            if resolved not in nodes_map and rel_type not in self.config.allow_missing_target_for:
                unresolved.append((target, resolved))
                continue

            if is_valid_target(resolved, source_id):
                self._create_edge_and_inverse(source_id, rel_type, resolved, edges_set, stats)

        # Pass 2 (segmentation once per field)
        if (
            self.config.enable_segmentation_fallback
            and unresolved
            and raw_val_str
            and rel_type in SEGMENT_REL_TYPES
        ):
            seg = self.alias.segment_by_alias(
                raw_text=raw_val_str,
                nodes_map=nodes_map,
                expected_target_types=expected_target_types,
            )
            if len(seg) >= 2:
                stats.segmented_by_alias += 1
                self._process_segmented_targets(
                    source_id=source_id,
                    source_type=source_type,
                    rel_key=rel_key,
                    rel_type=rel_type,
                    seg=seg,
                    nodes_map=nodes_map,
                    edges_set=edges_set,
                    stats=stats,
                    audit=audit,
                    registry=registry,
                    expected_target_types=expected_target_types,
                )
                return  # consider handled

        # Audit remaining unresolved
        for raw_t, resolved_t in unresolved:
            stats.skipped_missing_target += 1
            audit.record_missing(
                rel_type=rel_type,
                rel_key=rel_key,
                source_id=source_id,
                source_type=source_type,
                raw_target=raw_t,
                resolved_target=resolved_t,
            )

    def _process_segmented_targets(
        self,
        *,
        source_id: str,
        source_type: str,
        rel_key: str,
        rel_type: str,
        seg: List[str],
        nodes_map: Dict[str, Dict[str, Any]],
        edges_set: Set[Tuple[str, str, str]],
        stats: EdgeStats,
        audit: AuditCollector,
        registry: SyntheticRegistry,
        expected_target_types: List[str],
    ) -> None:
        for seg_target in seg:
            resolved, reason = self.alias.resolve(
                raw_target=seg_target,
                rel_type=rel_type,
                nodes_map=nodes_map,
                expected_target_types=expected_target_types,
            )
            self._bump_resolution_stats(stats, reason)

            if self.alias.is_placeholder(resolved):
                continue

            resolved = self._maybe_create_synthetic_target(
                rel_type=rel_type,
                expected_target_types=expected_target_types,
                resolved_target=resolved,
                nodes_map=nodes_map,
                registry=registry,
            )

            if resolved not in nodes_map and rel_type not in self.config.allow_missing_target_for:
                stats.skipped_missing_target += 1
                audit.record_missing(
                    rel_type=rel_type,
                    rel_key=rel_key,
                    source_id=source_id,
                    source_type=source_type,
                    raw_target=seg_target,
                    resolved_target=resolved,
                    note="segmented_fallback_failed",
                )
                continue

            if is_valid_target(resolved, source_id):
                self._create_edge_and_inverse(source_id, rel_type, resolved, edges_set, stats)

    # --------------------------
    # Synthetic policies
    # --------------------------

    def _maybe_create_synthetic_target(
        self,
        *,
        rel_type: str,
        expected_target_types: List[str],
        resolved_target: str,
        nodes_map: Dict[str, Dict[str, Any]],
        registry: SyntheticRegistry,
    ) -> str:
        # BELONGS_TO -> House synthetic
        if (
            self.config.enable_synthetic_house_for_belongs_to
            and rel_type == "BELONGS_TO"
            and "House" in expected_target_types
            and resolved_target not in nodes_map
        ):
            return ensure_synthetic_house(
                raw_house=resolved_target,
                nodes_map=nodes_map,
                alias_add_fn=self.alias.add_alias_safely,
                registry=registry,
            )

        # DIED_IN_EPISODE -> Episode synthetic (optional)
        if (
            self.config.enable_synthetic_episode_for_died_in_episode
            and rel_type == "DIED_IN_EPISODE"
            and resolved_target not in nodes_map
        ):
            return ensure_synthetic_episode(
                raw_episode=resolved_target,
                nodes_map=nodes_map,
                alias_add_fn=self.alias.add_alias_safely,
                registry=registry,
            )

        return resolved_target

    # --------------------------
    # Edge creation
    # --------------------------

    def _create_edge_and_inverse(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        edges_set: Set[Tuple[str, str, str]],
        stats: EdgeStats,
    ) -> None:
        direct = (source_id, rel_type, target_id)
        if direct not in edges_set:
            edges_set.add(direct)
            stats.created += 1

        inv = self.config.inverse_map.get(rel_type)
        if inv:
            back = (target_id, inv, source_id)
            if back not in edges_set:
                edges_set.add(back)
                stats.inverse_created += 1
        elif rel_type in {"MARRIED_TO", "SIBLING_OF", "LOVER_OF"}:
            sym = (target_id, rel_type, source_id)
            if sym not in edges_set:
                edges_set.add(sym)
                stats.inverse_created += 1

    def _add_season_edges(
        self,
        node: Dict[str, Any],
        edges_set: Set[Tuple[str, str, str]],
        nodes_map: Dict[str, Dict[str, Any]],
        stats: EdgeStats,
    ) -> None:
        props = node.get("properties", {}) or {}
        season_val = props.get("Season") or props.get("Appearances")
        if not season_val:
            return

        for s_num in re.findall(r"\d+", str(season_val)):
            if len(s_num) in {1, 2}:
                target = f"Season {s_num}"
                if target not in nodes_map and "APPEARED_IN_SEASON" not in self.config.allow_missing_target_for:
                    stats.skipped_missing_target += 1
                    continue
                edges_set.add((node["id"], "APPEARED_IN_SEASON", target))

    # --------------------------
    # Logging / helpers
    # --------------------------

    def _bump_resolution_stats(self, stats: EdgeStats, reason: str) -> None:
        if reason == "alias":
            stats.targets_resolved_by_alias += 1
        elif reason == "house_prefix":
            stats.targets_resolved_by_house_prefix += 1
        elif reason == "ambiguous_unresolved":
            stats.ambiguous_alias_unresolved += 1

    def _log_stats(self, edges_list: List[Dict[str, Any]], stats: EdgeStats) -> None:
        logger.info("📊 Edge Creation Statistics:")
        logger.info(f"   Total Unique Edges Created: {len(edges_list)}")
        logger.info(f"   Relationships Skipped by Schema: {stats.skipped_schema}")
        logger.info(f"   Missing-Target Edges Skipped: {stats.skipped_missing_target}")
        logger.info(f"   Inferred/Symmetric Edges Created: {stats.inverse_created}")
        logger.info(f"   Targets Resolved by Alias: {stats.targets_resolved_by_alias}")
        logger.info(f"   Targets Resolved by House Prefix: {stats.targets_resolved_by_house_prefix}")
        logger.info(f"   Ambiguous Alias Unresolved: {stats.ambiguous_alias_unresolved}")
        logger.info(f"   Values Segmented by Alias Fallback: {stats.segmented_by_alias}")
        logger.info(f"   Synthetic Nodes Created: {stats.synthetic_nodes_created}")

    def _norm_alias(self, s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())
