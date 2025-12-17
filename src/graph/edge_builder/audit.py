import json
import os
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditCollector:
    max_examples_per_rel: int = 10
    missing_by_rel: Counter = field(default_factory=Counter)
    missing_by_key: Counter = field(default_factory=Counter)
    missing_examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def record_missing(
        self,
        *,
        rel_type: str,
        rel_key: str,
        source_id: str,
        source_type: str,
        raw_target: str,
        resolved_target: str,
        note: Optional[str] = None,
    ) -> None:
        self.missing_by_rel[rel_type] += 1
        self.missing_by_key[rel_key] += 1

        ex_list = self.missing_examples[rel_type]
        if len(ex_list) < self.max_examples_per_rel:
            ex = {
                "source": source_id,
                "source_type": source_type,
                "rel_key": rel_key,
                "rel_type": rel_type,
                "raw_target": raw_target,
                "resolved_target": resolved_target,
            }
            if note:
                ex["note"] = note
            ex_list.append(ex)

    def log_summary(self, total_missing: int) -> None:
        if total_missing <= 0:
            return

        logger.info("📉 Missing-targets by relationship type (top 20):")
        for rel, cnt in self.missing_by_rel.most_common(20):
            logger.info(f"   - {rel}: {cnt} ({(cnt/total_missing)*100:.1f}%)")

        logger.info("📉 Missing-targets by property key (top 20):")
        for k, cnt in self.missing_by_key.most_common(20):
            logger.info(f"   - {k}: {cnt} ({(cnt/total_missing)*100:.1f}%)")

    def save(self, data_dir: str, total_missing: int) -> None:
        if total_missing <= 0:
            return

        audit_path = os.path.join(data_dir, "missing_targets_audit.json")
        try:
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_missing": total_missing,
                        "missing_by_rel": dict(self.missing_by_rel),
                        "missing_by_key": dict(self.missing_by_key),
                        "examples": {k: v for k, v in self.missing_examples.items()},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"📝 Missing-target audit saved to: {audit_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save missing-target audit: {e}")
