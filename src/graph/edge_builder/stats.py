from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class EdgeStats:
    created: int = 0
    skipped_schema: int = 0
    inverse_created: int = 0
    skipped_missing_target: int = 0
    skipped_placeholder: int = 0
    manual_alias_used: int = 0

    targets_resolved_by_alias: int = 0
    targets_resolved_by_house_prefix: int = 0
    ambiguous_alias_unresolved: int = 0

    segmented_by_alias: int = 0
    synthetic_nodes_created: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dict(asdict(self))
