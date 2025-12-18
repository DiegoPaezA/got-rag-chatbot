import re
from dataclasses import dataclass, field
from typing import Dict, List, Set


# Common titles/honorifics that cause mismatches: "Lord Cregan Stark" vs "Cregan Stark"
HONORIFICS: Set[str] = {
    "lord", "lady", "ser", "king", "queen", "prince", "princess",
    "maester", "septa", "sept", "captain", "commander", "warden",
    "hand", "grand", "master", "protector",
}

# Placeholder targets
PLACEHOLDER: Set[str] = {"unknown", "none", "n/a", "na", "?", "unnamed"}

PLACEHOLDER_PATTERNS = [
    re.compile(r"^(unknown|none|n/?a|\?)$", re.I),
    re.compile(r"^(wife|wives|husband|spouse|mistress|paramour)$", re.I),
    re.compile(r"^(brother\(s\)|sister\(s\)|sibling\(s\))$", re.I),
    re.compile(r"^all of (his|her) (daughters|sons|children|offspring)$", re.I),
    re.compile(r"^\d+\s+(salt\s+wives|wives|sons|daughters|children|bastards)$", re.I), # e.g., "3 sons"
    re.compile(r"^(andals|the rhoynar|the first men)$", re.I),
    re.compile(r"^unidentified\s.*", re.I), # Ej: "Unidentified Stark"
]
# Relations where concatenated values are very common (no clear delimiters)
SEGMENT_REL_TYPES: Set[str] = {
    "CHILD_OF", "PARENT_OF", "SIBLING_OF", "MARRIED_TO", "LOVER_OF",
    "BELONGS_TO", "SWORN_TO", "AFFILIATED_WITH", "VASSAL_OF", "OVERLORD_OF",
    "LED_BY",
}

MANUAL_ALIASES: Dict[str, str] = {
    "Catelyn Tully": "Catelyn Stark",
    "Lysa Tully": "Lysa Arryn",
    "Cersei Lannister": "Cersei Baratheon",
    "Sansa Bolton": "Sansa Stark",
    "Jeyne Westerling": "Jeyne Stark",
    "Littlefinger": "Petyr Baelish",
    "The Kingslayer": "Jaime Lannister",
    "The Imp": "Tyrion Lannister",
    "Khaleesi": "Daenerys Targaryen",
    "The Spider": "Varys",
    "Lord Snow": "Jon Snow"
}

INVERSE_MAP = {
    "FATHER": "FATHER_OF",
    "MOTHER": "MOTHER_OF",
    "CHILD_OF": "PARENT_OF",
    "PARENT_OF": "CHILD_OF",
    "SUCCEEDED_BY": "PRECEDED_BY",
    "PRECEDED_BY": "SUCCEEDED_BY",
    "OWNS_WEAPON": "OWNED_BY",
    "OWNED_BY": "OWNS_WEAPON",
    "VASSAL_OF": "OVERLORD_OF",
    "OVERLORD_OF": "VASSAL_OF",
    "SWORN_TO": "HAS_MEMBER",
}

SCHEMA_CONSTRAINTS = {
    # Family and Personal
    "FATHER": ["Character"],
    "MOTHER": ["Character"],
    "CHILD_OF": ["Character", "Creature"],
    "PARENT_OF": ["Character", "Creature"],
    "SIBLING_OF": ["Character", "Creature"],
    "MARRIED_TO": ["Character"],
    "LOVER_OF": ["Character"],

    # Loyalty and Politics
    "BELONGS_TO": ["Character", "Creature", "House"],
    "SWORN_TO": ["House", "Character", "Organization"],
    "AFFILIATED_WITH": ["Character", "Organization", "House"],
    "VASSAL_OF": ["House"],
    "OVERLORD_OF": ["House"],
    "HAS_MEMBER": ["House", "Organization"],
    "SUCCEEDED_BY": ["Character", "House"],
    "PRECEDED_BY": ["Character", "House"],
    "LED_BY": ["Organization", "House", "Battle", "Army"],

    # Geography
    "LOCATED_IN": ["Location", "Battle", "House", "City", "Castle", "Organization", "Event"],
    "SEATED_AT": ["House"],

    # Culture and Religion
    "HAS_CULTURE": ["Character", "House", "Location"],
    "FOLLOWS_RELIGION": ["Character", "Organization", "House"],

    # War
    "PARTICIPANT_IN": ["Character", "House", "Organization", "Creature"],
    "COMMANDED_BY": ["Battle", "Army"],
    "PART_OF_CONFLICT": ["Battle"],
    "PART_OF_WAR": ["Battle"],

    # Objects
    "CREATED_BY": ["Object"],
    "OWNED_BY": ["Object", "Creature"],
    "WIELDED_BY": ["Object"],
    "OWNS_WEAPON": ["Character", "House"],
    "HAS_ARMS": ["Character", "House"],

    # Meta / Production
    "PLAYED_BY": ["Character"],
    # "DIED_IN_EPISODE": ["Character", "Creature"],
    "APPEARED_IN_SEASON": ["Character", "Creature"],
}

TARGET_CONSTRAINTS = {
    "FATHER": ["Character"],
    "MOTHER": ["Character"],
    "CHILD_OF": ["Character", "Creature"],
    "PARENT_OF": ["Character", "Creature"],
    "SIBLING_OF": ["Character", "Creature"],
    "MARRIED_TO": ["Character"],
    "LOVER_OF": ["Character"],

    "BELONGS_TO": ["House", "Organization"],
    "SWORN_TO": ["House", "Organization"],
    "HAS_MEMBER": ["Character", "House", "Organization"],
    "AFFILIATED_WITH": ["House", "Organization"],
    "VASSAL_OF": ["House"],
    "OVERLORD_OF": ["House"],

    "SEATED_AT": ["Location", "Castle"],
    "LOCATED_IN": ["Location", "Region", "City", "Castle"],
    "FOLLOWS_RELIGION": ["Religion"],

    "OWNS_WEAPON": ["Object"],
    "OWNED_BY": ["Character", "House", "Creature"],
    "WIELDED_BY": ["Character", "Creature"],
    "CREATED_BY": ["Character", "Organization", "Lore"],

    # "DIED_IN_EPISODE": ["Episode", "Lore"],
    "APPEARED_IN_SEASON": ["Episode", "Lore"],
}

# Property-to-relationship mapping (with Culture/Arms/Actor disabled as edges)
REL_MAP = {
    "Father": "FATHER",
    "Mother": "MOTHER",
    "Issue": "PARENT_OF",
    "Children": "PARENT_OF",

    "Spouse": "MARRIED_TO",
    "Siblings": "SIBLING_OF",
    "Lovers": "LOVER_OF",

    "House": "BELONGS_TO",
    "Allegiance": "SWORN_TO",
    "Affiliation": "AFFILIATED_WITH",
    "Overlords": "VASSAL_OF",
    "Vassals": "OVERLORD_OF",

    "Successor": "SUCCEEDED_BY",
    "Predecessor": "PRECEDED_BY",
    "Heir": "SUCCEEDED_BY",

    "Leader": "LED_BY",
    "Head": "LED_BY",
    "Rulers": "LED_BY",

    "Region": "LOCATED_IN",
    "Seat": "SEATED_AT",

    # "Culture": "HAS_CULTURE",
    "Religion": "FOLLOWS_RELIGION",

    "Combatants": "PARTICIPANT_IN",
    "Commanders": "COMMANDED_BY",
    "Conflict": "PART_OF_CONFLICT",
    "War": "PART_OF_WAR",

    "Creator": "CREATED_BY",
    "Owners": "OWNED_BY",
    "Wielder": "WIELDED_BY",

    "Weapon": "OWNS_WEAPON",
    "Ancestral Weapon": "OWNS_WEAPON",
    # "Arms": "HAS_ARMS",
    # "Actor": "PLAYED_BY",

    #"DeathEp": "DIED_IN_EPISODE",
}


@dataclass(frozen=True)
class EdgeBuilderConfig:
    allow_missing_target_for: Set[str] = field(default_factory=lambda: {"APPEARED_IN_SEASON"})

    # Reduce missing targets
    enable_segmentation_fallback: bool = True

    # Synthetic nodes to improve MATCH in loader
    enable_synthetic_house_for_belongs_to: bool = True
    enable_synthetic_seasons: bool = True
    enable_synthetic_episode_for_died_in_episode: bool = False  # keep False by default

    # Maps
    inverse_map: Dict[str, str] = field(default_factory=lambda: dict(INVERSE_MAP))
    schema_constraints: Dict[str, List[str]] = field(default_factory=lambda: dict(SCHEMA_CONSTRAINTS))
    target_constraints: Dict[str, List[str]] = field(default_factory=lambda: dict(TARGET_CONSTRAINTS))
    rel_map: Dict[str, str] = field(default_factory=lambda: dict(REL_MAP))
