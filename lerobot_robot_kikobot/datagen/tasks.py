"""The tasks: one kind of item, picked off the desk and put where it belongs.

Each task is its own dataset (`kikobot_desk_<task>` once converted), with one
language instruction. The strategy says how the expert holds and places it:

    cup_insert  pen-like items: grasped near one end, turned upright over the
                cup and lowered tip first into it.
    ring_drop   the tape roll: carried flat over the peg and dropped round the post.
    tray_place  everything else: carried flat, turned to lie along its tray
                (or tray compartment) and lowered into it.

Sticky-note pads are left out: at 76 mm across they are wider than the open
jaws (53 mm), so no grasp holds one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..desk import layout as L


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    station: str
    strategy: str
    instruction: str
    # supply_tray only: which of its compartments (0 is the one at -X of the tray).
    compartment: int | None = None

    @property
    def station_spec(self) -> L.Station:
        return next(s for s in L.STATIONS if s.name == self.station)


_STATION_WORDS = {
    "pen_cup": "the pen cup",
    "marker_cup": "the marker cup",
    "tape_peg": "the tape holder",
    "supply_tray": "the supply tray",
    "long_tray": "the long tray",
}

_ITEM_WORDS = {
    "glue_stick": "glue stick",
    "tape_roll": "tape roll",
    "paper_clip_box": "box of paper clips",
    "sharpener": "pencil sharpener",
}

# supply_tray has three compartments for four kinds; the two small ones share.
_COMPARTMENTS = {"glue_stick": 0, "eraser": 1, "sharpener": 2, "paper_clip_box": 2}

_STRATEGY = {
    "pen": "cup_insert",
    "pencil": "cup_insert",
    "marker": "cup_insert",
    "highlighter": "cup_insert",
    "tape_roll": "ring_drop",
}

EXCLUDED_KINDS = ("sticky_notes",)


def _make(kind: str) -> TaskSpec:
    station = L.station_for(kind).name
    item = _ITEM_WORDS.get(kind, kind)
    verb = "put" if station != "tape_peg" else "hang"
    where = "in" if station != "tape_peg" else "on"
    comp = _COMPARTMENTS.get(kind) if station == "supply_tray" else None
    return TaskSpec(
        name=f"{kind}_to_{station}",
        kind=kind,
        station=station,
        strategy=_STRATEGY.get(kind, "tray_place"),
        instruction=f"{verb} the {item} {where} {_STATION_WORDS[station]}",
        compartment=comp,
    )


TASKS: dict[str, TaskSpec] = {
    t.name: t for t in (_make(k) for k in L.KINDS if k not in EXCLUDED_KINDS)
}
TASK_NAMES = tuple(TASKS)


def get_task(name: str) -> TaskSpec:
    if name in TASKS:
        return TASKS[name]
    # Also accept the bare kind: "pen" -> pen_to_pen_cup.
    for t in TASKS.values():
        if t.kind == name:
            return t
    raise KeyError(f"Unknown task {name!r}. Tasks: {', '.join(TASK_NAMES)}")
