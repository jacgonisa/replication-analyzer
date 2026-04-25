"""Shared constants for the CODEX weak-supervision pipeline."""

CLASS_ID_TO_NAME = {
    0: "background",
    1: "left_fork",
    2: "right_fork",
    3: "origin",
    4: "termination",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}

IGNORE_INDEX = -1

EVENT_CLASS_IDS = [1, 2, 3, 4]
