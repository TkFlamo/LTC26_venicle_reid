from __future__ import annotations

CANONICAL_REID_COLUMNS = [
    "sample_id", "path", "vehicle_id", "camera_id", "dataset", "split", "image_id", "source_row",
]

# v0.6 deliberately keeps only classes that are directly supervised by Carparts-Seg.
# Active semantic ontology: only classes directly supervised by Carparts-Seg.
PART_CLASSES = [
    "wheel",
    "front_light",
    "rear_light",
    "front_glass",
    "rear_glass",
    "mirror",
    "door",
    "hood",
    "trunk_tailgate",
    "front_bumper",
    "rear_bumper",
]
PART_CLASS_TO_ID = {name: i for i, name in enumerate(PART_CLASSES)}
PART_SLOTS = list(PART_CLASSES)
PART_SLOT_IDS = list(range(len(PART_CLASSES)))
