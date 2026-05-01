"""
utils/gluoncv_utils.py
======================
Parse PersonPath22 annotations and convert to COCO JSON for mAP evaluation.

ACTUAL PersonPath22 annotation layout (anno_amodal_2022 / anno_visible_2022)
-----------------------------------------------------------------------------
The annotations are NOT one big JSON file.
They are a FOLDER, one JSON file per video:

    annotation/
    └── anno_amodal_2022/
        ├── uid_vid_00000.json
        ├── uid_vid_00001.json
        ├── ...
        └── uid_vid_00236.json     (two IDs skipped: 00138, 00139)

Each per-video JSON has this structure:

    {
      "<frame_id_str>": {           # frame_id is a 1-based integer string
        "labels": [
          {
            "box": [x1, y1, x2, y2],   # pixel coords, xyxy format
            "label": "person",
            "attributes": { ... }      # optional extra fields
          },
          ...
        ]
      },
      ...
    }

Only annotated key-frames (~5 FPS) appear as keys -- most frames are absent.

COCO GT image_id encoding
--------------------------
    image_id = video_index * 1_000_000 + frame_id_1based

The same formula is used in run_detection.py, so prediction image_ids
match GT image_ids exactly without any remapping table.

config.py pointers
------------------
    ANNOTATION_PATH  should be the FOLDER, e.g.:
        .../annotation/anno_amodal_2022

    Set VIDEO_DIR to the raw_data/videos folder.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Debug: inspect one annotation file so you can confirm the schema
# ---------------------------------------------------------------------------

def inspect_annotation_file(anno_root: Path, uid: str = None) -> None:
    """
    Print the structure of one per-video annotation JSON.
    Call this once to verify field names before running the full pipeline.

    Parameters
    ----------
    anno_root : Path to the annotation folder (e.g. .../anno_amodal_2022)
    uid       : specific uid to inspect (e.g. "uid_vid_00000").
                If None, uses the first .json file found.
    """
    if uid is not None:
        candidate = anno_root / f"{uid}.json"
        if not candidate.exists():
            print(f"  [inspect] File not found: {candidate}")
            return
        path = candidate
    else:
        files = sorted(anno_root.glob("*.json"))
        if not files:
            print(f"  [inspect] No .json files found in {anno_root}")
            return
        path = files[0]

    print(f"\n  -- Annotation file inspection --")
    print(f"  File: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  type(data)        : {type(data)}")
    print(f"  len(data)         : {len(data)}  (number of annotated frames)")

    if not data:
        print("  (file is empty)")
        return

    # Show first frame
    first_fid = next(iter(data))
    first_frame = data[first_fid]
    print(f"  First frame_id    : {first_fid!r}")
    print(f"  Frame entry type  : {type(first_frame)}")

    if isinstance(first_frame, dict):
        print(f"  Frame entry keys  : {list(first_frame.keys())}")
        labels = first_frame.get("labels", [])
        print(f"  len(labels)       : {len(labels)}")
        if labels:
            first_label = labels[0]
            print(f"  First label keys  : {list(first_label.keys())}")
            print(f"  First label       : {first_label}")
    elif isinstance(first_frame, list):
        print(f"  Frame entry is list, len={len(first_frame)}")
        if first_frame:
            print(f"  First item        : {first_frame[0]}")

    print(f"  --------------------------------\n")


# ---------------------------------------------------------------------------
# Core loader: scan the annotation folder for all per-video JSONs
# ---------------------------------------------------------------------------

def load_anno_folder(anno_root: Path) -> Dict[str, Path]:
    """
    Return a mapping  { uid_stem: json_path }  for every .json in anno_root.

    uid_stem is the file stem, e.g. "uid_vid_00000" (without .json extension).
    """
    if not anno_root.exists():
        raise FileNotFoundError(
            f"Annotation folder not found: {anno_root}\n"
            f"Set ANNOTATION_PATH in config.py to the folder "
            f"(e.g. .../annotation/anno_amodal_2022), not to a .json file."
        )
    mapping = {p.stem: p for p in sorted(anno_root.glob("*.json"))}
    print(f"  Annotation folder: {anno_root}")
    print(f"  Found {len(mapping)} per-video JSON files")
    return mapping


# ---------------------------------------------------------------------------
# Per-video JSON parser
# ---------------------------------------------------------------------------

def parse_video_anno(json_path, debug_first_call=False):
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # NEW: iterate over entities list
    entities = data.get("entities", [])

    for entity in entities:

        # Skip reflections
        if entity.get("labels", {}).get("reflection") == 1:
            continue

        bbox = entity.get("bb", None)
        if bbox is None:
            continue

        x, y, w, h = bbox

        frame_idx = entity.get("blob", {}).get("frame_idx", None)
        if frame_idx is None:
            continue

        records.append({
            "frame_id": frame_idx,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "track_id": entity.get("id", -1),
            "confidence": entity.get("confidence", 1.0)
        })
    if debug_first_call:
        print(f"  Parsed {len(records)} valid annotations from {json_path}")

    return records

# ---------------------------------------------------------------------------
# Main converter: annotation folder -> COCO JSON
# ---------------------------------------------------------------------------

def convert_personpath_samples_to_coco(
    anno_root: Path,
    out_json: Path,
    eval_video_uids: Optional[List[str]] = None,
    person_category_id: int = 1,
) -> dict:
    """
    Convert PersonPath22 per-video annotation files to a single COCO GT JSON.

    Parameters
    ----------
    anno_root          : Path to the annotation folder (anno_amodal_2022/).
    out_json           : Where to save the output COCO JSON.
    eval_video_uids    : If given, only include these UIDs (e.g. test split UIDs).
                         If None, include all UIDs found in the folder.
    person_category_id : COCO category_id for person (1 by convention).

    Image ID encoding
    -----------------
    image_id = video_index * 1_000_000 + frame_id_1based

    This matches the encoding used in run_detection.py so prediction image_ids
    align with GT image_ids without any look-up table.

    Returns
    -------
    COCO dict with keys: 'images', 'annotations', 'categories'
    """
    coco: dict = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": person_category_id, "name": "person",
                         "supercategory": "person"}],
    }

    anno_map = load_anno_folder(anno_root)   # { uid_stem: json_path }

    # Decide which UIDs to process and in what order.
    # The ORDER must match run_detection.py (both use sorted uid_list).
    if eval_video_uids is not None:
        uid_list = sorted(eval_video_uids)
    else:
        uid_list = sorted(anno_map.keys())

    ann_id   = 0
    found    = 0
    skipped  = 0
    debug_done = False

    for video_idx, uid in enumerate(uid_list):
        json_path = anno_map.get(uid)
        if json_path is None:
            skipped += 1
            continue
        found += 1

        # Print full structure for the first video processed
        records = parse_video_anno(json_path, debug_first_call=not debug_done)
        debug_done = True

        # Group by frame_id to build one image entry per annotated frame
        frames_seen: Dict[int, int] = {}   # frame_id -> image_id

        for rec in records:
            fid      = rec["frame_id"]
            x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]

            if fid not in frames_seen:
                image_id = video_idx * 1_000_000 + fid
                frames_seen[fid] = image_id
                coco["images"].append({
                    "id":        image_id,
                    "file_name": f"{uid}/frame_{fid:06d}.jpg",
                    "video_uid": uid,
                    "frame_id":  fid,
                    # width/height: not available from per-video JSON alone;
                    # pycocotools doesn't strictly require them for bbox eval
                    "width":     0,
                    "height":    0,
                })

            ann_id += 1
            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    frames_seen[fid],
                "category_id": person_category_id,
                "bbox":        [x, y, w, h],
                "area":        w * h,
                "iscrowd":     0,
            })

    if skipped:
        print(f"  [WARNING] {skipped}/{len(uid_list)} UIDs had no annotation file")

    n_img = len(coco["images"])
    n_ann = len(coco["annotations"])
    print(f"\n  COCO GT summary:")
    print(f"    Videos processed  : {found}")
    print(f"    Annotated frames  : {n_img}")
    print(f"    GT annotations    : {n_ann}")

    if n_img == 0:
        print(
            "\n  [ERROR] 0 images produced. Check:\n"
            "    1. ANNOTATION_PATH points to the FOLDER anno_amodal_2022/\n"
            "       not to a .json file inside it.\n"
            "    2. The UIDs in eval_video_uids match the file stems in that folder.\n"
            "    3. Run inspect_annotation_file() to verify the per-video JSON schema."
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"  COCO JSON saved -> {out_json}")

    return coco


# ---------------------------------------------------------------------------
# Image-id look-up  (used in evaluate_map.py)
# ---------------------------------------------------------------------------

def build_image_id_map(coco: dict) -> Dict[Tuple[str, int], int]:
    """
    Return  {(video_uid, frame_id_1based): image_id}  for every GT image.
    Used to verify that prediction image_ids align with GT.
    """
    return {
        (img["video_uid"], img["frame_id"]): img["id"]
        for img in coco["images"]
    }


def build_uid_to_video_idx(uid_list: List[str]) -> Dict[str, int]:
    """
    Map uid string -> 0-based index in the sorted uid_list.
    Must match the indexing used in convert_personpath_samples_to_coco().
    Both this function and the converter use sorted(uid_list).
    """
    return {uid: idx for idx, uid in enumerate(sorted(uid_list))}


# ---------------------------------------------------------------------------
# Splits helper
# ---------------------------------------------------------------------------

def load_splits(splits_path: Path) -> dict:
    """Load splits.json -> {'train': [...], 'val': [...], 'test': [...]}."""
    with open(splits_path, "r") as f:
        return json.load(f)


def get_split_uids(splits_path: Path, split: str = "test") -> List[str]:
    """Return UIDs for the requested split. Returns [] if splits.json is missing."""
    if not splits_path.exists():
        print(f"  [WARNING] splits.json not found at {splits_path}. Using all videos.")
        return []
    splits = load_splits(splits_path)
    uids = splits.get(split, [])
    print(f"  Split '{split}': {len(uids)} videos")
    return uids


# ---------------------------------------------------------------------------
# MOTChallenge tracker-output writer  (used by run_tracking.py)
# ---------------------------------------------------------------------------

def write_mot_result(
    tracks_per_frame: List[Tuple[int, int, float, float, float, float, float]],
    out_path: Path,
) -> None:
    """
    Write tracks in MOTChallenge format for TrackEval compatibility.
    Format per line: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for (frame, tid, x1, y1, x2, y2, conf) in tracks_per_frame:
            w = x2 - x1
            h = y2 - y1
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{conf:.4f},-1,-1,-1\n")


# ---------------------------------------------------------------------------
# Legacy shim: keep old callers working
# ---------------------------------------------------------------------------

def load_gluoncv_anno(annotation_path: Path, debug: bool = True) -> dict:
    """
    Backward-compatible shim.

    If annotation_path is a FOLDER, delegates to load_anno_folder().
    If annotation_path is a FILE, raises a clear error explaining the new layout.
    """
    p = Path(annotation_path)
    if p.is_dir():
        return load_anno_folder(p)
    raise ValueError(
        f"load_gluoncv_anno() received a FILE path, but PersonPath22 annotations\n"
        f"are stored as a FOLDER of per-video JSONs.\n"
        f"Set ANNOTATION_PATH in config.py to the FOLDER:\n"
        f"    {p.parent / p.stem}   (without the .json extension)\n"
        f"e.g.:  ANNOTATION_PATH = DATASET_ROOT / 'annotation' / 'anno_amodal_2022'"
    )


def gluoncv_to_coco(
    db,   # ignored -- kept for signature compatibility
    video_uids: Optional[List[str]] = None,
    person_category_id: int = 1,
    anno_root: Optional[Path] = None,
) -> dict:
    """
    Legacy shim.  Prefer calling convert_personpath_samples_to_coco() directly.
    """
    if anno_root is None:
        raise ValueError(
            "gluoncv_to_coco() now requires anno_root=<Path to annotation folder>.\n"
            "Call convert_personpath_samples_to_coco() instead."
        )
    import tempfile, os
    tmp = Path(tempfile.mktemp(suffix=".json"))
    result = convert_personpath_samples_to_coco(
        anno_root          = anno_root,
        out_json           = tmp,
        eval_video_uids    = video_uids,
        person_category_id = person_category_id,
    )
    try:
        os.unlink(tmp)
    except Exception:
        pass
    return result


def save_coco(coco: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"  COCO JSON saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI: inspect + convert
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as cfg

    anno_root = cfg.ANNOTATION_PATH   # should be the FOLDER

    if not Path(anno_root).exists():
        sys.exit(f"[ERROR] ANNOTATION_PATH not found: {anno_root}")

    # Inspect first file
    files = sorted(Path(anno_root).glob("*.json"))
    if files:
        inspect_annotation_file(Path(anno_root), uid=files[0].stem)

    # Get split UIDs
    uids = get_split_uids(cfg.SPLITS_PATH, cfg.EVAL_SPLIT)
    if not uids:
        uids = None   # use all

    out_path = cfg.METRICS_DIR / f"gt_coco_{cfg.EVAL_SPLIT}.json"
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    convert_personpath_samples_to_coco(
        anno_root       = Path(anno_root),
        out_json        = out_path,
        eval_video_uids = uids,
    )
    print(f"\nDone. Run:  python evaluate_map.py")
