"""
evaluate_map.py
===============
Compute mAP50-95, mAP50, and mAP75 for each YOLOv11 variant using pycocotools.

Pipeline
--------
1. Scan the annotation folder (anno_amodal_2022/) for per-video JSON files.
2. Convert GT to a single COCO JSON, cached at outputs/metrics/gt_coco_test.json.
3. Load per-video prediction JSONs written by run_detection.py.
4. Match predictions to GT by image_id = video_idx * 1_000_000 + frame_id_1based.
5. Run COCOeval; save mAP metrics to outputs/metrics/map_results.csv.

Prediction / GT alignment
--------------------------
Both run_detection.py and the GT converter use:
    uid_list = sorted(uids_in_split)
    image_id = video_idx * 1_000_000 + frame_id_1based

So image_ids match without any look-up table.
Predictions on non-annotated frames (no GT entry) are silently ignored by
pycocotools -- this is correct, since PersonPath22 only annotates ~5 FPS.

Usage
-----
    python evaluate_map.py

If you get "0 images, 0 annotations":
    1. Confirm ANNOTATION_PATH in config.py points to the FOLDER
       (e.g. .../annotation/anno_amodal_2022), not a .json file.
    2. Delete the stale cache:  del outputs\\metrics\\gt_coco_test.json
    3. Rerun.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from utils.gluoncv_utils import (
    build_image_id_map,
    build_uid_to_video_idx,
    convert_personpath_samples_to_coco,
    get_split_uids,
    inspect_annotation_file,
)
from utils.yolo_utils import model_variant_name


# ---------------------------------------------------------------------------
# Ground-truth preparation
# ---------------------------------------------------------------------------

def get_coco_gt(uid_list: list) -> tuple:
    """
    Build (or load cached) COCO GT JSON for the uid_list.

    Automatically rebuilds if the cached file has 0 images.

    Returns (gt_path, image_id_map)
        image_id_map: {(video_uid, frame_id_1based) -> image_id}
    """
    gt_path  = cfg.METRICS_DIR / f"gt_coco_{cfg.EVAL_SPLIT}.json"
    anno_root = Path(cfg.ANNOTATION_PATH)

    # Validate ANNOTATION_PATH is a folder
    if anno_root.suffix == ".json" or (anno_root.exists() and anno_root.is_file()):
        sys.exit(
            f"[ERROR] ANNOTATION_PATH must be a FOLDER, not a file.\n"
            f"  Current value: {anno_root}\n"
            f"  Fix in config.py:\n"
            f"    ANNOTATION_PATH = DATASET_ROOT / 'annotation' / 'anno_amodal_2022'"
        )
    if not anno_root.exists():
        sys.exit(
            f"[ERROR] Annotation folder not found: {anno_root}\n"
            f"  Set ANNOTATION_PATH in config.py to the folder containing "
            f"per-video .json files."
        )

    rebuild = not gt_path.exists()
    if not rebuild:
        try:
            with open(gt_path) as f:
                cached = json.load(f)
            n_cached = len(cached.get("images", []))
            if n_cached == 0:
                print(f"  Cached GT has 0 images -- rebuilding.")
                rebuild = True
            else:
                print(f"  Using cached GT: {gt_path}  ({n_cached} images)")
        except Exception:
            rebuild = True

    if rebuild:
        print("  Converting annotation folder -> COCO JSON ...")
        coco = convert_personpath_samples_to_coco(
            anno_root       = anno_root,
            out_json        = gt_path,
            eval_video_uids = uid_list,
        )
    else:
        with open(gt_path) as f:
            coco = json.load(f)

    n_img = len(coco.get("images", []))
    n_ann = len(coco.get("annotations", []))
    print(f"  GT: {n_img} annotated frames, {n_ann} person annotations")

    if n_img == 0:
        print("\n  [HINT] Run this to inspect the annotation file structure:")
        print(f"    python utils/gluoncv_utils.py")
        sys.exit("[ERROR] GT has 0 images. See hints above.")

    img_id_map = build_image_id_map(coco)
    return gt_path, img_id_map


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions(
    model_name: str,
    uid_list: list,
    img_id_map: dict,
) -> list:
    """
    Load per-video prediction JSONs written by run_detection.py and filter
    to only frames that have a GT entry.

    uid_list must be sorted -- same ordering as during detection so that
    video_idx * 1_000_000 + frame_id gives the same image_id values.
    """
    det_dir = cfg.DETECTION_DIR / model_name
    if not det_dir.exists():
        print(f"  [WARNING] No detections found for {model_name} at {det_dir}")
        print(f"  Run:  python run_detection.py --models {model_name}.pt")
        return []

    valid_ids  = set(img_id_map.values())
    all_preds  = []
    files_read = 0

    for uid in uid_list:
        json_path = det_dir / f"{uid}_preds.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            preds = json.load(f)
        files_read += 1
        for pred in preds:
            if pred["image_id"] in valid_ids:
                all_preds.append(pred)

    print(f"  {model_name}: {files_read} files -> "
          f"{len(all_preds)} predictions match GT frames")
    return all_preds


# ---------------------------------------------------------------------------
# mAP evaluation
# ---------------------------------------------------------------------------

def evaluate_map(gt_path: Path, predictions: list, model_name: str) -> dict:
    """
    Run COCOeval and return a metrics dict.

    COCOeval.stats layout:
        0: AP @ IoU 0.50:0.95  (mAP50-95)
        1: AP @ IoU 0.50       (mAP50)
        2: AP @ IoU 0.75       (mAP75)
        3: AP small / 4: medium / 5: large
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        sys.exit("[ERROR] pip install pycocotools")

    if not predictions:
        print(f"  [WARNING] No predictions for {model_name} -- skipping eval.")
        return _empty_metrics(model_name)

    print(f"\n  COCOeval for {model_name}  ({len(predictions)} predictions) ...")
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(predictions)
    ev      = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = ev.stats
    return {
        "model_name": model_name,
        "mAP50_95":   round(float(s[0]), 4),
        "mAP50":      round(float(s[1]), 4),
        "mAP75":      round(float(s[2]), 4),
        "mAP_small":  round(float(s[3]), 4),
        "mAP_medium": round(float(s[4]), 4),
        "mAP_large":  round(float(s[5]), 4),
    }


def _empty_metrics(model_name: str) -> dict:
    return {"model_name": model_name,
            "mAP50_95": None, "mAP50": None, "mAP75": None,
            "mAP_small": None, "mAP_medium": None, "mAP_large": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("YOLOv11 mAP Evaluation -- PersonPath22")
    print("=" * 65)

    # ---- determine uid_list (same order as run_detection.py) ----------------
    split_uids = get_split_uids(cfg.SPLITS_PATH, cfg.EVAL_SPLIT)

    if not split_uids:
        # Fall back: derive UIDs from detection output
        for weights in cfg.MODELS:
            det_dir = cfg.DETECTION_DIR / model_variant_name(weights)
            if det_dir.exists():
                split_uids = sorted(
                    p.stem.replace("_preds", "")
                    for p in det_dir.glob("*_preds.json")
                )
                print(f"  No splits.json -- using {len(split_uids)} UIDs from "
                      f"detection output ({model_variant_name(weights)})")
                break

    if not split_uids:
        sys.exit(
            "[ERROR] No UIDs found.\n"
            "  Run run_detection.py first, OR set SPLITS_PATH in config.py."
        )

    uid_list = sorted(set(split_uids))   # sorted = same order as convertor
    print(f"\n  Evaluating {len(uid_list)} videos (split='{cfg.EVAL_SPLIT}')")

    # ---- build / load GT ----------------------------------------------------
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    gt_path, img_id_map = get_coco_gt(uid_list)

    # ---- evaluate each model ------------------------------------------------
    results = []
    for weights in cfg.MODELS:
        model_name = model_variant_name(weights)
        print(f"\n{'--'*32}\nModel: {model_name}\n{'--'*32}")

        preds   = collect_predictions(model_name, uid_list, img_id_map)
        metrics = evaluate_map(gt_path, preds, model_name)
        results.append(metrics)

        print(f"\n  mAP50-95={metrics['mAP50_95']}  "
              f"mAP50={metrics['mAP50']}  "
              f"mAP75={metrics['mAP75']}")

    # ---- save CSV -----------------------------------------------------------
    out_path = cfg.METRICS_DIR / "map_results.csv"
    fields   = ["model_name", "mAP50_95", "mAP50", "mAP75",
                "mAP_small", "mAP_medium", "mAP_large"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nmAP results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
