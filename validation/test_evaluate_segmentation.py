"""Self-check of evaluate_segmentation.py on synthetic labels with known errors.

Run with ``python validation/test_evaluate_segmentation.py`` (or pytest).
"""

import sys
from pathlib import Path

import numpy as np
from skimage.segmentation import relabel_sequential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import (  # noqa: E402
    detection_metrics, match_pairs, overlap_table, resample_to_shape,
    split_merge_errors,
)


def cube_labels():
    """Six separated 10-voxel cubes in a 20 x 40 x 60 volume, labels 1-6."""
    gt = np.zeros((20, 40, 60), dtype=np.int64)
    label = 1
    for y in (5, 25):
        for x in (5, 25, 45):
            gt[5:15, y:y + 10, x:x + 10] = label
            label += 1
    return gt


def scores(gt, pred, threshold=0.5):
    # evaluate() relabels every image to 1..N after loading; do the same here.
    pred, _, _ = relabel_sequential(pred)
    gt_sizes, pred_sizes, pairs = overlap_table(gt, pred)
    matched = match_pairs(pairs, threshold)
    det = detection_metrics(matched, len(gt_sizes) - 1, len(pred_sizes) - 1)
    return det, split_merge_errors(pairs, gt_sizes, pred_sizes)


def test_identical_is_perfect():
    gt = cube_labels()
    det, sm = scores(gt, gt.copy())
    assert (det["tp"], det["fp"], det["fn"]) == (6, 0, 0)
    assert det["f1"] == 1.0 and det["mean_matched_iou"] == 1.0
    assert sm["split_gt_objects"] == 0 and sm["merged_predictions"] == 0


def test_permuted_label_ids_do_not_matter():
    gt = cube_labels()
    det, _ = scores(gt, np.array([0, 4, 6, 1, 2, 5, 3])[gt])
    assert det["tp"] == 6 and det["f1"] == 1.0


def test_split_merge_miss_and_spurious():
    gt = cube_labels()
    pred = gt.copy()
    pred[5:15, 5:15, 5:10] = 7            # object 1 split into halves (1 and 7)
    pred[pred == 3] = 2                   # objects 2 and 3 merged into one
    pred[pred == 6] = 0                   # object 6 missed
    pred[0:3, 0:3, 0:3] = 8               # spurious object in the background

    det, sm = scores(gt, pred)
    # Matches at IoU 0.5: objects 4 and 5 exact, object 1 to one half (IoU 0.5).
    # Merge has IoU 0.5 with object 2 or 3 (one match), the other is missed.
    assert det["tp"] == 4, det
    assert det["fn"] == 2 and det["fp"] == 2
    assert sm["split_gt_objects"] == 1 and sm["extra_split_pieces"] == 1
    assert sm["merged_predictions"] == 1 and sm["gt_objects_in_merges"] == 2
    assert sm["missed_gt_objects"] == 1
    assert sm["background_predictions"] == 1

    det_strict, _ = scores(gt, pred, threshold=0.75)
    assert det_strict["tp"] == 2  # only the two untouched objects


def test_resample_from_isotropic_grid():
    gt = cube_labels()
    # Pipeline-like prediction: Z upsampled 2.5x to an isotropic grid.
    iso = np.repeat(gt, 5, axis=0)[::2]
    assert iso.shape == (50, 40, 60)
    back = resample_to_shape(iso, gt.shape)
    assert back.shape == gt.shape
    det, _ = scores(gt, back)
    assert det["f1"] == 1.0 and det["mean_matched_iou"] > 0.95


def test_empty_prediction():
    gt = cube_labels()
    det, sm = scores(gt, np.zeros_like(gt))
    assert (det["tp"], det["fn"], det["fp"]) == (0, 6, 0)
    assert det["f1"] == 0.0 and sm["missed_gt_objects"] == 6


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"ok  {name}")
    print("All checks passed.")
