"""Evaluate NucleoSeg3D nuclei segmentations against manual ground truth.

Compares instance label images (0 = background, 1..N = objects) from one or
more segmentation configurations with hand-annotated ground truth, image by
image, and writes:

- ``summary_metrics.csv``   dataset-level precision / recall / F1 / accuracy
                            per method and IoU threshold (TP/FP/FN summed
                            over all images, as in StarDist's matching_dataset)
- ``per_image_metrics.csv`` the same metrics for every image
- ``split_merge.csv``       over-segmentation (split) and under-segmentation
                            (merge) counts per image
- ``matched_objects.csv``   every matched GT/prediction pair at IoU 0.5, with
                            its IoU and sizes (volume agreement)
- ``accuracy_vs_iou.png``   F1 and accuracy against the IoU threshold

File layout: one ground-truth folder and one folder per method, each holding
a label TIFF per image with the same file name, e.g.::

    ground_truth/embryo01.tif
    validation_labels/full/embryo01.tif
    validation_labels/no_refinement/embryo01.tif

Predictions on a different grid than the ground truth (the pipeline's labels
are on the isotropic, resampled grid) are mapped onto the ground-truth grid by
nearest-neighbour sampling, so both must cover the same field of view (the
same ROI).

Usage::

    python validation/evaluate_segmentation.py --gt ground_truth \\
        --pred full=validation_labels/full \\
        --pred no_refinement=validation_labels/no_refinement \\
        --out validation_results --voxel-size 0.5 0.2 0.2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import relabel_sequential

LABEL_EXTENSIONS = (".tif", ".tiff")
DEFAULT_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
# A prediction (or GT object) "belongs" to an object of the other image when
# at least this fraction of its voxels lie inside it (split/merge analysis).
OWNERSHIP_FRACTION = 0.5


def load_labels(path):
    """Read a label TIFF, drop singleton axes and relabel to 1..N."""
    labels = np.squeeze(tifffile.imread(str(path)))
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.mod(labels, 1) == 0):
            raise ValueError(f"{path} is not a label image (non-integer values).")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() < 0:
        raise ValueError(f"{path} contains negative labels.")
    labels, _, _ = relabel_sequential(labels)
    return labels


def resample_to_shape(labels, shape):
    """Nearest-neighbour resampling of a label image onto ``shape``.

    Uses voxel-centre sampling, so the output has exactly ``shape`` and label
    values are never mixed.
    """
    if labels.shape == tuple(shape):
        return labels
    if labels.ndim != len(shape):
        raise ValueError(f"Cannot map a {labels.ndim}D prediction onto a "
                         f"{len(shape)}D ground truth.")
    index = [
        np.minimum(((np.arange(n_out) + 0.5) * n_in / n_out).astype(np.int64), n_in - 1)
        for n_in, n_out in zip(labels.shape, shape)
    ]
    return labels[np.ix_(*index)]


def overlap_table(gt, pred):
    """Sizes of every object and intersections of every overlapping pair.

    Returns
    -------
    gt_sizes, pred_sizes : ndarray
        Voxel counts indexed by label (index 0 = background).
    pairs : DataFrame
        Columns ``gt``, ``pred``, ``inter``, ``iou`` for every GT/prediction
        pair sharing at least one voxel.
    """
    gt_sizes = np.bincount(gt.ravel(), minlength=1)
    pred_sizes = np.bincount(pred.ravel(), minlength=1)

    both = (gt > 0) & (pred > 0)
    n_pred = len(pred_sizes)
    keys, inter = np.unique(gt[both] * n_pred + pred[both], return_counts=True)
    gt_ids, pred_ids = np.divmod(keys, n_pred)

    union = gt_sizes[gt_ids] + pred_sizes[pred_ids] - inter
    pairs = pd.DataFrame({
        "gt": gt_ids, "pred": pred_ids, "inter": inter, "iou": inter / union,
    })
    return gt_sizes, pred_sizes, pairs


def match_pairs(pairs, threshold):
    """One-to-one matching of pairs with IoU >= ``threshold`` (>= 0.5).

    Above 0.5 each object can overlap at most one other object that much, so
    matching by descending IoU gives the optimal assignment; the greedy pass
    only resolves exact ties at 0.5.
    """
    candidates = pairs[pairs["iou"] >= threshold].sort_values("iou", ascending=False)
    used_gt, used_pred, rows = set(), set(), []
    for row in candidates.itertuples(index=False):
        if row.gt in used_gt or row.pred in used_pred:
            continue
        used_gt.add(row.gt)
        used_pred.add(row.pred)
        rows.append(row)
    return pd.DataFrame(rows, columns=pairs.columns)


def detection_metrics(matched, n_gt, n_pred):
    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp
    sum_iou = float(matched["iou"].sum())
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_matched_iou": sum_iou,
        **rates(tp, fp, fn, sum_iou),
    }


def rates(tp, fp, fn, sum_iou):
    """Precision, recall, F1, accuracy and panoptic quality from counts."""
    def ratio(a, b):
        return a / b if b else np.nan
    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        # StarDist's "accuracy" / average precision at a single threshold.
        "accuracy": ratio(tp, tp + fp + fn),
        "mean_matched_iou": ratio(sum_iou, tp),
        "panoptic_quality": ratio(sum_iou, tp + 0.5 * fp + 0.5 * fn),
    }


def split_merge_errors(pairs, gt_sizes, pred_sizes):
    """Count split (over-segmented) GT objects and merged predictions.

    A GT object is split when two or more predictions each lie mostly
    (>= OWNERSHIP_FRACTION of their voxels) inside it; a prediction is a
    merge when it contains most of two or more GT objects.
    """
    pred_owned = pairs[pairs["inter"] >= OWNERSHIP_FRACTION * pred_sizes[pairs["pred"]]]
    pieces_per_gt = pred_owned.groupby("gt").size()
    split = pieces_per_gt[pieces_per_gt >= 2]

    gt_owned = pairs[pairs["inter"] >= OWNERSHIP_FRACTION * gt_sizes[pairs["gt"]]]
    gt_per_pred = gt_owned.groupby("pred").size()
    merged = gt_per_pred[gt_per_pred >= 2]

    n_gt = len(gt_sizes) - 1
    n_pred = len(pred_sizes) - 1
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "split_gt_objects": len(split),
        "extra_split_pieces": int((split - 1).sum()),
        "merged_predictions": len(merged),
        "gt_objects_in_merges": int(merged.sum()),
        # GT objects with no predicted voxel at all, and predictions with no
        # GT voxel (false detections in the background).
        "missed_gt_objects": n_gt - pairs["gt"].nunique(),
        "background_predictions": n_pred - pairs["pred"].nunique(),
    }


def find_label_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    return {p.name: p for p in sorted(folder.iterdir())
            if p.suffix.lower() in LABEL_EXTENSIONS}


def evaluate(gt_dir, methods, thresholds, voxel_volume=None):
    gt_files = find_label_files(gt_dir)
    if not gt_files:
        raise SystemExit(f"No label TIFFs in {gt_dir}")

    per_image, split_rows, object_rows = [], [], []
    for method, pred_dir in methods.items():
        pred_files = find_label_files(pred_dir)
        missing = sorted(set(gt_files) - set(pred_files))
        if missing:
            print(f"[{method}] no prediction for: {', '.join(missing)} (skipped)")

        for name in sorted(set(gt_files) & set(pred_files)):
            gt = load_labels(gt_files[name])
            pred_raw = load_labels(pred_files[name])
            if pred_raw.shape != gt.shape:
                print(f"[{method}] {name}: prediction {pred_raw.shape} resampled "
                      f"to ground-truth grid {gt.shape}")
            pred, _, _ = relabel_sequential(resample_to_shape(pred_raw, gt.shape))

            gt_sizes, pred_sizes, pairs = overlap_table(gt, pred)
            n_gt, n_pred = len(gt_sizes) - 1, len(pred_sizes) - 1

            for thr in thresholds:
                matched = match_pairs(pairs, thr)
                per_image.append({"method": method, "image": name, "iou_threshold": thr,
                                  **detection_metrics(matched, n_gt, n_pred)})

            split_rows.append({"method": method, "image": name,
                               **split_merge_errors(pairs, gt_sizes, pred_sizes)})

            matched = match_pairs(pairs, 0.5)
            for row in matched.itertuples(index=False):
                record = {
                    "method": method, "image": name,
                    "gt_label": row.gt, "pred_label": row.pred, "iou": row.iou,
                    "gt_voxels": gt_sizes[row.gt], "pred_voxels": pred_sizes[row.pred],
                    "relative_size_error": (pred_sizes[row.pred] - gt_sizes[row.gt])
                                           / gt_sizes[row.gt],
                }
                if voxel_volume is not None:
                    record["gt_size_um"] = gt_sizes[row.gt] * voxel_volume
                    record["pred_size_um"] = pred_sizes[row.pred] * voxel_volume
                object_rows.append(record)

    if not per_image:
        raise SystemExit("No image had both a ground truth and a prediction.")
    return pd.DataFrame(per_image), pd.DataFrame(split_rows), pd.DataFrame(object_rows)


def summarize(per_image, split_merge, objects):
    """Dataset-level metrics: counts summed over images, then rates."""
    rows = []
    for (method, thr), group in per_image.groupby(["method", "iou_threshold"], sort=False):
        tp, fp, fn = group["tp"].sum(), group["fp"].sum(), group["fn"].sum()
        sum_iou = group["sum_matched_iou"].sum()
        rows.append({"method": method, "iou_threshold": thr, "n_images": len(group),
                     "n_gt": group["n_gt"].sum(), "n_pred": group["n_pred"].sum(),
                     "tp": tp, "fp": fp, "fn": fn, **rates(tp, fp, fn, sum_iou)})
    summary = pd.DataFrame(rows)

    totals = split_merge.drop(columns="image").groupby("method", sort=False).sum()
    if not objects.empty:
        abs_err = objects["relative_size_error"].abs().groupby(objects["method"])
        totals["median_abs_size_error"] = abs_err.median()
    return summary, totals.reset_index()


def plot_curves(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for method, group in summary.groupby("method", sort=False):
        for ax, metric in zip(axes, ("f1", "accuracy")):
            ax.plot(group["iou_threshold"], group[metric], marker="o", label=method)
    for ax, title in zip(axes, ("F1", "Accuracy  TP / (TP + FP + FN)")):
        ax.set_title(title)
        ax.set_xlabel("IoU threshold")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_methods(entries):
    methods = {}
    for entry in entries:
        name, sep, folder = entry.partition("=")
        if not sep:
            name, folder = Path(entry).name, entry
        if name in methods:
            raise SystemExit(f"Method name used twice: {name}")
        methods[name] = folder
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nuclei label images against manual ground truth.")
    parser.add_argument("--gt", required=True, help="folder of ground-truth label TIFFs")
    parser.add_argument("--pred", required=True, action="append", metavar="NAME=FOLDER",
                        help="folder of predicted label TIFFs; repeat to compare methods")
    parser.add_argument("--out", default="validation_results", help="output folder")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                        help="IoU thresholds (>= 0.5), default 0.5 to 0.95")
    parser.add_argument("--voxel-size", type=float, nargs="+", metavar="UM",
                        help="ground-truth voxel size in um, Z Y X (or Y X for 2D); "
                             "adds sizes in um3 / um2 to matched_objects.csv")
    args = parser.parse_args()

    thresholds = sorted(set(np.round(args.thresholds, 4)))
    if min(thresholds) < 0.5 or max(thresholds) > 1:
        parser.error("IoU thresholds must lie between 0.5 and 1.")
    voxel_volume = float(np.prod(args.voxel_size)) if args.voxel_size else None

    per_image, split_merge, objects = evaluate(
        args.gt, parse_methods(args.pred), thresholds, voxel_volume)
    summary, split_totals = summarize(per_image, split_merge, objects)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    per_image.to_csv(out / "per_image_metrics.csv", index=False)
    split_merge.to_csv(out / "split_merge.csv", index=False)
    split_totals.to_csv(out / "split_merge_summary.csv", index=False)
    objects.to_csv(out / "matched_objects.csv", index=False)
    plot_curves(summary, out / "accuracy_vs_iou.png")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    at_05 = summary[np.isclose(summary["iou_threshold"], 0.5)]
    print("\nDetection at IoU 0.5:")
    print(at_05.drop(columns="iou_threshold").to_string(index=False))
    print("\nSplit / merge errors:")
    print(split_totals.to_string(index=False))
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
