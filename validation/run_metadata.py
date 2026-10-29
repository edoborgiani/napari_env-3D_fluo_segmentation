"""JSON metadata records for ground-truth annotations and validation labels.

Every label TIFF written for validation gets a ``<image name>.json`` next to
it with what is needed to reproduce it: the source file, ROI, grid, the
settings used and the software state (git commit, package versions).
evaluate_segmentation.py only reads the TIFFs, so the JSON files can sit in
the same folders.
"""

import datetime
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("numpy", "scipy", "scikit-image", "napari", "cellpose", "stardist",
            "tensorflow", "torch", "tifffile", "aicsimageio", "bioio")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def source_file_info(path):
    """Resolved path, size and modification time: enough to spot a changed file."""
    path = Path(path).resolve()
    info = {"path": str(path)}
    if path.is_file():
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["modified"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds")
    return info


def software_info():
    """Git commit of this repository, Python and installed package versions."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        info["git_commit"] = git("rev-parse", "HEAD")
        info["git_uncommitted_changes"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    info["packages"] = versions
    return info


def to_jsonable(value):
    """Convert numpy / pandas values (and containers of them) for json.dump."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):  # pandas DataFrame / Series
        return to_jsonable(value.to_dict())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(record), indent=2), encoding="utf-8")


def save_validation_labels(labels, input_file, method_name, roi, voxel_size, settings,
                           stain_complete_df=None, out="validation_labels"):
    """Save the pipeline's nuclei labels and their settings for evaluation.

    Writes ``<out>/<method_name>/<image name>.tif`` and a ``.json`` next to it
    with the ROI, the grid the labels are on (``voxel_size``, the resampled
    (Z, Y, X) voxel size), every setting passed in ``settings`` and the
    per-channel stain table (contrast, gamma, ...).
    """
    import tifffile
    from helpers.notebook_helpers import _image_stem

    labels = np.asarray(labels).astype(np.uint32)
    tif_path = Path(out) / method_name / f"{_image_stem(input_file)}.tif"
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(tif_path, labels, compression="zlib")

    record = {
        "kind": "segmentation",
        "method_name": method_name,
        "created": now(),
        "source_file": source_file_info(input_file),
        "roi_x0_x1_y0_y1_z0_z1": list(roi),
        "shape": list(labels.shape),
        "voxel_size_um_zyx": [float(v) for v in voxel_size],
        "n_nuclei": int(len(np.unique(labels)) - (labels.min() == 0)),
        "settings": settings,
        "software": software_info(),
    }
    if stain_complete_df is not None:
        record["stain_table"] = stain_complete_df.reset_index().to_dict(orient="records")
    write_record(tif_path.with_suffix(".json"), record)
    print(f"Saved {record['n_nuclei']} nuclei and settings to {tif_path} (+ .json)")
    return tif_path
