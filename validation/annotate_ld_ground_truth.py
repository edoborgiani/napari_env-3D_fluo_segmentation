"""Ground truth for the LIVE/DEAD notebook: correct its nuclei and cells in napari.

Called from ``Fluo_3D_LD_seg_v1.2.2_validation.ipynb`` once the LIVE/DEAD
analysis has run (after Cell 20). A napari window opens with the original
channels of the ROI and two editable Labels layers, drafted from the pipeline:

- ``GT nuclei``: the stained nuclei (``im_segmentation_stack['Nuclei (stained)']``;
  the estimated nuclei of cells without a nuclear stain are left out, they
  are not visible in the image);
- ``GT cells``: the whole cells (``im_segmentation_stack['Cytoplasm']``,
  nucleus included).

Both are shown on the **original voxel grid of the ROI** (before any zoom),
like the nuclei ground truth of ``annotate_ground_truth.py``: the pipeline's
labels, on the resampled grid, are mapped onto it by nearest-neighbour
sampling, the same mapping ``evaluate_segmentation.py`` uses. When the window
is closed (or with Shift+S) each layer is saved as::

    <out>/nuclei/<image name>.tif   + .json
    <out>/cells/<image name>.tif    + .json

The ``.json`` holds the source file, ROI, grid, stain table, the pipeline
settings the draft came from, the annotator and date/time of every session,
how much of the draft was changed, and the software versions. Running it
again resumes from the saved TIFFs (``restart=True`` starts from the
pipeline draft again).

Each folder is a ground-truth folder for ``evaluate_segmentation.py``::

    python validation/evaluate_segmentation.py --gt ground_truth_ld/nuclei \\
        --pred full=validation_labels/ld_nuclei/full --voxel-size Z Y X
    python validation/evaluate_segmentation.py --gt ground_truth_ld/cells \\
        --pred full=validation_labels/ld_cells/full --voxel-size Z Y X

Labels are saved as painted (not renumbered), so a nucleus and its cell can
keep the same label; ``evaluate_segmentation.py`` renumbers on reading.

Note: a ground truth corrected from the pipeline's own output is biased
towards that pipeline. The record says so (``draft``) and counts the voxels
changed, so it can be reported ("initialised with the pipeline and manually
corrected").
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from helpers.notebook_helpers import _image_stem  # noqa: E402
from validation.evaluate_segmentation import resample_to_shape  # noqa: E402
from validation.run_metadata import (  # noqa: E402
    now, software_info, source_file_info, write_record)

# compartment -> (layer name, draft key(s) in im_segmentation_stack, first found is used)
COMPARTMENTS = {
    "cells": ("GT cells", ("Cytoplasm",)),
    "nuclei": ("GT nuclei", ("Nuclei (stained)", "Nuclei")),
}
NAPARI_COLORMAPS = {"gray", "red", "green", "blue", "cyan", "magenta", "yellow"}


def _n_objects(labels):
    ids = np.unique(labels)
    return int(np.count_nonzero(ids))


def _draft(im_segmentation_stack, keys, shape):
    """Pipeline labels for one compartment, mapped onto the original grid."""
    for key in keys:
        if key in im_segmentation_stack:
            labels = np.asarray(im_segmentation_stack[key]).astype(np.uint32)
            if labels.ndim == len(shape) + 1 and labels.shape[0] == 1:  # 2D: (1, Y, X)
                labels = labels[0]
            return resample_to_shape(labels, shape), key
    raise KeyError(f"im_segmentation_stack has none of {keys}: run Cells 17-18 first.")


def _load_existing(gt_path, shape):
    labels = np.asarray(tifffile.imread(gt_path)).astype(np.uint32)
    if labels.ndim == len(shape) + 1 and labels.shape[0] == 1:  # 2D saved as (1, Y, X)
        labels = labels[0]
    if labels.shape != tuple(shape):
        raise SystemExit(f"{gt_path} has shape {labels.shape} but the image is {shape}: "
                         "use the same ROI as before, or restart=True.")
    record_path = gt_path.with_suffix(".json")
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    return labels, record


def _nuclei_outside_cells(nuclei, cells):
    """Number of GT nuclei with less than half of their voxels inside a GT cell."""
    if not nuclei.any():
        return 0
    total = np.bincount(nuclei.ravel())
    inside = np.bincount(nuclei.ravel(), weights=(cells.ravel() > 0), minlength=len(total))
    ids = np.nonzero(total)[0]
    ids = ids[ids > 0]
    return int(np.sum(inside[ids] < 0.5 * total[ids]))


def _save(layers, drafts, paths, records, session):
    """Write every compartment's labels and record; ``session`` gets the counts."""
    session["saved"] = now()
    for comp, layer in layers.items():
        labels = np.asarray(layer.data).astype(np.uint32)
        paths[comp].parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(paths[comp], labels, compression="zlib")
        n = _n_objects(labels)
        changed = int(np.count_nonzero(labels != drafts[comp]))
        session[f"n_{comp}"] = n
        record = records[comp]
        record.update(saved=session["saved"], n_objects=n,
                      changed_voxels_vs_pipeline_draft=changed,
                      changed_fraction_vs_pipeline_draft=changed / labels.size)
        write_record(paths[comp].with_suffix(".json"), record)
        print(f"Saved {n} {comp} to {paths[comp]} "
              f"({100 * changed / labels.size:.2f}% of voxels changed from the pipeline draft)")
    outside = _nuclei_outside_cells(np.asarray(layers["nuclei"].data),
                                    np.asarray(layers["cells"].data))
    if outside:
        print(f"Check: {outside} GT nuclei lie mostly outside every GT cell.")


def annotate_ld_from_notebook(im_final_stack, im_segmentation_stack, stain_df,
                              input_file, roi, voxel_size, annotator,
                              out="ground_truth_ld", restart=False,
                              pipeline_settings=None, stain_complete_df=None):
    """Correct the LIVE/DEAD notebook's nuclei and cells in napari and save them.

    Parameters
    ----------
    im_final_stack : dict
        The notebook's image stack; its ``'Original image'`` (Z, Y, X, C), the
        ROI on the original grid with channels in ``stain_df`` row order, is
        shown and sets the grid of the ground truth.
    im_segmentation_stack : dict
        Output of Cells 17-19 (``'Cytoplasm'``, ``'Nuclei (stained)'``, ...),
        the draft of the layers.
    stain_df : DataFrame
        Stain table (Condition -> Marker, Laser, Color), names the channels.
    voxel_size : tuple
        (r_Z, r_Y, r_X), the original voxel size in um.
    annotator : str
        Name of the person correcting, recorded per session.
    out : str
        Output folder; ``<out>/nuclei`` and ``<out>/cells`` are created.
    restart : bool
        True = ignore saved TIFFs and start from the pipeline draft again.
    pipeline_settings : dict, optional
        Settings of the run that made the draft, stored in the records.

    Returns
    -------
    dict compartment -> Path of the saved TIFF.
    """
    annotator = str(annotator or "").strip()
    if not annotator:
        raise ValueError("Set annotator to your name.")

    image = np.asarray(im_final_stack["Original image"])  # (Z, Y, X, C)
    scale = tuple(float(v) for v in voxel_size)
    if image.shape[0] == 1:  # 2D: annotate (Y, X)
        image, scale = image[0], scale[1:]
    shape = image.shape[:-1]

    stem = _image_stem(input_file)
    paths = {comp: Path(out) / comp / f"{stem}.tif" for comp in COMPARTMENTS}
    channels = [f"{cond} ({row['Marker']})" for cond, row in stain_df.iterrows()]
    colors = [str(c).strip().lower() for c in stain_df["Color"]]
    if len(channels) != image.shape[-1]:  # should not happen: channels follow stain_df rows
        channels = [f"Channel {c + 1}" for c in range(image.shape[-1])]
        colors = ["gray"] * image.shape[-1]

    drafts, start, records = {}, {}, {}
    for comp, (_, keys) in COMPARTMENTS.items():
        draft, draft_key = _draft(im_segmentation_stack, keys, shape)
        drafts[comp] = draft
        if paths[comp].exists() and not restart:
            start[comp], saved_record = _load_existing(paths[comp], shape)
            print(f"Resuming {paths[comp]} ({_n_objects(start[comp])} {comp}).")
        else:
            start[comp], saved_record = draft.copy(), {}
        records[comp] = {
            "kind": "ground_truth",
            "compartment": comp,
            "created": saved_record.get("created", now()),
            "draft": f"LIVE/DEAD pipeline output, im_segmentation_stack['{draft_key}'], "
                     "manually corrected",
            "n_objects_pipeline_draft": _n_objects(draft),
            "source_file": source_file_info(input_file),
            "roi_x0_x1_y0_y1_z0_z1": list(roi),
            "shape": list(shape),
            "voxel_size_um": list(scale),
            "channels": channels,
            "pipeline_settings": pipeline_settings or {},
            "software": software_info(),
            "sessions": saved_record.get("sessions", []),
        }
        if stain_complete_df is not None:
            records[comp]["stain_table"] = stain_complete_df.reset_index().to_dict(
                orient="records")

    session = {"annotator": annotator, "started": now(),
               **{f"n_{comp}_at_start": _n_objects(start[comp]) for comp in COMPARTMENTS}}
    for record in records.values():
        record["sessions"] = record["sessions"] + [session]
        record["annotators"] = sorted({s["annotator"] for s in record["sessions"]})

    import napari
    from napari.settings import get_settings
    from helpers.notebook_helpers import _close_all_napari_viewers, _run_blocking_viewer

    # The viewer only blocks when ipy_interactive is off when it is created.
    settings = get_settings()
    was_ipy_interactive = settings.application.ipy_interactive
    settings.application.ipy_interactive = False
    _close_all_napari_viewers(napari)
    try:
        viewer = napari.Viewer(title=f"LIVE/DEAD ground truth - {stem}")
        for c, name in enumerate(channels):
            data = image[..., c]
            low, high = np.percentile(data, (1, 99.8))
            color = colors[c]
            viewer.add_image(data, name=name, scale=scale, blending="additive",
                             colormap=color if color in NAPARI_COLORMAPS else "gray",
                             contrast_limits=(float(low), float(max(high, low + 1))))
        layers = {}
        for comp in ("cells", "nuclei"):  # nuclei on top
            layer = viewer.add_labels(start[comp], name=COMPARTMENTS[comp][0],
                                      scale=scale, opacity=0.5)
            if len(shape) == 3:
                layer.n_edit_dimensions = 2  # paint slice by slice
            layer.selected_label = int(start[comp].max()) + 1
            layers[comp] = layer
        layers["cells"].contour = 2  # outlines, so nuclei and signal stay visible
        viewer.layers.selection.active = layers["nuclei"]

        @viewer.bind_key("Shift-S", overwrite=True)
        def _save_key(_viewer):
            _save(layers, drafts, paths, records, session)

        print("\nCorrect 'GT nuclei' and 'GT cells' (select the layer to edit).\n"
              "  M = new label, paint / fill / erase / pick tools in the layer controls,\n"
              "  tick 'preserve labels' to avoid painting over neighbours,\n"
              "  a nucleus and its cell can share a label (optional).\n"
              "  Shift+S = save, closing the window saves too.")
        _run_blocking_viewer(viewer, napari)
    finally:
        settings.application.ipy_interactive = was_ipy_interactive
    _save(layers, drafts, paths, records, session)
    return paths
