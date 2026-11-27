"""Ground truth for the LIVE/DEAD notebook: correct its nuclei and cells in napari.

Called from ``Fluo_3D_LD_seg_v1.2.2_validation.ipynb`` once the LIVE/DEAD
analysis has run (after Cell 20). A napari window opens with the original
channels of the ROI and two editable Labels layers, drafted from the pipeline:

- ``GT nuclei``: the stained nuclei (``im_segmentation_stack['Nuclei (stained)']``;
  the estimated nuclei of cells without a nuclear stain are left out, they
  are not visible in the image);
- ``GT cells``: the whole cells (``im_segmentation_stack['Cytoplasm']``,
  nucleus included).

The draft errs on the side of too many objects, so that correcting it is
deleting false positives, not painting missed objects: to the pipeline's
objects are added **candidates** wherever a channel shows minimal signal
(background + ``sensitivity`` x noise), each grown from its brightest core
until the intensity above background falls below ``edge_fraction`` (10 %)
of the core's, highlighted in yellow. Shift+D
deletes the whole object under the mouse.

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


def _sphere_voxels(diameter_um, spacing):
    """Voxels of a sphere (3D) or disc (2D) of ``diameter_um`` on a grid of ``spacing``."""
    if len(spacing) == 3:
        return np.pi / 6 * diameter_um ** 3 / np.prod(spacing)
    return np.pi / 4 * diameter_um ** 2 / np.prod(spacing)


def _drop_small(labels, min_voxels):
    """Remove the labels with fewer than ``min_voxels`` voxels."""
    sizes = np.bincount(labels.ravel())
    small = sizes < min_voxels
    small[0] = False
    if small.any():
        labels = labels.copy()
        labels[small[labels]] = 0
    return labels


def _grow_from_cores(signal, detection, spacing, split_diameter_um, noise, sensitivity,
                     edge_fraction):
    """Candidates of one channel: a core (local maximum) grown down to
    ``edge_fraction`` x its own peak.

    ``signal`` is the smoothed channel minus its background. Cores are the
    local maxima of ``signal`` inside ``detection`` (>= ``sensitivity`` x
    noise), at least half a ``split_diameter_um`` apart. Every core claims the
    voxels around it along the intensity (watershed from the cores, so
    touching cells are split at the dimmest seam between them), and keeps
    those with ``signal`` >= ``edge_fraction`` x its peak. Two limits keep a
    dim core from spreading into the background: a voxel must also stand
    >= 1 x noise above background, and lie within ``split_diameter_um`` of a
    core.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.segmentation import watershed

    radius = np.maximum(1, np.round(0.5 * split_diameter_um / spacing)).astype(int)
    footprint = np.ones(tuple(2 * radius + 1), dtype=bool)
    cores = peak_local_max(signal, footprint=footprint, labels=label(detection),
                           threshold_abs=sensitivity * noise, exclude_border=False)
    if len(cores) == 0:
        return np.zeros(signal.shape, dtype=np.uint32)
    markers = np.zeros(signal.shape, dtype=np.int32)
    markers[tuple(cores.T)] = np.arange(1, len(cores) + 1)
    peaks = signal[tuple(cores.T)]

    # Edge of each core: edge_fraction x its peak, never below 1 x noise.
    edge = np.maximum(edge_fraction * peaks, noise)
    reach = ndi.distance_transform_edt(markers == 0, sampling=spacing) <= split_diameter_um
    labels = watershed(-signal, markers, mask=(signal >= edge.min()) & reach)
    edge_of_label = np.concatenate([[np.inf], edge]).astype(np.float32)
    labels[signal < edge_of_label[labels]] = 0
    return labels.astype(np.uint32)


def _candidate_objects(image, channel_idx, spacing, smooth_um, split_diameter_um,
                       min_diameter_um, sensitivity, edge_fraction=0.1):
    """Candidate objects wherever any of ``channel_idx`` shows minimal signal.

    Per channel: smoothed (Gaussian, ``smooth_um``), background and noise
    from its median and MAD (so each channel gets its own threshold); signal
    above background + ``sensitivity`` x noise is detected, specks smaller
    than a quarter of a ``min_diameter_um`` sphere are dropped, and every
    core is grown down to ``edge_fraction`` x its own peak
    (`_grow_from_cores`). The channels are then combined: a candidate of a
    later channel lying mostly inside an earlier one is left out.
    """
    from scipy import ndimage as ndi
    from skimage.measure import label

    spacing = np.asarray(spacing, dtype=float)
    sigma = np.maximum(0.5, smooth_um / spacing)
    min_voxels = 0.25 * _sphere_voxels(min_diameter_um, spacing)
    found = np.zeros(image.shape[:-1], dtype=np.uint32)
    for c in channel_idx:
        signal = ndi.gaussian_filter(image[..., c].astype(np.float32), sigma)
        sample = signal.ravel()[::max(1, signal.size // 5_000_000)]
        background = float(np.median(sample))
        noise = 1.4826 * float(np.median(np.abs(sample - background)))
        if noise <= 0:
            noise = float(sample.std()) or 1.0
        signal -= background
        detection = _drop_small(label(signal > sensitivity * noise), min_voxels) > 0
        if detection.any():
            grown = _drop_small(_grow_from_cores(signal, detection, spacing,
                                                 split_diameter_um, noise, sensitivity,
                                                 edge_fraction), min_voxels)
            found, _, _ = _add_candidates(found, grown)
        del signal, detection
    return found


def _add_candidates(base, candidates):
    """``base`` plus the candidates lying mostly (>= 50 %) outside its objects.

    Existing objects are never changed: candidates only fill background.
    Returns the labels, the number of candidates added and their first label
    (every label >= it is an added candidate).
    """
    first = int(base.max()) + 1
    sizes = np.bincount(candidates.ravel())
    inside = np.bincount(candidates.ravel(), weights=(base.ravel() > 0),
                         minlength=len(sizes))
    new = np.nonzero((sizes > 0) & (inside < 0.5 * sizes))[0]
    new = new[new > 0]
    lut = np.zeros(len(sizes), dtype=np.uint32)
    lut[new] = np.arange(first, first + len(new), dtype=np.uint32)
    added = lut[candidates]
    out = base.copy()
    free = (base == 0) & (added > 0)
    out[free] = added[free]
    return out, len(new), first


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
        kept = ""
        candidates = record.get("candidates") or {}
        if candidates.get("added"):
            ids = np.unique(labels)
            candidates["kept_at_save"] = int(np.count_nonzero(
                (ids >= candidates["first_label"]) & (ids <= candidates["last_label"])))
            kept = f", {candidates['kept_at_save']} of {candidates['added']} added candidates kept"
        write_record(paths[comp].with_suffix(".json"), record)
        print(f"Saved {n} {comp} to {paths[comp]} "
              f"({100 * changed / labels.size:.2f}% of voxels changed from the pipeline "
              f"draft{kept})")
    outside = _nuclei_outside_cells(np.asarray(layers["nuclei"].data),
                                    np.asarray(layers["cells"].data))
    if outside:
        print(f"Check: {outside} GT nuclei lie mostly outside every GT cell.")


def annotate_ld_from_notebook(img, file_meta, stain_dict, im_segmentation_stack, stain_df,
                              input_file, roi, voxel_size, annotator,
                              out="ground_truth_ld", restart=False,
                              localization=None, nuclei_diameter=None, cell_diameter=None,
                              sensitivity=3.0, edge_fraction=0.1,
                              pipeline_settings=None, stain_complete_df=None):
    """Correct the LIVE/DEAD notebook's nuclei and cells in napari and save them.

    The draft is deliberately generous, so that correcting it means deleting
    false positives rather than painting missed objects: the pipeline's
    objects plus **candidates** wherever a channel shows minimal signal
    (background + ``sensitivity`` x noise), each grown from its brightest
    core down to ``edge_fraction`` x that core's intensity above background
    (see `_candidate_objects`). Nuclei
    candidates come from the nuclear markers, cell candidates from every
    marker. Candidates only fill the background, the pipeline's objects are
    kept as they are; they are highlighted in yellow in the viewer, and
    Shift+D deletes the whole object under the mouse.

    Parameters
    ----------
    img : array (Z, Y, X, C)
        The ROI as loaded in Cell 5, before any processing or zoom: it is
        shown and sets the grid of the ground truth. (``im_final_stack``'s
        'Original image' cannot be used: Cell 20 frees it.)
    file_meta, stain_dict : dict
        From Cells 5-6; put the channels of ``img`` in ``stain_df`` row order
        and drop the channels not in the stain table, as the pipeline does.
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
    localization : dict
        Condition -> 'nuclear' / 'cytoplasmic' (Cell 6): which markers give
        the nuclei candidates.
    nuclei_diameter, cell_diameter : float
        In um (Cells 3 and 8): candidates smaller than a quarter of a nucleus
        are dropped, touching nuclei / cells are split at these sizes.
    sensitivity : float
        Candidate threshold, background + this x noise. Lower = more
        candidates (more to delete); 0 or None = pipeline objects only.
        Only used when a compartment starts from the pipeline (not when
        resuming a saved ground truth).
    edge_fraction : float
        A candidate extends from its core (local maximum) as long as the
        intensity above background stays >= this x the core's (0.1 = 10 %).
    pipeline_settings : dict, optional
        Settings of the run that made the draft, stored in the records.

    Returns
    -------
    dict compartment -> Path of the saved TIFF.
    """
    annotator = str(annotator or "").strip()
    if not annotator:
        raise ValueError("Set annotator to your name.")

    from helpers.notebook_helpers import build_stain_dataframe
    channel_order = build_stain_dataframe(stain_dict, file_meta)["Channel_index"].to_numpy()
    image = np.asarray(img)[..., channel_order.astype(int)]  # (Z, Y, X, C), stain_df order
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

    # Markers giving the candidates: nuclei <- nuclear markers, cells <- all markers.
    loc = {str(k).strip().upper(): v for k, v in (localization or {}).items()}
    conditions = [str(cond).strip().upper() for cond in stain_df.index]
    candidate_channels = {
        "nuclei": [c for c, cond in enumerate(conditions) if loc.get(cond) == "nuclear"],
        "cells": list(range(len(conditions))),
    }
    split_diameter = {"nuclei": nuclei_diameter, "cells": cell_diameter or nuclei_diameter}
    use_candidates = bool(sensitivity) and nuclei_diameter is not None
    if sensitivity and nuclei_diameter is None:
        print("No nuclei_diameter given: no extra candidates, pipeline objects only.")

    drafts, start, records, highlight = {}, {}, {}, {}
    for comp, (_, keys) in COMPARTMENTS.items():
        draft, draft_key = _draft(im_segmentation_stack, keys, shape)
        drafts[comp] = draft
        candidates_info = None
        if paths[comp].exists() and not restart:
            start[comp], saved_record = _load_existing(paths[comp], shape)
            candidates_info = saved_record.get("candidates")
            print(f"Resuming {paths[comp]} ({_n_objects(start[comp])} {comp})"
                  + ("; set gt_restart = True to start again with the candidates."
                     if use_candidates and not candidates_info else "."))
        elif use_candidates and candidate_channels[comp]:
            print(f"Finding {comp} candidates (minimal signal in "
                  f"{', '.join(conditions[c] for c in candidate_channels[comp])})...")
            found = _candidate_objects(
                image, candidate_channels[comp], scale,
                smooth_um=0.1 * nuclei_diameter,
                split_diameter_um=split_diameter[comp],
                min_diameter_um=nuclei_diameter,
                sensitivity=sensitivity,
                edge_fraction=edge_fraction)
            start[comp], n_added, first = _add_candidates(draft, found)
            saved_record = {}
            candidates_info = {"sensitivity": sensitivity, "edge_fraction": edge_fraction,
                               "added": n_added,
                               "first_label": first, "last_label": first + n_added - 1,
                               "markers": [conditions[c] for c in candidate_channels[comp]]}
            print(f"  {_n_objects(draft)} pipeline {comp} + {n_added} candidates "
                  f"(labels {first}-{first + n_added - 1}, yellow in the viewer).")
            del found
        else:
            start[comp], saved_record = draft.copy(), {}
        if candidates_info and candidates_info.get("added"):
            highlight[comp] = ((start[comp] >= candidates_info["first_label"])
                               & (start[comp] <= candidates_info["last_label"]))
        records[comp] = {
            "kind": "ground_truth",
            "compartment": comp,
            "created": saved_record.get("created", now()),
            "draft": f"LIVE/DEAD pipeline output, im_segmentation_stack['{draft_key}']"
                     + (", plus candidates of minimal signal" if candidates_info else "")
                     + ", manually corrected",
            "candidates": candidates_info,
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
        for comp, mask in highlight.items():  # static reference: not updated by edits
            viewer.add_image(mask.astype(np.uint8), name=f"candidates added - {comp}",
                             scale=scale, colormap="yellow", blending="additive",
                             opacity=0.35, contrast_limits=(0, 1))
        viewer.layers.selection.active = layers["nuclei"]

        @viewer.bind_key("Shift-S", overwrite=True)
        def _save_key(_viewer):
            _save(layers, drafts, paths, records, session)

        @viewer.bind_key("Shift-D", overwrite=True)
        def _delete_key(_viewer):
            layer = _viewer.layers.selection.active
            if layer not in layers.values():
                print("Shift+D: select 'GT nuclei' or 'GT cells' first.")
                return
            value = layer.get_value(_viewer.cursor.position, world=True)
            if not value:
                return
            data = layer.data
            data[data == value] = 0  # whole object, every slice
            layer.refresh()
            print(f"Deleted label {value} from '{layer.name}'.")

        print("\nCorrect 'GT nuclei' and 'GT cells' (select the layer to edit).\n"
              "  Yellow = candidates added to the pipeline's objects (reference layer,\n"
              "  it does not update): keep the real ones, delete the rest.\n"
              "  Shift+D = delete the whole object under the mouse (all slices),\n"
              "  M = new label, paint / fill / erase / pick tools in the layer controls,\n"
              "  tick 'preserve labels' to avoid painting over neighbours.\n"
              "  Shift+S = save, closing the window saves too.")
        _run_blocking_viewer(viewer, napari)
    finally:
        settings.application.ipy_interactive = was_ipy_interactive
    _save(layers, drafts, paths, records, session)
    return paths
