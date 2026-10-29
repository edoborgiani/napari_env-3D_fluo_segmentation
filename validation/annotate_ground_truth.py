"""Annotate ground-truth nuclei by hand in napari, for evaluate_segmentation.py.

Opens the same ROI of a microscopy file (.nd2, .tif, .czi, .lif, .oib) that the
NucleoSeg3D notebook processed, read with the pipeline's own loader, together
with a Labels layer at the correct voxel scale. Paint one label per nucleus;
when the napari window is closed the labels are saved as
``<out>/<image name>.tif`` (the name evaluate_segmentation.py expects), with a
``<image name>.json`` record of the source file, ROI, channel, voxel size and
draft used, the annotator and date/time of every session, and the software
versions.

The Labels layer starts from, in order of priority:

1. an existing ``<out>/<image name>.tif`` -- annotation resumes where you left;
2. a draft to correct (``--draft cellpose`` or ``--draft path/to/labels.tif``);
3. an empty layer.

Do not use NucleoSeg3D's own output as the draft: the ground truth would then
be biased towards the pipeline being evaluated. Report a Cellpose draft in the
methods ("initialised with Cellpose and manually corrected").

Usage::

    python validation/annotate_ground_truth.py path/to/image.nd2 \\
        --roi 100 400 100 400 0 0 --channel DAPI --annotator "Jane Doe" --out ground_truth

    # Cellpose draft to correct instead of painting from scratch:
    python validation/annotate_ground_truth.py path/to/image.nd2 --channel DAPI \\
        --annotator "Jane Doe" --draft cellpose --nuclei-diameter 10

In napari: press M for a new (unused) label, use the paint / fill / erase
tools of the layer controls, and tick "preserve labels" to avoid painting
over neighbouring nuclei. Shift+S saves without closing the window.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import relabel_sequential

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from helpers.notebook_helpers import _image_stem, load_image_and_metadata  # noqa: E402
from validation.run_metadata import (  # noqa: E402
    now, software_info, source_file_info, write_record)

LAYER_NAME = "ground truth"


def pick_channel(channel, channel_names, n_channels):
    """Channel index from an index or a (case-insensitive) part of its name."""
    if channel is None:
        return 0
    if channel.isdigit():
        index = int(channel)
        if index >= n_channels:
            raise SystemExit(f"Channel {index} does not exist ({n_channels} channels).")
        return index
    hits = [i for i, name in enumerate(channel_names[:n_channels])
            if channel.lower() in str(name).lower()]
    if len(hits) != 1:
        raise SystemExit(f"Channel '{channel}' matches {len(hits)} of the channels "
                         f"{channel_names[:n_channels]}; give its index instead.")
    return hits[0]


def cellpose_draft(nuclei, voxel_size, nuclei_diameter):
    """Cellpose nuclei segmentation of a (Z, Y, X) or (Y, X) image.

    Supports Cellpose 3 (``models.Cellpose``, 'nuclei' model) and Cellpose 4
    (``models.CellposeModel``, Cellpose-SAM, which has no 'nuclei' model).
    """
    from cellpose import models

    xy_spacing = float(np.mean(voxel_size[-2:]))
    kwargs = {"diameter": nuclei_diameter / xy_spacing}
    if nuclei.ndim == 3:
        kwargs.update(do_3D=True, anisotropy=float(voxel_size[0] / xy_spacing))

    print("Running Cellpose for the draft (this can take a while on CPU)...")
    if hasattr(models, "Cellpose"):
        model = models.Cellpose(model_type="nuclei", gpu=True)
        result = model.eval(nuclei, channels=[0, 0], **kwargs)
    else:
        model = models.CellposeModel(gpu=True)
        result = model.eval(nuclei, **kwargs)
    labels = np.asarray(result[0], dtype=np.uint32)
    print(f"Cellpose draft: {int(labels.max())} nuclei.")
    return labels


def load_draft(draft, nuclei, voxel_size, nuclei_diameter):
    if draft == "cellpose":
        if nuclei_diameter is None:
            raise SystemExit("--draft cellpose needs --nuclei-diameter (um).")
        return cellpose_draft(nuclei, voxel_size, nuclei_diameter)

    labels = np.squeeze(tifffile.imread(draft))
    if labels.shape != nuclei.shape:
        raise SystemExit(f"Draft {draft} has shape {labels.shape}, the image ROI "
                         f"{nuclei.shape}; the draft must be on the original grid "
                         "of the same ROI.")
    return labels.astype(np.uint32)


def save_labels(labels, gt_path, record, session):
    """Write the labels and their record; ``session`` (part of the record's
    ``sessions`` list) gets this save's time and nucleus count."""
    labels, _, _ = relabel_sequential(np.asarray(labels).astype(np.uint32))
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(gt_path, labels.astype(np.uint32), compression="zlib")
    session.update(saved=now(), n_nuclei=int(labels.max()))
    record.update(saved=session["saved"], n_nuclei=session["n_nuclei"])
    write_record(gt_path.with_suffix(".json"), record)
    print(f"Saved {record['n_nuclei']} nuclei to {gt_path}")


def initial_labels(nuclei, scale, gt_path, draft, nuclei_diameter, restart):
    """Labels to start from (existing annotation, draft or empty) and their record."""
    if gt_path.exists() and not restart:
        labels = np.squeeze(tifffile.imread(gt_path)).astype(np.uint32)
        if labels.shape != nuclei.shape:
            raise SystemExit(f"{gt_path} has shape {labels.shape} but the image is "
                             f"{nuclei.shape}: annotate the same ROI as before, "
                             "or restart.")
        print(f"Resuming {gt_path} ({int(labels.max())} labels).")
        record = json.loads(gt_path.with_suffix(".json").read_text(encoding="utf-8")) \
            if gt_path.with_suffix(".json").exists() else {}
        return labels, record
    if draft:
        labels = load_draft(draft, nuclei, scale, nuclei_diameter)
    else:
        labels = np.zeros(nuclei.shape, dtype=np.uint32)
    record = {"kind": "ground_truth", "created": now(), "draft": draft or "none"}
    if draft == "cellpose":
        record["draft_nuclei_diameter_um"] = nuclei_diameter
    elif draft:
        record["draft"] = str(Path(draft).resolve())
    return labels, record


def annotate(img, channel_names, nuc_index, scale, gt_path, record, annotator,
             draft=None, nuclei_diameter=None, restart=False, in_notebook=False):
    """Open napari on ``img`` (..., C) and save the painted labels to ``gt_path``.

    ``annotator`` (a name) is recorded with the start and save time of this
    session in the ``sessions`` list of ``<gt_path>.json``; resuming an
    annotation adds a session. ``in_notebook`` makes the viewer block inside
    Jupyter until it is closed, as the notebook's own napari windows do.
    """
    annotator = str(annotator or "").strip()
    if not annotator:
        raise SystemExit("Give the annotator's name.")
    nuclei = img[..., nuc_index]
    labels, saved_record = initial_labels(nuclei, scale, gt_path, draft,
                                          nuclei_diameter, restart)
    record = {**saved_record, **record,
              "shape": list(nuclei.shape),
              "voxel_size_um": [float(s) for s in scale],
              "nuclear_channel": channel_names[nuc_index],
              "software": software_info()}
    session = {"annotator": annotator, "started": now(),
               "n_nuclei_at_start": int(labels.max())}
    record["sessions"] = record.get("sessions", []) + [session]
    record["annotators"] = sorted({s["annotator"] for s in record["sessions"]})

    import napari

    if in_notebook:
        # The viewer only blocks when ipy_interactive is off when it is created.
        from napari.settings import get_settings
        from helpers.notebook_helpers import _close_all_napari_viewers, _run_blocking_viewer
        settings = get_settings()
        was_ipy_interactive = settings.application.ipy_interactive
        settings.application.ipy_interactive = False
        _close_all_napari_viewers(napari)
    try:
        viewer = napari.Viewer(title=f"Ground truth - {gt_path.stem}")
        for c in range(img.shape[-1]):
            data = img[..., c]
            low, high = np.percentile(data, (1, 99.8))
            viewer.add_image(data, name=channel_names[c], scale=scale, blending="additive",
                             contrast_limits=(float(low), float(max(high, low + 1))),
                             colormap="gray" if c == nuc_index else "magenta",
                             visible=c == nuc_index)
        layer = viewer.add_labels(labels, name=LAYER_NAME, scale=scale, opacity=0.5)
        layer.mode = "paint"
        layer.selected_label = int(labels.max()) + 1
        if nuclei.ndim == 3:
            layer.n_edit_dimensions = 2  # paint slice by slice

        @viewer.bind_key("Shift-S", overwrite=True)
        def _save(_viewer):
            save_labels(layer.data, gt_path, record, session)

        print("\nPaint one label per nucleus. M = new label, Shift+S = save, "
              "closing the window saves too.")
        if in_notebook:
            _run_blocking_viewer(viewer, napari)
        else:
            napari.run()
    finally:
        if in_notebook:
            settings.application.ipy_interactive = was_ipy_interactive
    save_labels(layer.data, gt_path, record, session)
    return gt_path


def annotate_from_notebook(img, input_file, roi, file_meta, stain_dict, voxel_size,
                           annotator, out="ground_truth", draft=None,
                           nuclei_diameter=None, restart=False):
    """Annotate the NucleoSeg3D notebook's ROI on the original voxel grid.

    The nuclear channel is the stain table's NUCLEI channel (``stain_dict``
    from Cell 6). ``img`` is the notebook's (Z, Y, X, C) ROI as loaded, before
    any zoom, and ``voxel_size`` its (Z, Y, X) voxel size, (r_Z, r_Y, r_X):
    the ground truth is the same as the command line would give, and
    evaluate_segmentation.py maps the pipeline's resampled labels onto it.
    """
    channel_names = list(file_meta.get("channels") or [])
    nuclear_entry = {str(k).upper(): v for k, v in stain_dict.items()}.get("NUCLEI")
    if nuclear_entry is None:
        raise ValueError("The stain table has no NUCLEI row.")
    nuclear = str(nuclear_entry[1]).strip().upper()  # [marker, channel, color]
    hits = [i for i, name in enumerate(channel_names) if str(name).strip().upper() == nuclear]
    if not hits:
        raise ValueError(f"NUCLEI channel '{nuclear}' not in the file's channels "
                         f"{channel_names}.")
    nuc_index = hits[0]

    img = np.asarray(img)  # possibly a dask array before this
    channel_names += [f"Channel {i + 1}" for i in range(len(channel_names), img.shape[-1])]
    scale = tuple(float(v) for v in voxel_size)
    if img.shape[0] == 1:
        img, scale = img[0], scale[1:]

    record = {
        "source_file": source_file_info(input_file),
        "roi_x0_x1_y0_y1_z0_z1": list(roi),
    }
    gt_path = Path(out) / f"{_image_stem(input_file)}.tif"
    return annotate(img, channel_names, nuc_index, scale, gt_path, record, annotator,
                    draft=draft, nuclei_diameter=nuclei_diameter, restart=restart,
                    in_notebook=True)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate ground-truth nuclei in napari for evaluate_segmentation.py.")
    parser.add_argument("image", help="microscopy file (.nd2, .tif, .czi, .lif, .oib)")
    parser.add_argument("--roi", type=int, nargs=6, default=[0, 0, 0, 0, 0, 0],
                        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                        help="same ROI as in the notebook; 0 = full range (default)")
    parser.add_argument("--channel", help="nuclear channel: index or part of its name "
                                          "(default: first channel)")
    parser.add_argument("--annotator", required=True,
                        help="name of the person annotating (saved in the .json)")
    parser.add_argument("--out", default="ground_truth", help="output folder")
    parser.add_argument("--draft", help="'cellpose' or a label TIFF to start from")
    parser.add_argument("--nuclei-diameter", type=float,
                        help="approximate nucleus diameter in um (for --draft cellpose)")
    parser.add_argument("--restart", action="store_true",
                        help="ignore an existing annotation and start again")
    args = parser.parse_args()

    meta, img, r_X, r_Y, r_Z, file_meta, _, _ = load_image_and_metadata(args.image, args.roi)
    img = np.asarray(img)  # (Z, Y, X, C), possibly a dask array before this
    channel_names = list(file_meta.get("channels") or [])
    channel_names += [f"Channel {i + 1}" for i in range(len(channel_names), img.shape[-1])]
    nuc_index = pick_channel(args.channel, channel_names, img.shape[-1])

    if img.shape[0] == 1:
        img = img[0]
        scale = (r_Y, r_X)
    else:
        scale = (r_Z, r_Y, r_X)

    record = {
        "source_file": source_file_info(args.image),
        "roi_x0_x1_y0_y1_z0_z1": list(args.roi),
    }
    gt_path = Path(args.out) / f"{_image_stem(args.image)}.tif"
    annotate(img, channel_names, nuc_index, scale, gt_path, record, args.annotator,
             draft=args.draft, nuclei_diameter=args.nuclei_diameter, restart=args.restart)


if __name__ == "__main__":
    main()
