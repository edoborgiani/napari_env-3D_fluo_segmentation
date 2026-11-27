from collections import defaultdict
import gc
import json
import os
import re
import sys

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from PIL import Image as PILImage
from scipy import ndimage as ndi
from skimage.draw import line as draw_line
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from skimage.segmentation import watershed


__all__ = [
    "apply_contrast_gamma_per_channel",
    "apply_gaussian_smoothing",
    "apply_histogram_equalization_per_channel",
    "apply_median_denoise",
    "apply_per_channel_filter",
    "apply_threshold_per_channel",
    "assign_channel_labels",
    "assign_labels",
    "build_full_labels_df",
    "build_full_labels_dict",
    "build_histogram_report",
    "complete_skipped_steps",
    "reset_processing_steps",
    "build_labels_df",
    "build_labels_dict",
    "collect_histogram_data",
    "compute_full_marker_stats_for_marker",
    "compute_marker_stats_for_marker",
    "compute_nuclei_cytoplasm_stats",
    "compute_percell_marker_intensity_df",
    "contr_limit",
    "contr_stretch",
    "create_row_pdf",
    "crop_nucleus_with_padding",
    "detect_peaks_xy_intensity_aware",
    "detect_peaks_xy_with_best_z",
    "double_plateau_hist_equalization_nd",
    "export_channel_histograms",
    "build_vtk_volumes",
    "export_fea_mesh",
    "export_marker_stl",
    "export_nucleus_vtk_crop",
    "export_quantification_to_excel",
    "extract_roi_from_metadata",
    "fix_image_axes_order",
    "gamma_trans",
    "get_cyto_split_config",
    "get_nuclei_split_config",
    "get_stain_name",
    "grow_labels",
    "expand_labels_um",
    "grow_markers_within_islands_limited",
    "hist_plot",
    "hist_plot_with_thresholds",
    "ImageProcessing",
    "labels_dict_to_dataframe",
    "make_anisotropic_footprint",
    "merge_small_touching_labels",
    "merge_touching_labels",
    "napari_gamma",
    "napari_contrast_gamma_uint8",
    "normalize_image_channels",
    "initialize_dataset",
    "load_image_and_metadata",
    "load_image_with_roi",
    "open_image_file",
    "select_roi_interactively",
    "plot_marker_intensity_clouds",
    "plot_nucleus_kdes",
    "plot_size_distributions",
    "plot_spatial_distributions",
    "build_stain_dataframe",
    "prepare_and_preview",
    "prepare_image_stack",
    "prepare_stain_settings",
    "view_original_channels",
    "view_processing_results",
    "print_population_summary",
    "read_file_metadata",
    "remove_small_island_labels",
    "remove_small_islands",
    "resample_to_isotropic",
    "save_merged_figure",
    "save_raw_png",
    "save_single_channel_png",
    "detect_aggregates",
    "segment_nuclei",
    "segment_nuclei_cellpose",
    "segment_nuclei_watershed",
    "refine_oversized_nuclei",
    "merge_undersized_nuclei",
    "estimate_reference_nucleus_size",
    "refine_nuclei_by_size",
    "segment_cytoplasm",
    "segment_cytoplasm_cellpose",
    "segment_pcm",
    "set_notebook_context",
    "shrink_to_markers",
    "shrink_to_markers_robust",
    "stardist3d_from_2d",
    "truncate_cell",
    "voxel_volume",
    "watershed_nuclei",
    "reload_helpers",
    "run_normalize",
    "run_resample",
    "run_denoise",
    "run_contrast_gamma",
    "run_smooth",
    "run_equalize",
    "run_threshold",
    "build_cytoplasm_mask",
    "BatchCompleted",
    "get_batch_output_dir",
    "is_batch_input",
    "list_image_files",
    "print_channel_list",
    "edit_stain_dict",
    "StainDictEditor",
    "StainTableNotConfirmed",
    "run_batch_folder",
    # Splitting improvements (LD v1.2.1, nuclei v1.6.3)
    "apply_split_gamma",
    "absorb_small_split_pieces",
    # LIVE/DEAD notebook (NucleoSeg3D_LD, since v1.2.1)
    "LDStainDictEditor",
    "edit_ld_stain_dict",
    "segment_ld_nuclei",
    "segment_ld_cells",
    "classify_live_dead",
    "view_ld_results",
    "print_ld_summary",
    "plot_live_dead_classification",
    "plot_ld_spatial_distributions",
    "plot_ld_size_distributions",
    "export_live_dead_to_excel",
    "run_ld_batch_folder",
    # LIVE/DEAD notebook v1.2.2: separate LIVE / DEAD files, validated settings
    "combine_live_dead_inputs",
    "ld_nuclear_conditions",
    "ld_calibrated_nuclear_contrast",
    "ld_histogram_features",
    "ld_adapt_to_histograms",
    "ld_adapt_settings_to_image",
    "ld_remove_nuclear_haze",
    "ld_correct_depth",
    "adapt_scale_2d",
    "ld_binary_threshold",
    "ld_role",
    "ld_call_conditions",
]


def reload_helpers():
    """Reload this module and re-import all public names into the caller's globals."""
    import importlib
    import helpers.notebook_helpers as _mod
    importlib.reload(_mod)
    import sys
    caller_globals = sys._getframe(1).f_globals
    for name in _mod.__all__:
        caller_globals[name] = getattr(_mod, name)
    print("Helper functions loaded from helpers/notebook_helpers.py")


class _ND2ReaderFallback:
    """Minimal AICSImage-compatible wrapper for legacy ND2 files that fail with
    'Invalid ChunkMap signature'.  Uses nd2reader (PIMS-based) under the hood,
    which handles older/non-standard Nikon file formats.
    """

    class _PixelSizes:
        def __init__(self, x, y, z):
            self.X = x
            self.Y = y
            self.Z = z

    def __init__(self, path: str):
        from nd2reader import ND2Reader

        self._path = path
        with ND2Reader(path) as f:
            sizes = f.sizes
            c = sizes.get('c', 1)
            z = sizes.get('z', 1)
            y = sizes.get('y', f.height)
            x = sizes.get('x', f.width)
            self.shape = (1, c, z, y, x)

            # Physical pixel sizes
            px_um = f.metadata.get('pixel_microns', 1.0) or 1.0
            z_step = f.metadata.get('z_step', 1.0) or 1.0
            self._pixel_sizes = self._PixelSizes(px_um, px_um, z_step)

            # Load full data into memory as ZYXC
            f.bundle_axes = 'zyx'
            f.iter_axes = 'c'
            data_czyx = np.stack([np.array(f[i]) for i in range(c)], axis=0)
        # CZYX -> ZYXC
        self._data_zyxc = np.moveaxis(data_czyx, 0, -1)

    @property
    def physical_pixel_sizes(self):
        return self._pixel_sizes

    def get_image_data(self, dim_order="ZYXC", T=0):
        if dim_order == "ZYXC":
            return self._data_zyxc
        raise NotImplementedError(
            f"_ND2ReaderFallback: dim_order '{dim_order}' is not supported"
        )

    def get_image_dask_data(self, dim_order="ZYXC"):
        import dask.array as da
        if dim_order == "ZYXC":
            return da.from_array(self._data_zyxc, chunks=self._data_zyxc.shape)
        raise NotImplementedError(
            f"_ND2ReaderFallback: dim_order '{dim_order}' is not supported"
        )


class _PhysicalPixelSizesOverride:
    """Drop-in replacement for aicsimageio's PhysicalPixelSizes namedtuple,
    holding manually-entered values for axes the file metadata couldn't provide.
    """

    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z


class _MetaWithVoxelSizeOverride:
    """Wraps an image-metadata object (AICSImage or _ND2ReaderFallback) so that
    ``physical_pixel_sizes`` returns manually-entered values, while every other
    attribute/method (shape, channel_names, metadata, get_image_data, ...) is
    forwarded to the original object unchanged.

    Needed because some helpers (e.g. ``resample_to_isotropic``) re-read
    ``meta.physical_pixel_sizes`` directly rather than reusing the r_X/r_Y/r_Z
    values returned by ``load_image_and_metadata`` — without this wrapper, a
    manually-entered voxel size would be lost again later in the pipeline.
    """

    def __init__(self, meta, r_X, r_Y, r_Z):
        self._wrapped_meta = meta
        self._override = _PhysicalPixelSizesOverride(r_X, r_Y, r_Z)

    @property
    def physical_pixel_sizes(self):
        return self._override

    def __getattr__(self, name):
        return getattr(self._wrapped_meta, name)


def _settings_float(value):
    """Parse a number from an Olympus settings value (may be a quoted string)."""
    try:
        return float(str(value).strip().strip('"'))
    except (TypeError, ValueError):
        return None


class _OifFileReader:
    """Minimal AICSImage-compatible wrapper for Olympus .oib files.

    aicsimageio can only open Olympus files through Bio-Formats (which needs
    Java), so they are read with the pure-Python ``oiffile`` package instead.
    The whole file is loaded into memory as ZYXC; any extra axis (e.g. time)
    is reduced to its first index. Voxel sizes and channel names are read
    from the file's settings on a best-effort basis -- a voxel size that can't
    be found is left as None, so ``_prompt_missing_voxel_sizes`` asks for it.
    """

    def __init__(self, path: str):
        import oiffile

        oif = oiffile.OifFile(path)
        try:
            data = np.asarray(oif.asarray())
            axes = str(getattr(oif, 'axes', '') or '').upper()
            mainfile = oif.mainfile
        finally:
            oif.close()

        self._data_zyxc = self._to_zyxc(data, axes)
        z, y, x, c = self._data_zyxc.shape
        self.shape = (1, c, z, y, x)
        self.dtype = self._data_zyxc.dtype
        self.channel_names = self._channel_names(mainfile, c)
        self._pixel_sizes = _PhysicalPixelSizesOverride(*self._voxel_sizes(mainfile))

    @staticmethod
    def _to_zyxc(data, axes):
        if len(axes) != data.ndim:
            raise ValueError(
                f"_OifFileReader: can't interpret axes '{axes}' for an array of shape {data.shape}"
            )
        axes = list(axes)
        # Keep only Z/Y/X/C: take the first index of any other axis (e.g. T).
        for i in reversed(range(len(axes))):
            if axes[i] not in "ZYXC":
                data = np.take(data, 0, axis=i)
                del axes[i]
        for missing in "ZC":
            if missing not in axes:
                data = data[np.newaxis]
                axes.insert(0, missing)
        return np.transpose(data, [axes.index(a) for a in "ZYXC"])

    @staticmethod
    def _channel_names(mainfile, n_channels):
        names = []
        for i in range(1, n_channels + 1):
            section = mainfile.get(f"Channel {i} Parameters", {}) or {}
            name = str(section.get("DyeName", "")).strip().strip('"')
            names.append(name or f"Channel {i}")
        return names

    @staticmethod
    def _voxel_sizes(mainfile):
        """Return (X, Y, Z) in um; None for any value that can't be read."""
        ref = mainfile.get("Reference Image Parameter", {}) or {}
        r_x = _settings_float(ref.get("WidthConvertValue"))
        r_y = _settings_float(ref.get("HeightConvertValue"))
        r_z = None
        for section_name, section in mainfile.items():
            if not (section_name.startswith("Axis ") and section_name.endswith("Parameters Common")):
                continue
            if str(section.get("AxisCode", "")).strip().strip('"').upper() != "Z":
                continue
            start = _settings_float(section.get("StartPosition"))
            end = _settings_float(section.get("EndPosition"))
            n = _settings_float(section.get("MaxSize"))
            if start is None or end is None or not n or n < 2:
                break
            step = abs(end - start) / (n - 1)
            unit = str(section.get("UnitName", "")).strip().strip('"').lower()
            if unit == "nm":
                step /= 1000.0
            elif unit == "mm":
                step *= 1000.0
            r_z = step
            break
        return tuple(v if (v is not None and v > 0) else None for v in (r_x, r_y, r_z))

    @property
    def physical_pixel_sizes(self):
        return self._pixel_sizes

    def get_image_data(self, dim_order="ZYXC", T=0):
        if dim_order == "ZYXC":
            return self._data_zyxc
        raise NotImplementedError(
            f"_OifFileReader: dim_order '{dim_order}' is not supported"
        )

    def get_image_dask_data(self, dim_order="ZYXC"):
        import dask.array as da
        if dim_order == "ZYXC":
            return da.from_array(self._data_zyxc, chunks=self._data_zyxc.shape)
        raise NotImplementedError(
            f"_OifFileReader: dim_order '{dim_order}' is not supported"
        )


class _PlainTiffReader:
    """AICSImage-compatible reader for plain (non-OME) TIFF files.

    Without OME metadata aicsimageio guesses the axes, and reads a 2D
    two-channel picture stored as a (2, Y, X) array as a 2-plane Z stack
    with one channel. Here the axes come from the array itself:

    - axes stated in the file (ImageJ hyperstack 'C', or RGB samples 'S')
      are used as they are;
    - otherwise, after dropping singleton axes: 2 dims = (Y, X), one channel;
      **3 dims = 2D image + channels** (the extra axis is the channel axis);
      **4 dims = 3D stack + channels** -- of the two extra axes the smaller
      is taken as the channel axis (on a tie, the second one, as in ImageJ's
      Z, C, Y, X order).

    Extra axes (e.g. time) are reduced to their first index. Voxel sizes come
    from aicsimageio when it can read them; a missing one is asked for later
    (``_prompt_missing_voxel_sizes``). The image is read fully into memory.
    """

    def __init__(self, path: str):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            data = np.asarray(series.asarray())
            axes = str(series.axes).upper()
            labels = (tif.imagej_metadata or {}).get('Labels') if tif.is_imagej else None

        self._data_zyxc = self._to_zyxc(data, axes)
        z, y, x, c = self._data_zyxc.shape
        self.shape = (1, c, z, y, x)
        self.dtype = self._data_zyxc.dtype
        if labels and len(labels) == c:
            self.channel_names = [str(l) for l in labels]
        else:
            self.channel_names = [f"Channel {i + 1}" for i in range(c)]

        r_x = r_y = r_z = None
        try:
            from aicsimageio import AICSImage
            sizes = AICSImage(path).physical_pixel_sizes
            r_x, r_y = sizes.X, sizes.Y
            r_z = sizes.Z if z > 1 else None
        except Exception as exc:
            print(f"[_PlainTiffReader] Could not read the voxel size ({exc}).")
        self._pixel_sizes = _PhysicalPixelSizesOverride(r_x, r_y, r_z)
        print(f"[_PlainTiffReader] TIFF axes '{axes}' {tuple(data.shape)} -> "
              f"{'2D' if z == 1 else '3D'} image, {c} channel(s), (Z, Y, X, C) = {self._data_zyxc.shape}")

    @staticmethod
    def _to_zyxc(data, axes):
        if len(axes) != data.ndim:
            axes = 'Q' * (data.ndim - 2) + 'YX'
        axes = list(axes)
        # Channel axis stated in the file: ImageJ 'C' or RGB samples 'S'.
        if 'S' in axes and 'C' not in axes:
            axes[axes.index('S')] = 'C'
        # Drop singleton axes (except Y/X), keep the first index of time.
        for i in reversed(range(len(axes))):
            if axes[i] in 'YX':
                continue
            if data.shape[i] == 1 or axes[i] == 'T':
                data = np.take(data, 0, axis=i)
                del axes[i]
        extra = [i for i, a in enumerate(axes) if a not in 'YX']
        if 'C' not in axes:
            # No channel axis in the metadata: decide from the dimensions.
            if len(extra) == 1:        # 2D + C
                axes[extra[0]] = 'C'
            elif len(extra) == 2:      # 3D + C: smaller extra axis = channels
                a, b = extra
                c_axis = a if data.shape[a] < data.shape[b] else b
                z_axis = b if c_axis == a else a
                axes[c_axis], axes[z_axis] = 'C', 'Z'
            elif len(extra) > 2:
                raise ValueError(
                    f"_PlainTiffReader: can't interpret a {data.ndim}D TIFF "
                    f"{data.shape} as (Z, Y, X, C); save it as OME-TIFF."
                )
        else:
            others = [i for i in extra if axes[i] != 'C']
            if len(others) > 1:
                raise ValueError(
                    f"_PlainTiffReader: TIFF axes {''.join(axes)} {data.shape} have "
                    "more than one non-channel axis besides Y/X; save it as OME-TIFF."
                )
            for i in others:
                axes[i] = 'Z'
        for missing in 'ZC':
            if missing not in axes:
                data = data[np.newaxis]
                axes.insert(0, missing)
        return np.transpose(data, [axes.index(a) for a in 'ZYXC'])

    @property
    def physical_pixel_sizes(self):
        return self._pixel_sizes

    def get_image_data(self, dim_order="ZYXC", T=0):
        if dim_order == "ZYXC":
            return self._data_zyxc
        raise NotImplementedError(
            f"_PlainTiffReader: dim_order '{dim_order}' is not supported"
        )

    def get_image_dask_data(self, dim_order="ZYXC"):
        import dask.array as da
        if dim_order == "ZYXC":
            return da.from_array(self._data_zyxc, chunks=self._data_zyxc.shape)
        raise NotImplementedError(
            f"_PlainTiffReader: dim_order '{dim_order}' is not supported"
        )


# LIVE/DEAD v1.2.2: a LIVE and a DEAD image acquired as two separate files are
# read as one multichannel image through a small JSON "pair file" (written by
# `combine_live_dead_inputs`). The pair file stands in for an image file
# everywhere (loading, ROI, channel list, batch, output names), so the rest of
# the pipeline does not need to know the channels come from two files.
LD_PAIR_EXTENSION = ".ldpair"


class _PairedImageReader:
    """AICSImage-compatible reader for a LIVE/DEAD pair file (``.ldpair``).

    The pair file lists the image files (LIVE first, then DEAD). Every file
    is opened with `open_image_file`; their channels are stacked along C, in
    that order, each named ``'<LABEL> - <channel name in its file>'`` (e.g.
    ``'LIVE - EGFP'``, ``'DEAD - TRITC'``), so the names stay unique even when
    both files call their channel 'Channel 1'. The files must have the same
    Z, Y, X size and voxel size (1% tolerance; a voxel size missing in one
    file is taken from the other).
    """

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        entries = record.get("files") or []
        if len(entries) < 2:
            raise ValueError(f"{path}: a LIVE/DEAD pair file needs at least two image files.")
        base = os.path.dirname(os.path.abspath(path))
        self.files, self.labels, self._metas = [], [], []
        names, dates = [], []
        for entry in entries:
            file_path = entry["path"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(base, file_path)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"{path}: image file not found: {file_path}")
            label = str(entry.get("label") or f"FILE {len(self.files) + 1}").upper()
            meta = open_image_file(file_path)
            n_c = int(meta.shape[1])
            try:
                file_meta = read_file_metadata(file_path, meta)
            except Exception as exc:
                print(f"[pair] Could not read the metadata of {os.path.basename(file_path)} ({exc}).")
                file_meta = {"date": None, "channels": None}
            ch = list(file_meta.get("channels") or [])
            if len(ch) != n_c:
                ch = list(getattr(meta, "channel_names", None) or [])
            if len(ch) != n_c:
                ch = [f"Channel {i + 1}" for i in range(n_c)]
            names += [f"{label} - {c}" for c in ch]
            dates.append(file_meta.get("date"))
            self.files.append(file_path)
            self.labels.append(label)
            self._metas.append(meta)

        zyx = [tuple(int(v) for v in m.shape[2:5]) for m in self._metas]
        if len(set(zyx)) > 1:
            raise ValueError(
                "The LIVE and DEAD images must have the same size to be combined; (Z, Y, X) = "
                + ", ".join(f"{l} {s}" for l, s in zip(self.labels, zyx)))
        sizes = []
        for axis in "XYZ":
            values = [getattr(m.physical_pixel_sizes, axis) for m in self._metas]
            known = [float(v) for v in values if v]
            if known and (max(known) - min(known)) > 0.01 * max(known):
                raise ValueError(
                    f"The LIVE and DEAD images have different {axis} voxel sizes ("
                    + ", ".join(f"{l} {v}" for l, v in zip(self.labels, values))
                    + " um): they can't be combined voxel by voxel.")
            sizes.append(known[0] if known else None)
        self._pixel_sizes = _PhysicalPixelSizesOverride(*sizes)
        self.channel_names = names
        self.shape = (1, len(names)) + zyx[0]
        self.dtype = np.result_type(*[np.dtype(getattr(m, "dtype", np.uint16)) for m in self._metas])
        self.date = next((d for d in dates if d), None)
        print(f"[pair] {len(self.files)} files combined into one {len(names)}-channel image "
              f"(Z, Y, X) = {zyx[0]}:")
        for label, file_path in zip(self.labels, self.files):
            print(f"  {label}: {file_path}")

    @property
    def physical_pixel_sizes(self):
        return self._pixel_sizes

    def get_image_data(self, dim_order="ZYXC", T=0):
        if dim_order != "ZYXC":
            raise NotImplementedError(f"_PairedImageReader: dim_order '{dim_order}' is not supported")
        return np.concatenate(
            [np.asarray(m.get_image_data("ZYXC", T=0)).astype(self.dtype, copy=False)
             for m in self._metas], axis=-1)

    def get_image_dask_data(self, dim_order="ZYXC"):
        import dask.array as da
        if dim_order != "ZYXC":
            raise NotImplementedError(f"_PairedImageReader: dim_order '{dim_order}' is not supported")
        parts = []
        for m in self._metas:
            part = _lazy_image_data(m)
            if isinstance(part, np.ndarray):
                part = da.from_array(part, chunks=part.shape)
            parts.append(part.astype(self.dtype))
        return da.concatenate(parts, axis=-1)


def _is_plain_tiff(input_file):
    """True for a .tif/.tiff file without OME metadata."""
    if not str(input_file).lower().endswith(('.tif', '.tiff')):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(str(input_file)) as tif:
            return not tif.is_ome
    except Exception:
        return False


def _prompt_missing_voxel_sizes(r_X, r_Y, r_Z):
    """Prompt for any voxel size axis the file metadata didn't provide.

    aicsimageio (and the nd2reader fallback) return None for any axis whose
    physical pixel size couldn't be read from the file. Downstream steps
    (isotropic resampling, volume/diameter calculations, mesh export) all
    assume real values, so ask for the missing ones here — a text box will
    appear below the cell — instead of failing later with a cryptic
    ``NoneType`` error.

    Returns
    -------
    r_X, r_Y, r_Z : float
    was_prompted : bool
        True if any value had to be entered manually.
    """
    axes = {"X": r_X, "Y": r_Y, "Z": r_Z}
    missing = [name for name, value in axes.items() if value is None]
    if not missing:
        return r_X, r_Y, r_Z, False

    print(
        f"Could not read the voxel size for axis/axes {missing} from the file "
        "metadata. Enter the physical pixel size (in micrometers) for each "
        "missing axis below."
    )
    for name in missing:
        while True:
            raw = input(f"  Voxel size for {name} (um): ").strip()
            try:
                value = float(raw)
            except ValueError:
                print(f"  '{raw}' is not a valid number, try again.")
                continue
            if value <= 0:
                print("  Voxel size must be a positive number, try again.")
                continue
            axes[name] = value
            break

    return axes["X"], axes["Y"], axes["Z"], True


def open_image_file(input_file: str):
    """Open a microscopy file with AICSImage, falling back to a nd2reader-based
    wrapper for legacy ND2 files that raise ``ValueError: Invalid ChunkMap signature``.
    Olympus .oib files are opened with ``_OifFileReader`` (oiffile) instead.

    Supported: .nd2, .tif/.tiff (incl. .ome.tif), .czi (aicspylibczi),
    .lif (readlif), .oib (oiffile). For multi-scene .czi/.lif files only the
    first scene is read.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file.

    Returns
    -------
    meta : AICSImage, _ND2ReaderFallback or _OifFileReader
        Object with the same interface used by the notebook cells
        (``shape``, ``physical_pixel_sizes``, ``get_image_data``,
        ``get_image_dask_data``).
    """
    if str(input_file).lower().endswith(LD_PAIR_EXTENSION):
        return _PairedImageReader(str(input_file))

    if str(input_file).lower().endswith('.oib'):
        return _OifFileReader(str(input_file))

    if _is_plain_tiff(input_file):
        return _PlainTiffReader(str(input_file))

    if str(input_file).lower().endswith('.czi'):
        import importlib.util
        if importlib.util.find_spec('aicspylibczi') is None:
            import sys
            raise ImportError(
                f"Cannot read '{input_file}': .czi files need aicspylibczi, which "
                f"is not installed (it has no wheels for Python "
                f"{sys.version_info[0]}.{sys.version_info[1]}). Use Python "
                "3.10-3.13 for .czi files, or export the image to OME-TIFF."
            )

    from aicsimageio import AICSImage
    try:
        return AICSImage(input_file)
    except ValueError as exc:
        if 'ChunkMap' not in str(exc) or not input_file.lower().endswith('.nd2'):
            raise
        print(
            f"[open_image_file] AICSImage raised '{exc}'.\n"
            "Falling back to nd2reader for legacy ND2 file — note: this "
            "fallback always loads the full image into memory, regardless "
            "of the automatic lazy-loading decision made for other formats."
        )
        return _ND2ReaderFallback(input_file)


def load_image_with_roi(input_file: str, roi_coords):
    """Open a microscopy file and load the requested ROI in ZYXC order.

    Automatically loads lazily (dask, ROI-only) or eagerly depending on the
    file's full size — see ``extract_roi_from_metadata``.
    """
    meta = open_image_file(input_file)
    image, _, used_lazy_loading = extract_roi_from_metadata(meta, roi_coords)
    print(f"ROI of the image (Z, Y, X, C): {image.shape}")
    print(f"Lazy (chunked) loading used: {used_lazy_loading}")
    return meta, image, used_lazy_loading


def load_image_and_metadata(input_file, roi_coords):
    """Load image with ROI, read voxel sizes and file metadata.

    Combines ``load_image_with_roi`` and ``read_file_metadata`` into a
    single call so the notebook cell stays clean.

    Returns
    -------
    meta : AICSImage
    img : ndarray (Z, Y, X, C)
    r_X, r_Y, r_Z : float
        Physical voxel sizes in micrometers.
    file_meta : dict
        Keys ``'date'`` and ``'channels'``.
    ROI_print : list
        Copy of the ROI coordinates for display in reports.
    used_lazy_loading : bool
        Whether the dask/chunked read path was used.
    """
    ROI_print = list(roi_coords)
    meta, img, used_lazy_loading = load_image_with_roi(input_file, roi_coords)
    r_X = meta.physical_pixel_sizes.X
    r_Y = meta.physical_pixel_sizes.Y
    r_Z = meta.physical_pixel_sizes.Z
    is_2d = img.shape[0] == 1
    if is_2d:
        # Single plane (2D file, or a one-slice Z ROI): there is no Z step.
        # Give the plane a 1 um thickness, so every "volume" computed
        # downstream (voxel count x r_X x r_Y x r_Z) is the area in um2.
        r_Z = 1.0
    r_X, r_Y, r_Z, was_prompted = _prompt_missing_voxel_sizes(r_X, r_Y, r_Z)
    if was_prompted or is_2d:
        meta = _MetaWithVoxelSizeOverride(meta, r_X, r_Y, r_Z)
    if is_2d:
        print("2D image (single plane): Z voxel size set to 1 um, so volumes "
              "[um3] are areas [um2].")
    file_meta = read_file_metadata(input_file, meta)
    print(f"Voxel size: [{r_X}, {r_Y}, {r_Z}] um")
    print(f"Date: {file_meta['date']}")
    print(f"Channels: {file_meta['channels']}")
    return meta, img, r_X, r_Y, r_Z, file_meta, ROI_print, used_lazy_loading


def _saved_stain_display(stain_table_path):
    """``{file channel name: (layer name, color)}`` from a saved stain table
    (``<name_setup>_stain_dict.json`` or ``<name_setup>_ld_stain_dict.json``),
    or ``{}`` if there is none. Only used to label the preview: the table
    itself is still edited and confirmed in the stain editor."""
    if not stain_table_path or not os.path.exists(stain_table_path):
        return {}
    try:
        with open(stain_table_path, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
    except Exception as exc:
        print(f"[preview] Could not read {stain_table_path} ({exc}); "
              "using the file's channel names.")
        return {}
    display = {}
    for row in rows:
        channel = row.get("channel", _STAIN_NO_CHANNEL)
        if channel == _STAIN_NO_CHANNEL:
            continue
        condition = str(row.get("condition", "")).upper()
        marker = str(row.get("marker", "")).strip()
        name = f"{condition} ({marker})" if marker else condition
        display[str(channel)] = (name, str(row.get("color", "white")).lower())
    if display:
        print(f"Preview channels named and coloured from the saved stain table "
              f"{stain_table_path}.")
    return display


def select_roi_interactively(input_file, roi_coords, napari_module=None,
                             stain_table_path=None):
    """Open a napari window on the full image so the ROI can be set by
    dragging a rectangle instead of typing pixel coordinates.

    Every channel of the *full* image is loaded lazily (via dask — the file
    is opened independently of the main pipeline load, and no more of it is
    read than napari actually renders) and shown as a separate layer, named
    from the file metadata. If a saved stain table exists at
    ``stain_table_path``, the channels it uses are named after their
    condition and marker (e.g. 'LIVE (CALCEIN)') and shown in its colors;
    the table can still be changed in the stain editor afterwards. A yellow 'ROI' rectangle seeded from
    ``roi_coords`` is added on top; drag its edges/corners to the region you
    want, then close the window to continue.

    Only X/Y are interactively editable, since a 2D rectangle can't
    represent a depth range — Z is passed through from ``roi_coords``
    unchanged. Edit ROI's z0/z1 in Cell 3 directly if it needs to change.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file.
    roi_coords : list of 6 ints
        [x0, x1, y0, y1, z0, z1]. Use 0 for any limit to mean "full range" —
        this is resolved against the file's real extent before the
        rectangle is drawn.
    napari_module : module, optional
        Pass the already-imported ``napari`` module (avoids re-importing).

    Returns
    -------
    list of 6 ints
        Updated [x0, x1, y0, y1, z0, z1]: X/Y from the final rectangle
        position (clipped to the image extent), Z unchanged from the input.
    """
    if napari_module is None:
        import napari as napari_module

    meta = open_image_file(input_file)
    file_meta = read_file_metadata(input_file, meta)
    channel_names = file_meta.get("channels") or []

    # Small files are read directly; only large ones lazily (see
    # _lazy_image_data -- the lazy TIFF reader can fail on some installs).
    if _estimate_full_image_bytes(meta) > _LAZY_LOAD_THRESHOLD_BYTES:
        lazy = _lazy_image_data(meta)
    else:
        lazy = meta.get_image_data("ZYXC", T=0)
    z_size, y_size, x_size, c_size = lazy.shape

    x0, x1, y0, y1, z0, z1 = roi_coords
    x1 = x1 if x1 else x_size
    y1 = y1 if y1 else y_size
    z1 = z1 if z1 else z_size
    x0 = int(np.clip(x0, 0, x_size))
    x1 = int(np.clip(x1, 0, x_size))
    y0 = int(np.clip(y0, 0, y_size))
    y1 = int(np.clip(y1, 0, y_size))

    # napari.run() only blocks when ipy_interactive is off *at the time the
    # Viewer is created* — otherwise Qt is already integrated with IPython's
    # event loop and the viewer never actually waits, so the window closes
    # again as soon as this function keeps executing. Must be set before
    # napari_module.Viewer(...) is constructed, not just before .run().
    from napari.settings import get_settings
    settings = get_settings()
    was_ipy_interactive = settings.application.ipy_interactive
    settings.application.ipy_interactive = False
    try:
        _close_all_napari_viewers(napari_module)
        viewer = napari_module.Viewer(
            title="Drag the ROI rectangle, then close this window to continue",
            ndisplay=2,
        )

        default_colors = ["gray", "red", "green", "blue", "magenta", "cyan", "yellow"]
        saved = _saved_stain_display(stain_table_path)
        for c in range(c_size):
            name = channel_names[c] if c < len(channel_names) else f"Channel {c}"
            colormap = default_colors[c % len(default_colors)]
            if name in saved:
                name, colormap = saved[name]
            viewer.add_image(
                lazy[:, :, :, c],
                name=name,
                colormap=colormap,
                blending="additive",
            )

        roi_rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)
        shapes_layer = viewer.add_shapes(
            [roi_rect],
            shape_type="rectangle",
            name="ROI",
            edge_color="yellow",
            face_color="transparent",
            edge_width=max(2, round(0.004 * max(x_size, y_size))),
        )
        # Make the ROI shape immediately draggable/resizable — otherwise the
        # viewer opens in pan/zoom mode with nothing selected, and the user
        # has to click the "select shapes" tool and then click the rectangle
        # by hand before its resize handles appear.
        viewer.layers.selection.active = shapes_layer
        shapes_layer.mode = "select"
        shapes_layer.selected_data = {0}

        print(
            "Drag the corners/edges of the yellow 'ROI' rectangle to the region "
            "you want to process, then close the napari window to continue.\n"
            "The Z range from Cell 3 is kept as-is (not editable here)."
        )
        _run_blocking_viewer(viewer, napari_module)
    finally:
        settings.application.ipy_interactive = was_ipy_interactive

    if len(shapes_layer.data) == 0:
        print("ROI rectangle was deleted — keeping the original X/Y bounds.")
        new_x0, new_x1, new_y0, new_y1 = x0, x1, y0, y1
    else:
        verts = shapes_layer.data[0]
        new_y0 = int(np.clip(round(float(np.min(verts[:, 0]))), 0, y_size))
        new_y1 = int(np.clip(round(float(np.max(verts[:, 0]))), 0, y_size))
        new_x0 = int(np.clip(round(float(np.min(verts[:, 1]))), 0, x_size))
        new_x1 = int(np.clip(round(float(np.max(verts[:, 1]))), 0, x_size))

    _close_all_napari_viewers(napari_module)

    new_roi = [new_x0, new_x1, new_y0, new_y1, z0, z1]
    print(f"ROI updated interactively: {new_roi}")
    return new_roi


def initialize_dataset(input_file, roi_coords,
                       nuclei_diameter=10.0, cell_diameter=30.0,
                       scale_factor=1.0, zoom_factors=None):
    """Load image, read metadata, and compute derived parameters.

    Wraps ``load_image_and_metadata`` and adds expansion-factor and
    zoom-factor computation so the notebook cell is a single call. Whether
    the file is read lazily (dask, ROI-only) or eagerly is decided
    automatically from its full size — see ``extract_roi_from_metadata``.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file.
    roi_coords : list of 6 ints
        [x0, x1, y0, y1, z0, z1].
    nuclei_diameter, cell_diameter : float
        Approximate diameters in micrometers.
    scale_factor : float
        Extra manual resolution scaling (default 1.0).
    zoom_factors : list of 3 floats or None
        Base [Z, Y, X] zoom factors (default [1, 1, 1]).

    Returns
    -------
    meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
    cyto_factor, PCM_factor, zoom_factors, used_lazy_loading
    """
    if zoom_factors is None:
        zoom_factors = [1.0, 1.0, 1.0]

    meta, img, r_X, r_Y, r_Z, file_meta, ROI_print, used_lazy_loading = (
        load_image_and_metadata(input_file, roi_coords)
    )

    cyto_factor = int(np.round(cell_diameter / nuclei_diameter))
    # PCM_factor is no longer used (segment_pcm takes pcm_thickness_factor);
    # still returned so the notebooks' unpacking keeps working.
    PCM_factor = int(cyto_factor * 1.1)
    if PCM_factor == cyto_factor:
        PCM_factor += 1

    zoom_factors = [x * scale_factor for x in zoom_factors]
    if img.shape[0] == 1:
        # 2D image: never resample the single plane along Z.
        zoom_factors[0] = 1.0

    return (meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
            cyto_factor, PCM_factor, zoom_factors, used_lazy_loading)


def prepare_and_preview(img, nuclei_diameter, cell_diameter,
                        stain_dict, file_meta,
                        napari_module, r_xyz=None, progress=None,
                        show_viewer=True):
    """Prepare image stack, build stain table, and open napari preview.

    Combines ``prepare_image_stack``, ``build_stain_dataframe``, and
    ``view_original_channels`` into a single call.

    Any real image channel with no corresponding entry in *stain_dict* is
    dropped from ``im_final_stack['Original image']`` here (and therefore
    from every stage derived from it later in the pipeline), and the
    remaining channels are reordered to match *stain_df*'s row order. This
    keeps the invariant, relied on throughout the rest of the pipeline, that
    channel ``c`` of the image corresponds to row ``c`` of ``stain_df``.

    Parameters
    ----------
    r_xyz : tuple of float (r_X, r_Y, r_Z), optional
        Physical voxel sizes in micrometers, forwarded to the napari preview
        so the displayed volume isn't stretched/squashed along Z.
    show_viewer : bool, optional
        If False, skip the napari preview (used by batch mode); ``viewer`` is
        then returned as None.

    Returns
    -------
    im_final_stack : dict
    nuclei_radius, cell_radius, nuclei_volume, cell_volume : float
    stain_df : DataFrame
    viewer : napari.Viewer or None
    """
    im_final_stack, nuclei_radius, cell_radius, nuclei_volume, cell_volume = (
        prepare_image_stack(img, nuclei_diameter, cell_diameter)
    )
    stain_df = build_stain_dataframe(stain_dict, file_meta)

    channel_indices = stain_df['Channel_index'].to_numpy()
    n_raw_channels = im_final_stack['Original image'].shape[3]
    if len(channel_indices) != n_raw_channels:
        laser_order = file_meta.get("channels") or []
        dropped = [i for i in range(n_raw_channels) if i not in channel_indices]
        dropped_names = [laser_order[i] if i < len(laser_order) else i for i in dropped]
        print(f"[prepare_and_preview] Channel(s) not listed in stain_dict, "
              f"dropped from analysis: {dropped_names}")
    im_final_stack['Original image'] = im_final_stack['Original image'][..., channel_indices]

    stain_df = stain_df.drop(columns='Channel_index')
    viewer = None
    if show_viewer:
        viewer = view_original_channels(im_final_stack, stain_df, napari_module, r_xyz=r_xyz, progress=progress)
    return (im_final_stack, nuclei_radius, cell_radius, nuclei_volume, cell_volume,
            stain_df, viewer)


def read_file_metadata(input_file: str, meta) -> dict:
    """Return file metadata (date, channel names) for nd2, czi, tif, or other formats.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file.
    meta : AICSImage
        Already-opened AICSImage object for the file.

    Returns
    -------
    dict with keys:
        'date'     : acquisition date string, or None if not available
        'channels' : list of channel name strings
    """
    from pathlib import Path as _Path
    ext = _Path(input_file).suffix.lower()

    if ext == LD_PAIR_EXTENSION:  # LIVE/DEAD pair file (v1.2.2)
        return {"date": getattr(meta, "date", None), "channels": list(meta.channel_names)}

    if ext == '.nd2':
        try:
            from nd2reader import ND2Reader
            with ND2Reader(input_file) as nd2:
                date = nd2.metadata.get("date")
                channels = nd2.metadata.get("channels")
            return {"date": date, "channels": channels}
        except Exception as exc:
            print(f"[read_file_metadata] nd2reader fallback: {exc}")

    if ext == '.czi':
        channels = list(meta.channel_names)
        date = None
        try:
            import xml.etree.ElementTree as ET
            xml_meta = meta.metadata  # ElementTree Element for CZI
            for xpath in [
                "./Metadata/Information/Document/CreationDate",
                "./Information/Document/CreationDate",
                "./Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/CreationDate",
            ]:
                elem = xml_meta.find(xpath)
                if elem is not None and elem.text:
                    date = elem.text.strip()
                    break
        except Exception:
            pass
        return {"date": date, "channels": channels}

    # Generic fallback (tif, tiff, lif, etc.) — AICSImage provides channel_names
    channels = list(meta.channel_names)
    return {"date": None, "channels": channels}


def _image_stem(input_file):
    """File name without extension; the double extension of OME-TIFF
    ('x.ome.tif') is stripped completely, giving 'x'."""
    from pathlib import Path as _Path
    stem = _Path(input_file).stem
    if stem.lower().endswith('.ome'):
        stem = stem[:-4]
    return stem


def _stem_output_path(input_file, suffix, output_dir=None):
    """Return ``<input stem><suffix>`` inside *output_dir*, or in the current
    working directory when *output_dir* is None (single-file behaviour)."""
    from pathlib import Path as _Path
    name = _image_stem(input_file) + suffix
    return str(_Path(output_dir) / name) if output_dir is not None else name


def set_notebook_context(**kwargs):
    """Store notebook variables that some helper functions read later."""
    globals().update(kwargs)


def _close_all_napari_viewers(napari_module=None):
    """Close every open Napari viewer, tolerating stale Qt C++ objects."""
    if napari_module is None:
        try:
            import napari as napari_module
        except ImportError:
            return
    for viewer in list(napari_module.Viewer._instances):
        try:
            qt_win = viewer.window._qt_window
            if qt_win.isVisible():
                viewer.close()
        except (RuntimeError, AttributeError):
            pass
    napari_module.Viewer._instances.clear()


def _run_blocking_viewer(viewer, napari_module):
    """Block until ``viewer`` is closed by the user.

    Other platforms use ``napari.run()``. On macOS, that nested Qt loop inside
    ipykernel can leave the window frozen (spinning wheel), so a local
    QEventLoop runs instead and quits once the window is no longer visible.
    """
    if sys.platform != "darwin":
        napari_module.run()
        return

    from qtpy.QtCore import QEventLoop, QTimer

    qt_win = viewer.window._qt_window
    loop = QEventLoop()
    timer = QTimer()

    def _quit_if_closed():
        try:
            closed = not qt_win.isVisible()
        except RuntimeError:  # Qt window already deleted
            closed = True
        if closed:
            timer.stop()
            loop.quit()

    timer.timeout.connect(_quit_if_closed)
    timer.start(200)
    qt_win.raise_()
    qt_win.activateWindow()
    loop.exec()


def _context(name, default=None):
    value = globals().get(name, default)
    if value is not None:
        return value

    import inspect

    for frame_info in inspect.stack()[1:]:
        frame = frame_info.frame
        if name in frame.f_locals and frame.f_locals[name] is not None:
            return frame.f_locals[name]
        if name in frame.f_globals and frame.f_globals[name] is not None:
            return frame.f_globals[name]

    if default is not None:
        return default
    raise RuntimeError(f"Notebook context '{name}' is required but was not set.")
    return value


def gamma_trans(im_in, gamma):
    """Apply gamma correction to an image scaled to uint8."""
    val_c = 255.0 / (np.max(im_in) ** gamma)
    return (val_c * (im_in ** gamma)).copy()


def napari_gamma(image, gamma):
    """Apply Napari-style gamma while preserving the original dtype."""
    dtype = image.dtype
    img = image.astype(np.float32)
    img /= img.max() if img.max() != 0 else 1.0
    img = img ** gamma

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        img = np.clip(img * info.max, 0, info.max).astype(dtype)
    else:
        img = img.astype(dtype)

    return img


def contr_limit(im_in, c_min=None, c_max=None):
    """Stretch contrast linearly to the 0-255 display range."""
    im_in = im_in.astype(float)

    if c_min is None:
        c_min = im_in.min()
    if c_max is None:
        c_max = im_in.max()
    if c_max == c_min:
        return np.zeros_like(im_in, dtype=np.uint8)

    alpha = 255.0 / (c_max - c_min)
    beta = -c_min * alpha
    im_out = alpha * im_in + beta
    return im_out.astype(np.uint8)


def contr_stretch(im_in, c_min=None, c_max=None):
    """Mimic Fiji-style brightness and contrast stretching."""
    im_in = im_in.astype(float)

    if c_min is None:
        c_min = im_in.min()
    if c_max is None:
        c_max = im_in.max()
    if c_max == c_min:
        return np.zeros_like(im_in, dtype=np.uint8)

    norm = (im_in - c_min) / (c_max - c_min)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)


def truncate_cell(val, width=15):
    """Truncate long table values for display."""
    val_str = str(val)
    return val_str if len(val_str) <= width else val_str[: width - 3] + "..."


def get_nuclei_split_config(profile="aggressive", **overrides):
    """Return the recommended shrink-based nuclei splitting settings for the notebook.

    Parameters
    ----------
    profile : {"conservative", "balanced", "aggressive"}, optional
        Preset controlling how strongly touching nuclei are separated.
        - conservative -> less splitting, lower risk of over-segmentation
        - balanced     -> middle-ground starting point
        - aggressive   -> stronger bridge breaking for clustered nuclei
    **overrides : dict
        Any keyword arguments matching `segment_nuclei_watershed` options.

    Returns
    -------
    dict
        A parameter dictionary ready to pass into `segment_nuclei_watershed(..., **config)`.
    """
    profiles = {
        "conservative": 0.18,
        "balanced": 0.22,
        "aggressive": 0.28,
    }
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose from {tuple(profiles)}."
        )

    config = {
        "nuclei_z_anisotropy_factor": 1.0,
        "nuclei_bridge_shrink_factor": profiles[profile],
        "nuclei_split_diameter_min_factor": 0.5,
        "nuclei_split_diameter_max_factor": 1.5,
        "nuclei_split_diameter_scales": 3,
        "nuclei_seed_min_fraction": 0.03,
        "nuclei_min_roundness": 0.45,
        # z_split_aggressive: split a label whenever any single z-slice has
        # several separate islands (e.g. a large nucleus that forks into two
        # lobes above and below its central part), not only when its 3D
        # regions are fully disconnected.
        "z_split_aggressive": True,
        # Intensity-aware seeding/splitting: when two individually round
        # nuclei touch with too shallow a geometric neck for erosion alone
        # to separate, also look for two seeds in a combined distance +
        # intensity peak map, and run the final watershed on a distance +
        # intensity + gradient elevation surface instead of pure distance,
        # so the split lands where the marker signal itself dips between
        # the two nuclei. Requires `intensity_img` to be passed through to
        # `segment_nuclei_watershed` (done automatically by `segment_nuclei`).
        "split_by_intensity_gradient": True,
        "nuclei_gradient_smooth_sigma": 1.0,
        "nuclei_distance_weight": 1.0,
        "nuclei_intensity_weight": 1.0,
        "nuclei_gradient_weight": 1.0,
    }
    config.update(overrides)
    return config


def get_cyto_split_config(profile="balanced", **overrides):
    """Return the recommended intensity-gradient cytoplasm splitting settings.

    Parameters
    ----------
    profile : {"conservative", "balanced", "aggressive"}, optional
        Preset controlling how strongly the cytoplasm watershed boundary is
        pulled toward dips/drops in marker intensity vs. pure geometric
        distance between touching cells.
        - conservative -> closer to pure distance watershed (low gradient_weight)
        - balanced     -> equal weighting of distance, intensity, and gradient
        - aggressive   -> boundary snaps hard to intensity dips (high gradient_weight)
    **overrides : dict
        Any keyword arguments matching `segment_cytoplasm`'s split options
        (split_by_intensity_gradient, gradient_smooth_sigma, distance_weight,
        intensity_weight, gradient_weight).

    Returns
    -------
    dict
        A parameter dictionary ready to pass into `segment_cytoplasm(..., **config)`.
    """
    profiles = {
        "conservative": 0.5,
        "balanced": 1.0,
        "aggressive": 2.0,
    }
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose from {tuple(profiles)}."
        )

    config = {
        # split_by_intensity_gradient: when two touching cells' cytoplasm merges
        # into one connected "island", split it where the marker signal itself
        # fades rather than only at the geometric midpoint between nuclei.
        "split_by_intensity_gradient": True,
        # gradient_smooth_sigma: smoothing (voxels) before measuring
        # intensity/gradient. Raise if the split follows noise instead of the
        # marker's real trend.
        "gradient_smooth_sigma": 1.0,
        # distance_weight / intensity_weight / gradient_weight: relative
        # weights of the three elevation terms (geometric distance, absolute
        # dimness, steepness of intensity drop).
        "distance_weight": 1.0,
        "intensity_weight": 1.0,
        "gradient_weight": profiles[profile],
    }
    config.update(overrides)
    return config


def hist_plot(im_in, stain_complete_df, thresh=0, legend=False):
    """Plot histogram and CDF for each channel."""
    # squeeze=False: one channel gives a list of one axis too, not a bare Axes
    fig, axs = plt.subplots(1, im_in.shape[3], figsize=(15, 2), squeeze=False)
    axs = axs[0]
    for c in range(im_in.shape[3]):
        hist, _ = np.histogram(im_in[:, :, :, c].flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        color = stain_complete_df.loc[stain_complete_df.index[c], "Color"]
        axs[c].plot(cdf_normalized, color="b")
        axs[c].hist(
            im_in[:, :, :, c].flatten(),
            256,
            [0, 256],
            color=color if color != "WHITE" else "GRAY",
        )
        axs[c].set_xlim([0, 256])
        if legend:
            axs[c].legend(("cdf", "histogram"), loc="upper left")
        if thresh > 0:
            axs[c].plot([thresh, thresh], [0, cdf_normalized.max()], color="g")
        axs[c].set_title(stain_complete_df.loc[stain_complete_df.index[c], "Marker"])
        axs[c].set_yscale("log")
    return fig, axs


def hist_plot_with_thresholds(im_in, stain_complete_df, global_thresholds, combined_thresholds):
    """Plot each channel's histogram with its global and combined threshold marked.

    Used by `apply_threshold_per_channel` (Cell 15) to show, right after
    thresholding, where the global (dashed orange) and combined — Sauvola +
    statistical + global — (solid red) threshold values landed for each
    channel.

    Parameters
    ----------
    im_in : ndarray, shape (Z, Y, X, C)
        Equalized image stack (the one thresholding was applied to).
    global_thresholds, combined_thresholds : sequence of float
        One value per channel, in the same order as `stain_complete_df.index`.
    """
    fig, axs = plt.subplots(1, im_in.shape[3], figsize=(15, 2), squeeze=False)
    axs = axs[0]  # one channel: still a list of axes
    for c in range(im_in.shape[3]):
        idx = stain_complete_df.index[c]
        color = stain_complete_df.loc[idx, "Color"]
        axs[c].hist(
            im_in[:, :, :, c].flatten(),
            256,
            [0, 256],
            color=color if color != "WHITE" else "GRAY",
        )
        axs[c].axvline(global_thresholds[c], color="orange", linestyle="--", label="Global")
        axs[c].axvline(combined_thresholds[c], color="red", linestyle="-", label="Combined")
        axs[c].set_xlim([0, 256])
        axs[c].set_title(stain_complete_df.loc[idx, "Marker"])
        axs[c].set_yscale("log")

    axs[0].legend(loc="upper right", fontsize=6)
    return fig, axs


def napari_contrast_gamma_uint8(image, contrast_limits, gamma):
    """Apply Napari-style contrast limits and gamma, returning uint8."""
    clim_min, clim_max = contrast_limits

    img = image.astype(np.float32)
    img = (img - clim_min) / (clim_max - clim_min)
    img = np.clip(img, 0.0, 1.0)
    img = img ** gamma

    return (img * 255).round().astype(np.uint8)


def _to_rgb_safe(color_value):
    """Convert a color value to RGB with a gray fallback."""
    try:
        return np.array(mcolors.to_rgb(color_value))
    except Exception:
        return np.array(mcolors.to_rgb("gray"))


def _display_uint8(image, contrast_limits=None, gamma=None):
    """Render to uint8 using the same contrast/gamma logic used for Napari display."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    if contrast_limits is not None:
        cmin = float(contrast_limits[0])
        cmax = float(contrast_limits[1])
        if cmax <= cmin:
            cmin = float(np.min(arr))
            cmax = float(np.max(arr))
    else:
        cmin = float(np.min(arr))
        cmax = float(np.max(arr))

    if cmax <= cmin:
        return np.zeros_like(arr, dtype=np.uint8)

    gamma_value = 1.0 if gamma is None else float(gamma)
    return napari_contrast_gamma_uint8(arr, (cmin, cmax), gamma_value)


def remove_small_islands(binary_matrix, area_threshold, progress=None, desc=None):
    """Remove small connected components from a binary mask.

    All component sizes are counted in one pass (``np.bincount``) and the
    small ones are removed with a single lookup, instead of scanning the
    whole volume once per component. *progress* and *desc* are accepted for
    backward compatibility and ignored.
    """
    labeled_array, num_features = ndi.label(binary_matrix)
    if num_features == 0:
        return binary_matrix
    is_small = np.bincount(labeled_array.ravel()) < area_threshold
    is_small[0] = False  # background
    binary_matrix[is_small[labeled_array]] = 0
    return binary_matrix


def stardist3d_from_2d(
    img_3d,
    model_name="2D_versatile_fluo",
    nucleus_radius=5,
    voxel_size=(1.0, 0.5, 0.5),
    norm=True,
    progress=None,
):
    """Apply StarDist2D slice by slice, then split merged objects in 3D.

    ``nucleus_radius`` is in um and ``voxel_size`` is (Z, Y, X) in um.
    """
    try:
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise ImportError(
            "StarDist requires TensorFlow, csbdeep and stardist, which are not "
            "installed (TensorFlow has no wheels for Python 3.14). Use Python "
            "3.10-3.13 for StarDist, or set trig_stardist=False."
        ) from exc

    if img_3d.ndim != 3:
        raise ValueError("Input must be 3D with shape (Z, Y, X)")

    z_spacing, y_spacing, x_spacing = voxel_size

    model = StarDist2D.from_pretrained(model_name)

    labels_3d = np.zeros_like(img_3d, dtype=np.int32)
    current_label = 1

    for z_index in _progress_iter(
        range(img_3d.shape[0]), progress, desc="StarDist - Predict Z-slices", leave=False
    ):
        img = img_3d[z_index]
        if norm:
            img = normalize(img, 1, 99.8, axis=None)

        labels_2d, _ = model.predict_instances(img)
        labels_2d = np.where(labels_2d > 0, labels_2d + current_label, 0)
        labels_3d[z_index] = labels_2d
        current_label = labels_2d.max() + 1

    labels_3d = skimage.measure.label(labels_3d > 0, connectivity=1)

    distance = ndi.distance_transform_edt(labels_3d > 0, sampling=voxel_size)

    # nucleus_radius is in um: converted to voxels per axis, so the peak
    # neighbourhood follows the voxel size (and scale_factor).
    footprint = np.ones(
        (
            max(1, int(round(nucleus_radius / z_spacing))),
            max(1, int(round(nucleus_radius / y_spacing))),
            max(1, int(round(nucleus_radius / x_spacing))),
        ),
        dtype=bool,
    )

    local_max = peak_local_max(
        distance,
        footprint=footprint,
        labels=labels_3d > 0,
        exclude_border=False,
    )

    markers = np.zeros_like(labels_3d, dtype=int)
    for marker_id, coord in enumerate(local_max, start=1):
        markers[tuple(coord)] = marker_id

    labels_split = watershed(-distance, markers, mask=labels_3d > 0)

    return labels_split


def make_anisotropic_footprint(radius_Z, radius_Y, radius_X):
    zz, yy, xx = np.ogrid[
        -radius_Z : radius_Z + 1,
        -radius_Y : radius_Y + 1,
        -radius_X : radius_X + 1,
    ]
    ellipsoid = (
        (zz / radius_Z) ** 2 + (yy / radius_Y) ** 2 + (xx / radius_X) ** 2
    ) <= 1
    return ellipsoid


def voxel_volume(ri_x, ri_y, ri_z, zooms):
    return (ri_x * ri_y * ri_z) / np.prod(zooms)


def save_raw_png(arr, filename, contrast_limits=None, gamma=None, color=None):
    """Save a 2D array to PNG, with optional Napari-like display and colorization."""
    arr = np.asarray(arr)

    try:
        if (contrast_limits is not None) or (gamma is not None):
            out = _display_uint8(arr, contrast_limits=contrast_limits, gamma=gamma)
            if color is not None:
                rgb = _to_rgb_safe(color)
                out_rgb = np.clip((out.astype(np.float32) / 255.0)[..., None] * rgb, 0.0, 1.0)
                PILImage.fromarray((out_rgb * 255).astype(np.uint8)).save(filename)
            else:
                PILImage.fromarray(out).save(filename)
            return filename
    except Exception:
        pass

    if arr.dtype in (np.uint8, np.uint16):
        PILImage.fromarray(arr).save(filename)
        return filename

    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv == 0:
            PILImage.fromarray(np.zeros_like(arr, dtype=np.uint8)).save(filename)
            return filename

        if maxv <= 255:
            arr_scaled = (arr / maxv) * 255.0
            arr_scaled = np.clip(arr_scaled, 0, 255).astype(np.uint8)
        else:
            arr_scaled = (arr / maxv) * 65535.0
            arr_scaled = np.clip(arr_scaled, 0, 65535).astype(np.uint16)

        PILImage.fromarray(arr_scaled).save(filename)
        return filename

    if np.issubdtype(arr.dtype, np.integer):
        maxv = int(arr.max()) if arr.size else 0
        if maxv <= 255:
            PILImage.fromarray(arr.astype(np.uint8)).save(filename)
            return filename
        if maxv <= 65535:
            PILImage.fromarray(arr.astype(np.uint16)).save(filename)
            return filename

        arr16 = (arr / maxv * 65535).astype(np.uint16)
        PILImage.fromarray(arr16).save(filename)
        return filename

    raise ValueError("Unsupported dtype for PNG saving.")


def crop_nucleus_with_padding(nucleus_mask, full_img_stack, pad=20):
    """Crop each channel around the best matching nucleus slice."""
    if nucleus_mask.ndim == 3:
        z_counts = nucleus_mask.sum(axis=(1, 2))
        best_z = int(np.argmax(z_counts))
        nucleus_2d = nucleus_mask[best_z]
    else:
        best_z = 0
        nucleus_2d = nucleus_mask

    ys, xs = np.where(nucleus_2d)
    if len(xs) == 0:
        return None, best_z, None, None

    y_min0 = max(0, ys.min() - pad)
    y_max0 = ys.max() + pad
    x_min0 = max(0, xs.min() - pad)
    x_max0 = xs.max() + pad

    crop_dict = {}
    heights = []
    widths = []

    for condition, img_3d in full_img_stack.items():
        _, height_full, width_full = img_3d.shape
        y_min = y_min0
        y_max = min(y_max0, height_full)
        x_min = x_min0
        x_max = min(x_max0, width_full)

        cropped = img_3d[best_z, y_min:y_max, x_min:x_max].astype(float)
        crop_dict[condition] = cropped
        heights.append(cropped.shape[0])
        widths.append(cropped.shape[1])

    min_h = int(min(heights)) if heights else 0
    min_w = int(min(widths)) if widths else 0

    return crop_dict, best_z, (y_min0, x_min0), (min_h, min_w)


def save_merged_figure(
    nucleus_mask,
    full_img_stack,
    condition_colors,
    nucleus_id,
    seg_stack,
    display_ranges=None,
    nucleus_color="blue",
    cytoplasm_color="green",
    pcm_color="magenta",
    pad=20,
    out_dir="merged_png",
):
    """Build and save a merged RGB overview around a single nucleus."""
    del nucleus_color, cytoplasm_color, pcm_color

    display_ranges = display_ranges or {}

    os.makedirs(out_dir, exist_ok=True)

    crop_dict, best_z, origin, crop_shape = crop_nucleus_with_padding(
        nucleus_mask,
        full_img_stack,
        pad=pad,
    )
    if crop_dict is None or crop_shape is None:
        return None

    y0, x0 = origin
    min_h, min_w = crop_shape
    if min_h <= 0 or min_w <= 0:
        return None

    merged_rgb = np.zeros((min_h, min_w, 3), dtype=float)
    structure_opacity = 0.2
    white_rgb = np.array([1.0, 1.0, 1.0])
    blue_rgb = np.array([0.0, 0.0, 1.0])

    nucleus_2d = nucleus_mask[best_z] if nucleus_mask.ndim == 3 else nucleus_mask
    nucleus_crop = nucleus_2d[y0 : y0 + min_h, x0 : x0 + min_w].astype(float)
    merged_rgb += nucleus_crop[..., None] * blue_rgb * structure_opacity

    cyto_mask = seg_stack.get("Cytoplasm", np.zeros_like(nucleus_mask)) == nucleus_id
    pcm_mask = seg_stack.get("PCM", np.zeros_like(nucleus_mask)) == nucleus_id
    combined_mask = cyto_mask | pcm_mask

    if np.any(combined_mask):
        combined_2d = combined_mask[best_z] if combined_mask.ndim == 3 else combined_mask
        combined_crop = combined_2d[y0 : y0 + min_h, x0 : x0 + min_w].astype(float)

        contours = find_contours(combined_crop, 0.5, fully_connected="low")
        contour_rgb = np.zeros((min_h, min_w, 3), dtype=float)
        for contour in contours:
            contour[:, 0] *= min_h / combined_crop.shape[0]
            contour[:, 1] *= min_w / combined_crop.shape[1]
            contour_int = contour.astype(int)
            for index in range(0, len(contour_int), 2):
                if index + 1 < len(contour_int):
                    rr, cc = draw_line(
                        int(contour_int[index, 0]),
                        int(contour_int[index, 1]),
                        int(contour_int[index + 1, 0]),
                        int(contour_int[index + 1, 1]),
                    )
                    contour_rgb[rr, cc] = white_rgb * structure_opacity
        merged_rgb += contour_rgb

    for condition, img in crop_dict.items():
        img_small = img[:min_h, :min_w].copy()

        clim = display_ranges.get(condition)
        if clim is not None:
            try:
                img_display = napari_contrast_gamma_uint8(
                    img_small.astype(np.float32),
                    (float(clim[0]), float(clim[1])),
                    1.0,
                )
                img_normalized = img_display.astype(float) / 255.0
            except Exception:
                img_normalized = img_small / (img_small.max() + 1e-6)
        else:
            img_normalized = img_small / (img_small.max() + 1e-6)

        color = _to_rgb_safe(condition_colors.get(condition, "gray"))
        merged_rgb += img_normalized[..., None] * color

    merged_rgb = np.clip(merged_rgb, 0, 1.0)
    merged_uint8 = (merged_rgb * 255).astype(np.uint8)

    filename = os.path.join(out_dir, f"n{nucleus_id}_merged.png")
    PILImage.fromarray(merged_uint8).save(filename)
    return filename


def double_plateau_hist_equalization_nd(
    img: np.ndarray,
    num_plateaus: int = 2,
    plateau_factor: float = 0.5,
) -> np.ndarray:
    """Multi-plateau histogram equalization for uint8 images and volumes."""
    if img.dtype != np.uint8:
        raise ValueError("Input must be uint8")

    if img.ndim == 2:
        return _mphe_channel(img, num_plateaus, plateau_factor)

    if img.ndim == 3 and img.shape[-1] != 3:
        flat = img.ravel()
        flat_eq = _mphe_flat(flat, num_plateaus, plateau_factor)
        return flat_eq.reshape(img.shape)

    if img.ndim == 3 and img.shape[-1] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_eq = _mphe_channel(y_channel, num_plateaus, plateau_factor)
        ycrcb_eq = cv2.merge([y_eq, cr_channel, cb_channel])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    raise ValueError("Unsupported input shape")


def _mphe_flat(flat: np.ndarray, num_plateaus: int, plateau_factor: float) -> np.ndarray:
    """Apply multi-plateau histogram equalization on a flat uint8 array."""
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    total_pixels = flat.size

    mean_count = total_pixels / 256.0
    base_plateau = plateau_factor * mean_count

    plateau_levels = np.linspace(
        base_plateau * 0.5,
        base_plateau * (0.5 + num_plateaus),
        num_plateaus,
    )

    clipped_hist = hist.copy()
    for level in plateau_levels:
        excess = np.maximum(clipped_hist - level, 0)
        clipped_hist = np.minimum(clipped_hist, level)
        clipped_hist += excess.sum() / 256.0

    cdf = np.cumsum(clipped_hist)
    cdf_norm = cdf / cdf[-1]
    lut = np.floor(255 * cdf_norm).astype(np.uint8)
    return lut[flat]


def _mphe_channel(channel: np.ndarray, num_plateaus: int, plateau_factor: float) -> np.ndarray:
    """Apply multi-plateau histogram equalization to one 2D channel."""
    flat = channel.ravel()
    flat_eq = _mphe_flat(flat, num_plateaus, plateau_factor)
    return flat_eq.reshape(channel.shape)


def _build_neighbor_offsets(connectivity):
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == dy == dx == 0:
                    continue
                if connectivity == 1 and (abs(dx) + abs(dy) + abs(dz) > 1):
                    continue
                offsets.append((dz, dy, dx))
    return offsets


def _find_label_neighbors(marker_labels, connectivity):
    """Map each label to the set of labels it touches.

    Vectorized: for each neighbour offset the volume is compared with a
    shifted copy of itself, so no Python loop runs over voxels. Only one
    offset of each +/- pair is needed, since touching is symmetric.
    """
    neighbors = defaultdict(set)
    marker_labels = np.asarray(marker_labels)

    def _axis_slices(d, n):
        if d > 0:
            return slice(0, n - d), slice(d, n)
        if d < 0:
            return slice(-d, n), slice(0, n + d)
        return slice(None), slice(None)

    pairs = []
    for offset in _build_neighbor_offsets(connectivity):
        if offset < (0, 0, 0):
            continue
        src, dst = zip(*(_axis_slices(d, n) for d, n in zip(offset, marker_labels.shape)))
        a = marker_labels[src]
        b = marker_labels[dst]
        touching = (a > 0) & (b > 0) & (a != b)
        if touching.any():
            pairs.append(np.stack([a[touching], b[touching]], axis=1))

    if pairs:
        for label_a, label_b in np.unique(np.concatenate(pairs), axis=0).tolist():
            neighbors[label_a].add(label_b)
            neighbors[label_b].add(label_a)
    return neighbors


def _merge_small_touching_regions(marker_labels, connectivity, size_ratio_thresh, min_size):
    """Merge each label smaller than ``min_size`` voxels — or smaller than
    ``size_ratio_thresh`` times its largest touching neighbor — into that
    neighbor. Shared by ``shrink_to_markers``'s ``merge_small_touching``
    option and ``remove_small_island_labels``.

    Returns
    -------
    relabeled : ndarray
        ``marker_labels`` with small labels reassigned to their neighbor's
        id (original id space, may contain gaps where a label vanished).
    final_labels : ndarray
        ``relabeled > 0`` re-run through connected-component labeling
        (sequential ids, no gaps).
    """
    labels, counts = np.unique(marker_labels, return_counts=True)
    sizes = dict(zip(labels.tolist(), counts.tolist()))
    sizes.pop(0, None)
    neighbors = _find_label_neighbors(marker_labels, connectivity)

    new_label = {label: label for label in sizes}
    for label, label_neighbors in neighbors.items():
        if not label_neighbors:
            continue

        size_label = sizes[label]
        biggest_neighbor = max(label_neighbors, key=lambda item: sizes.get(item, 0))
        size_biggest = sizes.get(biggest_neighbor, 0)
        if size_biggest <= 0:
            continue

        if size_label <= size_ratio_thresh * size_biggest or size_label < min_size:
            new_label[label] = biggest_neighbor

    # One lookup-table pass instead of one full-volume pass per merged label.
    lut = np.arange(int(marker_labels.max()) + 1, dtype=marker_labels.dtype)
    for label, target in new_label.items():
        lut[label] = target
    relabeled = lut[marker_labels]

    final_labels, _ = ndi.label(
        relabeled > 0,
        structure=ndi.generate_binary_structure(3, connectivity),
    )
    return relabeled, final_labels


def shrink_to_markers(
    binary_3d,
    connectivity=2,
    max_iter=100,
    min_final_size=3,
    max_final_cc=3,
    merge_small_touching=False,
    size_ratio_thresh=0.5,
):
    """Shrink each connected component until stable and use it as a marker."""
    binary_3d = binary_3d.astype(bool)
    structure = ndi.generate_binary_structure(3, connectivity)
    cc_labels, n_cc = ndi.label(binary_3d, structure=structure)
    marker_image = np.zeros_like(binary_3d, dtype=bool)

    for cc_id in range(1, n_cc + 1):
        current = cc_labels == cc_id
        iteration = 0

        while np.sum(current) > min_final_size and iteration < max_iter:
            eroded = ndi.binary_erosion(current, structure=structure)
            if not np.any(eroded):
                break

            n_cc_eroded = ndi.label(eroded, structure=structure)[1]
            if n_cc_eroded > max_final_cc:
                break

            current = eroded
            iteration += 1

        marker_image |= current

    marker_labels, n_markers = ndi.label(marker_image, structure=structure)
    if not merge_small_touching or n_markers <= 1:
        return marker_image, marker_labels

    _, final_labels = _merge_small_touching_regions(
        marker_labels, connectivity, size_ratio_thresh, min_size=20.0,
    )
    return final_labels > 0, final_labels


def shrink_to_markers_robust(binary_3d, min_marker_size=1, size_ratio_thresh=0.4, **kwargs):
    """Run marker shrinking with a final absolute size filter."""
    del size_ratio_thresh

    _, markers_lab = shrink_to_markers(binary_3d, connectivity=2, **kwargs)
    sizes = np.bincount(markers_lab.ravel())[1:]
    keep_labels = np.where(sizes >= min_marker_size)[0] + 1

    filtered = np.isin(markers_lab, keep_labels)
    final_labels, _ = ndi.label(filtered)
    return filtered, final_labels


def remove_small_island_labels(
    marker_labels,
    connectivity=1,
    size_ratio_thresh=0.5,
    min_cell_size=5.0,
):
    """Merge tiny touching label islands into their larger neighbors."""
    relabeled, final_labels = _merge_small_touching_regions(
        marker_labels, connectivity, size_ratio_thresh, min_size=min_cell_size,
    )
    return final_labels > 0, relabeled


def merge_touching_labels(
    label_matrix,
    contact_abs_min=15,
    contact_rel_min=0.10,
    size_ratio_max=3.0,
    connectivity=1,
):
    """Merge labels only when contact is strong and sizes are reasonably similar."""
    if label_matrix.max() == 0:
        return label_matrix.copy()

    labels, counts = np.unique(label_matrix, return_counts=True)
    volumes = dict(zip(labels.tolist(), counts.tolist()))
    volumes.pop(0, None)

    offsets = _build_neighbor_offsets(connectivity)
    touching_counts = defaultdict(int)
    z_size, y_size, x_size = label_matrix.shape
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                current = label_matrix[z_index, y_index, x_index]
                if current == 0:
                    continue
                for dz, dy, dx in offsets:
                    nz = z_index + dz
                    ny = y_index + dy
                    nx = x_index + dx
                    if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
                        neighbor = label_matrix[nz, ny, nx]
                        if neighbor > 0 and neighbor != current:
                            label_a, label_b = sorted((current, neighbor))
                            touching_counts[(label_a, label_b)] += 1

    parent = {label: label for label in volumes}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(label_a, label_b):
        root_a, root_b = find(label_a), find(label_b)
        if root_a != root_b:
            parent[root_a] = root_b

    for (label_a, label_b), contact in touching_counts.items():
        volume_a = volumes[label_a]
        volume_b = volumes[label_b]
        volume_min = min(volume_a, volume_b)
        volume_max = max(volume_a, volume_b)
        rel_contact = contact / float(volume_min)
        if (
            contact >= contact_abs_min
            and rel_contact >= contact_rel_min
            and volume_max / volume_min <= size_ratio_max
        ):
            union(label_a, label_b)

    merged = np.zeros_like(label_matrix, dtype=np.int32)
    for label in volumes:
        merged[label_matrix == label] = find(label)
    merged, _, _ = skimage.segmentation.relabel_sequential(merged)
    return merged


def assign_labels(mask, labels, connectivity=1):
    """Assign the first overlapping label to each connected mask component."""
    structure = np.ones((3, 3, 3), dtype=int) if connectivity == 2 else None
    labeled_mask, num_features = ndi.label(mask, structure=structure)
    assigned = np.zeros_like(mask, dtype=labels.dtype)

    for component_id in range(1, num_features + 1):
        component = labeled_mask == component_id
        overlapping = np.unique(labels[component & (labels > 0)])
        assigned[component] = overlapping[0] if len(overlapping) > 0 else 0

    return assigned


def grow_labels(label_matrix, volume_factor=5.0):
    """Grow labels into free space until a target volume factor is reached."""
    from skimage.morphology import ball

    structure = ball(volume_factor)
    output = label_matrix.copy()
    labels = np.unique(label_matrix)
    labels = labels[labels != 0]

    volumes = {label: np.sum(label_matrix == label) for label in labels}
    target_volumes = {label: volume_factor * volume for label, volume in volumes.items()}
    grown_masks = {label: label_matrix == label for label in labels}
    growing = {label: True for label in labels}

    while any(growing.values()):
        occupied = np.zeros_like(label_matrix, dtype=bool)
        for mask in grown_masks.values():
            occupied |= mask

        for label in labels:
            if not growing[label]:
                continue
            dilated = ndi.binary_dilation(grown_masks[label], structure)
            new_mask = dilated & ~occupied
            combined = grown_masks[label] | new_mask
            if np.sum(combined) >= target_volumes[label]:
                growing[label] = False
            grown_masks[label] = combined

        output[:] = 0
        for label, mask in grown_masks.items():
            output[mask] = label

    return output


def expand_labels_um(label_matrix, distance_um, spacing):
    """Grow every label into free space by *distance_um* micrometers.

    Distances use the physical voxel *spacing* (Z, Y, X), so the growth is
    the same in um whatever the voxel size (or ``scale_factor``). Each free
    voxel goes to its nearest label.
    """
    if distance_um <= 0 or not np.any(label_matrix):
        return label_matrix.copy()
    distance, indices = ndi.distance_transform_edt(
        label_matrix == 0, sampling=spacing, return_indices=True,
    )
    output = label_matrix[tuple(indices)]
    output[distance > distance_um] = 0
    return output


def merge_small_touching_labels(label_matrix, size_threshold, z_weight=2.0):
    """Merge small touching labels into their largest viable neighbor."""
    del z_weight

    if label_matrix.max() == 0:
        return label_matrix.copy()

    props = skimage.measure.regionprops(label_matrix)
    sizes = {prop.label: prop.area for prop in props}
    small_labels = {label for label, area in sizes.items() if area < size_threshold}
    neighbors = _find_label_neighbors(label_matrix, connectivity=2)
    merged = label_matrix.copy()

    for label in sorted(small_labels):
        if not np.any(merged == label):
            continue
        touching = neighbors.get(label, set())
        if not touching:
            continue
        candidate_sizes = {neighbor: sizes.get(neighbor, 0) for neighbor in touching if neighbor not in small_labels}
        if not candidate_sizes:
            candidate_sizes = {neighbor: sizes.get(neighbor, 0) for neighbor in touching}
        if not candidate_sizes:
            continue
        target = max(candidate_sizes, key=candidate_sizes.get)
        merged[merged == label] = target

    merged, _, _ = skimage.segmentation.relabel_sequential(merged)
    return merged


def _n_cells(seg_stack):
    """Number of cells the marker statistics run over (labels 1..N).

    `segment_cytoplasm` numbers the nuclei with a cytoplasm first, so this is
    the largest cytoplasm label; nuclei above it have no cytoplasm and are
    excluded. Without a cytoplasm (e.g. LD mode) every nucleus counts.
    """
    cyto = seg_stack.get("Cytoplasm")
    if cyto is not None and np.any(cyto):
        return int(np.max(cyto))
    return int(np.max(seg_stack["Nuclei"]))


def compute_nuclei_cytoplasm_stats(seg_stack, r_xyz, zooms, progress=None):
    """Compute centroid positions and volumes for nuclei and cytoplasm."""
    max_label = int(np.max(seg_stack["Nuclei"]))
    nucleus_positions = []
    nucleus_sizes = []
    cytoplasm_positions = []
    cytoplasm_sizes = []
    # Each label in its bounding box (one whole-image pass per label before):
    # the same voxels in the same order, so the same values.
    nuclei_img, cyto_img = seg_stack["Nuclei"], seg_stack["Cytoplasm"]
    nuc_slices = _union_label_slices([nuclei_img], max_label)
    cyto_slices = _union_label_slices([cyto_img], max_label)

    def _where_label(img, sl, label_id):
        if sl is None:
            return (np.empty(0, dtype=np.intp),) * 3
        idx = np.where(img[sl] == label_id)
        return tuple(i + s.start for i, s in zip(idx, sl))

    for label_id in _progress_iter(
        range(1, max_label + 1), progress, desc="Quantify Nuclei/Cytoplasm per cell", leave=False
    ):
        z_nuc, y_nuc, x_nuc = _where_label(nuclei_img, nuc_slices[label_id - 1], label_id)
        if x_nuc.size == 0:
            nucleus_positions.append((0.0, 0.0, 0.0))
            nucleus_sizes.append(0.0)
        else:
            nucleus_positions.append(
                (
                    float(np.mean(x_nuc) * r_xyz[0]),
                    float(np.mean(y_nuc) * r_xyz[1]),
                    float(np.mean(z_nuc) * r_xyz[2]),
                )
            )
            nucleus_sizes.append(float(x_nuc.size * r_xyz[0] * r_xyz[1] * r_xyz[2]))

        z_cyto, y_cyto, x_cyto = _where_label(cyto_img, cyto_slices[label_id - 1], label_id)
        if x_cyto.size == 0:
            cytoplasm_positions.append((0.0, 0.0, 0.0))
            cytoplasm_sizes.append(0.0)
        else:
            cytoplasm_positions.append(
                (
                    float(np.mean(x_cyto) * r_xyz[0]),
                    float(np.mean(y_cyto) * r_xyz[1]),
                    float(np.mean(z_cyto) * r_xyz[2]),
                )
            )
            cytoplasm_sizes.append(float(x_cyto.size * r_xyz[0] * r_xyz[1] * r_xyz[2]))

    # Nuclei excluded for having no cytoplasm (labels above _n_cells) have
    # no cytoplasm entry.
    n_cells = _n_cells(seg_stack)
    return nucleus_positions, nucleus_sizes, cytoplasm_positions[:n_cells], cytoplasm_sizes[:n_cells]


def _compute_marker_stats_core(
    marker_idx,
    seg_stack,
    filtered_img,
    r_xyz,
    stain_complete_df,
    seg_key,
    fallback_to_marker_when_missing=False,
    progress=None,
    progress_desc=None,
):
    """Shared implementation for compute_marker_stats_for_marker and
    compute_full_marker_stats_for_marker.

    ``fallback_to_marker_when_missing`` controls what happens when a
    per-compartment override image (``seg_key + "_cyto"`` / ``"_PCM"``) is
    absent from ``seg_stack``: False reports zeros for that compartment
    (compute_marker_stats_for_marker's behaviour), True falls back to the
    main marker mask intersected with the compartment
    (compute_full_marker_stats_for_marker's behaviour).
    """
    condition = stain_complete_df.index[marker_idx]

    marker_img = seg_stack.get(seg_key)
    marker_img_cyto = seg_stack.get(seg_key + "_cyto")
    marker_img_pcm = seg_stack.get(seg_key + "_PCM")
    if marker_img is None:
        return [], [], [], [], [], [], [], [], [], []

    channel_idx = None
    for idx, name in enumerate(stain_complete_df.index):
        if name == condition:
            channel_idx = idx
            break

    shared_labels = []
    marker_sizes = []
    avg_marker = []
    std_marker = []
    marker_cyto_sizes = []
    avg_cyto_marker = []
    std_cyto_marker = []
    marker_pcm_sizes = []
    avg_pcm_marker = []
    std_pcm_marker = []

    max_label = _n_cells(seg_stack)
    # Each cell in the bounding box of its nucleus, cytoplasm and PCM (one
    # whole-image pass per cell before): the same voxels in the same order,
    # so the same values.
    cell_slices = _union_label_slices(
        [seg_stack["Nuclei"], seg_stack["Cytoplasm"], seg_stack["PCM"]], max_label)
    for label_id in _progress_iter(
        range(1, max_label + 1), progress,
        desc=progress_desc or f"Quantify [{condition}] per cell", leave=False,
    ):
        sl = cell_slices[label_id - 1]
        if sl is None:
            continue
        nucleus_mask = seg_stack["Nuclei"][sl] == label_id
        cytoplasm_mask = seg_stack["Cytoplasm"][sl] == label_id
        pcm_mask = seg_stack["PCM"][sl] == label_id
        marker_mask = (marker_img[sl] > 0) & ((nucleus_mask + cytoplasm_mask + pcm_mask) > 0)
        if not np.any(marker_mask):
            continue
        filtered_c = filtered_img[sl]

        shared_labels.append(label_id)
        voxels = np.where(marker_mask)
        marker_sizes.append(voxels[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
        if channel_idx is not None:
            values = filtered_c[voxels[0], voxels[1], voxels[2], channel_idx]
            avg_marker.append(float(np.mean(values)) if values.size > 0 else 0.0)
            std_marker.append(float(np.std(values)) if values.size > 0 else 0.0)
        else:
            avg_marker.append(0.0)
            std_marker.append(0.0)

        if marker_img_cyto is not None:
            cytoplasm_marker_mask = (marker_img_cyto[sl] > 0) & cytoplasm_mask
        elif fallback_to_marker_when_missing:
            cytoplasm_marker_mask = (marker_img[sl] > 0) & cytoplasm_mask
        else:
            cytoplasm_marker_mask = None
        if cytoplasm_marker_mask is not None:
            voxels_cyto = np.where(cytoplasm_marker_mask)
            marker_cyto_sizes.append(voxels_cyto[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
            if channel_idx is not None and voxels_cyto[0].size > 0:
                values_cyto = filtered_c[voxels_cyto[0], voxels_cyto[1], voxels_cyto[2], channel_idx]
                avg_cyto_marker.append(float(np.mean(values_cyto)))
                std_cyto_marker.append(float(np.std(values_cyto)))
            else:
                avg_cyto_marker.append(0.0)
                std_cyto_marker.append(0.0)
        else:
            marker_cyto_sizes.append(0.0)
            avg_cyto_marker.append(0.0)
            std_cyto_marker.append(0.0)

        if marker_img_pcm is not None:
            pcm_marker_mask = (marker_img_pcm[sl] > 0) & pcm_mask
        elif fallback_to_marker_when_missing:
            pcm_marker_mask = (marker_img[sl] > 0) & pcm_mask
        else:
            pcm_marker_mask = None
        if pcm_marker_mask is not None:
            voxels_pcm = np.where(pcm_marker_mask)
            marker_pcm_sizes.append(voxels_pcm[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
            if channel_idx is not None and voxels_pcm[0].size > 0:
                values_pcm = filtered_c[voxels_pcm[0], voxels_pcm[1], voxels_pcm[2], channel_idx]
                avg_pcm_marker.append(float(np.mean(values_pcm)))
                std_pcm_marker.append(float(np.std(values_pcm)))
            else:
                avg_pcm_marker.append(0.0)
                std_pcm_marker.append(0.0)
        else:
            marker_pcm_sizes.append(0.0)
            avg_pcm_marker.append(0.0)
            std_pcm_marker.append(0.0)

    return (
        shared_labels,
        marker_sizes,
        avg_marker,
        std_marker,
        marker_cyto_sizes,
        avg_cyto_marker,
        std_cyto_marker,
        marker_pcm_sizes,
        avg_pcm_marker,
        std_pcm_marker,
    )


def compute_marker_stats_for_marker(marker_idx, seg_stack, filtered_img, r_xyz, zooms, progress=None):
    """Compute marker measurements per nucleus for one marker channel."""
    stain_complete_df = _context("stain_complete_df")
    stain_df = _context("stain_df")
    seg_key = stain_df.index[marker_idx]
    return _compute_marker_stats_core(
        marker_idx, seg_stack, filtered_img, r_xyz, stain_complete_df, seg_key,
        fallback_to_marker_when_missing=False,
        progress=progress,
    )


def compute_full_marker_stats_for_marker(marker_idx, seg_final, seg_stack, filtered_img, r_xyz, zooms, progress=None):
    """Compute full marker measurements from the filtered image channel.

    Uses the assigned segmentation label image (same presence test as
    compute_marker_stats_for_marker) but measures intensities from the
    original-resolution filtered_img for higher accuracy.
    """
    stain_complete_df = _context("stain_complete_df")
    stain_df = _context("stain_df")
    seg_key = stain_df.index[marker_idx] if stain_df is not None else stain_complete_df.index[marker_idx]
    return _compute_marker_stats_core(
        marker_idx, seg_stack, filtered_img, r_xyz, stain_complete_df, seg_key,
        fallback_to_marker_when_missing=True,
        progress=progress,
    )


LABELS_TABLE_COLUMNS = [
    "Condition",
    "Laser",
    "Color",
    "Number",
    "Shared labels",
    "Mean nuclei positions [um]",
    "Mean cytoplasm positions [um]",
    "Nuclei size [um3]",
    "Cytoplasm size [um3]",
    "Marker size [um3]",
    "Avg. marker intensity",
    "STD marker intensity",
    "Marker size cytoplasm [um3]",
    "Avg. marker intensity cytoplasm",
    "STD marker intensity cytoplasm",
    "Marker size PCM [um3]",
    "Avg. marker intensity PCM",
    "STD marker intensity PCM",
    "Avg. marker intensity (all cells)",
    "STD marker intensity (all cells)",
]


def _progress_iter(iterable, progress=None, desc=None, **kwargs):
    """Wrap an iterable with an optional notebook progress helper."""
    if progress is None:
        return iterable
    if desc is None:
        return progress(iterable, **kwargs)
    return progress(iterable, desc=desc, **kwargs)


def _make_labels_record(
    condition,
    laser,
    color,
    number,
    shared_labels=(),
    nucleus_positions=(),
    cytoplasm_positions=(),
    nucleus_sizes=(),
    cytoplasm_sizes=(),
    marker_sizes=(),
    avg_marker=(),
    std_marker=(),
    marker_cyto_sizes=(),
    avg_cyto_marker=(),
    std_cyto_marker=(),
    marker_pcm_sizes=(),
    avg_pcm_marker=(),
    std_pcm_marker=(),
    avg_marker_all=(),
    std_marker_all=(),
):
    """Create one quantification row in the shared notebook output format.

    ``avg_marker_all``/``std_marker_all`` (see
    ``compute_percell_marker_intensity_df``) are dense, one entry per
    segmented cell (positive or not) in cell-label order -- unlike every
    other per-marker field here, which only covers cells where that
    marker's threshold mask fired.
    """
    return [
        condition,
        laser,
        color,
        int(number),
        tuple(shared_labels),
        tuple(nucleus_positions),
        tuple(cytoplasm_positions),
        tuple(nucleus_sizes),
        tuple(cytoplasm_sizes),
        tuple(marker_sizes),
        tuple(avg_marker),
        tuple(std_marker),
        tuple(marker_cyto_sizes),
        tuple(avg_cyto_marker),
        tuple(std_cyto_marker),
        tuple(marker_pcm_sizes),
        tuple(avg_pcm_marker),
        tuple(std_pcm_marker),
        tuple(avg_marker_all),
        tuple(std_marker_all),
    ]


def _condition_color(condition, stain_complete_df, stain_df=None):
    """Return a display color for a single condition or a marker combination."""
    def _normalize(color_value):
        if not color_value or color_value == "WHITE":
            return "GRAY"
        return color_value

    if np.size(condition) == 1 and not isinstance(condition, tuple):
        if condition in stain_complete_df.index and "Color" in stain_complete_df.columns:
            return _normalize(stain_complete_df.loc[condition, "Color"])
        return "BLUE"

    rgb_list = []
    values = condition if isinstance(condition, tuple) else (condition,)
    for item in values:
        color_value = None
        if item in stain_complete_df.index and "Color" in stain_complete_df.columns:
            color_value = stain_complete_df.loc[item, "Color"]
        elif stain_df is not None and item in stain_df.index and "Color" in stain_df.columns:
            color_value = stain_df.loc[item, "Color"]
        rgb_list.append(_normalize(color_value or "GRAY"))

    colors_rgb = [mcolors.to_rgb(name) for name in rgb_list]
    r_final = min(sum(rgb[0] for rgb in colors_rgb), 1.0)
    g_final = min(sum(rgb[1] for rgb in colors_rgb), 1.0)
    b_final = min(sum(rgb[2] for rgb in colors_rgb), 1.0)
    return (r_final, g_final, b_final)


def _normalize_channel_to_uint8_view(im_channel):
    """Per-channel min-max stretch to uint8 [0, 255], for preview and auto-contrast."""
    imin = float(im_channel.min())
    imax = float(im_channel.max())
    if imax <= imin:
        return np.zeros_like(im_channel, dtype=np.uint8)
    return ((im_channel - imin) / (imax - imin) * 255).clip(0, 255).astype(np.uint8)


def _contrast_mode(automatic_contrast=False, calibrated_contrast=None):
    """``automatic_contrast`` value for the notebook switch ``calibrated_contrast``.

    True -> ``'calibrated'`` (from each channel's histogram, no napari window),
    False -> False (set in napari), None -> ``automatic_contrast`` unchanged.
    """
    if calibrated_contrast is not None:
        automatic_contrast = "calibrated" if calibrated_contrast else False
    if automatic_contrast not in (False, True, "calibrated"):
        raise ValueError("automatic_contrast must be False, True or 'calibrated'")
    return automatic_contrast


def _compute_automatic_contrast_limits(im_view, method="legacy"):
    """Pick (Cont_min, Cont_max) for one channel from its intensity histogram.

    ``im_view`` is the same per-channel min-max stretch to [0, 255] used for
    the interactive napari preview, so the result lands on the same scale
    ``Cont_min``/``Cont_max`` are interpreted against downstream (napari
    contrast limits, then ``run_normalize()`` + ``apply_contrast_gamma_per_channel``).

    ``method='legacy'`` (``automatic_contrast=True``):

    - Cont_min: one above the histogram peak (the background/noise floor in
      a typical fluorescence image), to exclude background.
    - Cont_max: the intensity below which 99% of pixels fall, so the top 1%
      brightest (often saturated/outlier) pixels don't wash out the contrast.

    ``method='calibrated'`` (v1.6.4, ``automatic_contrast='calibrated'``),
    fitted on the nuclei ground truths of GT1 and GT2
    (validation/tune_segmentation.py):

    - Cont_min = background - 0.25 x noise (background = median, noise =
      1.4826 x MAD). The segmentation is very sensitive to it: the tuned
      optimum sat at the background on both images, and 10 grey levels off
      it cost most of the score.
    - Cont_max = background + 2 x (99.95th percentile - background): well
      above the brightest nuclei. The score was flat above ~the 99.95th
      percentile, while the legacy 99th percentile saturated the nuclei.
    """
    if method == "calibrated":
        sample = im_view[::2, ::2, ::2] if im_view.ndim == 3 and im_view.size > 4_000_000 \
            else im_view
        sample = sample.astype(np.float32).ravel()
        bg = float(np.median(sample))
        noise = 1.4826 * float(np.median(np.abs(sample - bg)))
        p_hi = float(np.percentile(sample, 99.95))
        cont_min = int(np.clip(round(bg - 0.25 * noise), 0, 254))
        cont_max = int(np.clip(round(bg + 2.0 * (p_hi - bg)), cont_min + 1, 255))
        return cont_min, cont_max
    if method != "legacy":
        raise ValueError(f"method must be 'legacy' or 'calibrated', not {method!r}")
    hist_counts = np.bincount(im_view.ravel(), minlength=256)
    peak_value = int(np.argmax(hist_counts))
    cont_min = min(255, peak_value + 1)
    cont_max = int(np.percentile(im_view, 99))
    if cont_max <= cont_min:
        cont_max = min(255, cont_min + 1)
    return cont_min, cont_max


def calibrated_contrast(im_channel, gamma=1.0):
    """v1.6.4: ``{"Cont_min", "Cont_max", "Gamma"}`` of one channel (Z, Y, X)
    from its own histogram, with the ``'calibrated'`` rule of
    `_compute_automatic_contrast_limits`, on the same 0-255 min-max scale as
    the stain table. Used for the NUCLEI channel of an image the tuned
    settings were not tuned on: its Cont_min lands at the background, where
    the tuned values sat on all four ground truths (within 1-3 grey levels).

    v1.6.41: ``gamma`` (default 1.0 = v1.6.4). On the calibrated window, gamma
    1.15 scored better than 1.0 on all four ground truths treated as new
    images (mean F1 0.407 vs 0.385; the optimal settings v1.6.41 use it)."""
    lo, hi = _compute_automatic_contrast_limits(_normalize_channel_to_uint8_view(im_channel),
                                                method="calibrated")
    return {"Cont_min": int(lo), "Cont_max": int(hi), "Gamma": float(gamma)}


def suggest_contrast_settings(im_in, stain_complete_df, current=None, warn_noise=1.0):
    """Suggested contrast/gamma per channel, and a check of the chosen ones (v1.6.4).

    The suggestion is the ``'calibrated'`` rule of
    :func:`_compute_automatic_contrast_limits` (Gamma 1.0). With ``current``
    (a stain table with ``Cont_min``/``Cont_max``/``Gamma``, e.g. the one read
    from the setup CSV or set in napari), every channel whose ``Cont_min`` is
    more than ``warn_noise`` x the background noise away from the background,
    whose ``Cont_max`` is below the 99.95th percentile (bright nuclei
    saturated), or whose gamma is outside 0.7-1.3 is flagged.

    Parameters
    ----------
    im_in : ndarray (Z, Y, X, C)
        Original image (channels in the order of ``stain_complete_df``).
    stain_complete_df : DataFrame
        One row per channel (index: Condition, or Condition/Marker/Laser).

    Returns
    -------
    DataFrame, one row per channel: background, noise, p99.95, suggested
    Cont_min/Cont_max/Gamma and, with ``current``, the chosen values and
    a ``Check`` column.
    """
    rows = []
    for c, idx in enumerate(stain_complete_df.index):
        view = _normalize_channel_to_uint8_view(im_in[:, :, :, c])
        sample = (view[::2, ::2, ::2] if view.size > 4_000_000 else view).astype(np.float32).ravel()
        bg = float(np.median(sample))
        noise = 1.4826 * float(np.median(np.abs(sample - bg)))
        p_hi = float(np.percentile(sample, 99.95))
        s_min, s_max = _compute_automatic_contrast_limits(view, method="calibrated")
        cond = idx[0] if isinstance(idx, tuple) else idx
        row = {"Condition": cond, "Background": round(bg, 1), "Noise": round(noise, 2),
               "p99.95": round(p_hi, 1), "Suggested Cont_min": s_min,
               "Suggested Cont_max": s_max, "Suggested Gamma": 1.0}
        if current is not None:
            cur = current.loc[idx]
            cmin, cmax, gam = float(cur["Cont_min"]), float(cur["Cont_max"]), float(cur["Gamma"])
            notes = []
            if abs(cmin - bg) > warn_noise * max(noise, 1.0):
                notes.append(f"Cont_min {cmin:g} is {(cmin - bg) / max(noise, 1.0):+.1f} noise "
                             f"from the background ({bg:g}): try ~{s_min}")
            if cmax < p_hi:
                notes.append(f"Cont_max {cmax:g} < p99.95 ({p_hi:g}): bright nuclei saturate, "
                             f"try ~{s_max}")
            if not 0.7 <= gam <= 1.3:
                notes.append(f"Gamma {gam:g} is far from 1")
            row.update({"Cont_min": cmin, "Cont_max": cmax, "Gamma": gam,
                        "Check": "; ".join(notes) or "ok"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def prepare_stain_settings(
    im_in,
    stain_df,
    name_setup,
    use_setup=True,
    settings=None,
    napari_module=None,
    r_xyz=None,
    automatic_contrast=False,
    progress=None,
    suggest_contrast=False,
    calibrated_contrast=None,
):
    """Load or interactively collect contrast/gamma settings for each channel.

    Parameters
    ----------
    calibrated_contrast : bool, optional
        The notebook switch (Cell 3). True = contrast of the marker channels
        from each channel's histogram with the calibrated rule (no napari
        window; same as ``automatic_contrast='calibrated'``); False = set them
        yourself in the napari viewer (same as ``automatic_contrast=False``).
        None (default) = use ``automatic_contrast``.
    r_xyz : tuple of float (r_X, r_Y, r_Z), optional
        Physical voxel sizes in micrometers. Applied as the napari layer scale
        during interactive setup so the preview isn't stretched/squashed
        along Z relative to XY.
    automatic_contrast : bool, optional
        If False (default), behaviour is unchanged: a napari viewer opens for
        interactive contrast/gamma adjustment whenever no reusable setup CSV
        is found. If True, that viewer is skipped and ``Cont_min``/``Cont_max``
        are instead picked automatically per channel from the intensity
        histogram (see ``_compute_automatic_contrast_limits``); ``Gamma``
        stays at the default 1.0. Has no effect when a setup CSV is reused
        (``use_setup=True`` and the file exists) — that path never opens a
        viewer to begin with. ``'calibrated'`` (v1.6.4) uses the rule
        calibrated on the nuclei ground truths instead (see
        ``_compute_automatic_contrast_limits``).
    suggest_contrast : bool, optional
        v1.6.4. The napari setup viewer opens with every channel at the
        calibrated suggestion (instead of its full range), as a starting point
        to adjust; and the chosen (or reused) values are checked against the
        suggestion (:func:`suggest_contrast_settings`), with a note for each
        value far from it.
    """
    automatic_contrast = _contrast_mode(automatic_contrast, calibrated_contrast)
    auto_method ="calibrated" if automatic_contrast == "calibrated" else "legacy"
    stain_df = stain_df.reset_index(drop=False)
    stain_initial_df = stain_df.copy()
    stain_initial_df.set_index(["Condition", "Marker", "Laser"], inplace=True)
    stain_initial_df[["Cont_min", "Cont_max", "Gamma"]] = [0, 255, 1.0]
    stain_complete_df = stain_initial_df.copy()

    setup_path = f"{name_setup}_setup.csv"
    setup_exists = os.path.exists(setup_path)
    key_cols = ["Condition", "Marker", "Laser"]
    value_cols = ["Cont_min", "Cont_max", "Gamma"]

    def _read_setup_csv():
        # Key columns as text: a numeric channel name (e.g. '405') would
        # otherwise come back as an int and never match the stain table.
        df = pd.read_csv(setup_path, dtype={c: str for c in key_cols})
        df = df.set_index(key_cols)
        return df[~df.index.duplicated(keep="last")]

    if use_setup and setup_exists:
        stain_setup_df = _read_setup_csv()
        # One aligned lookup instead of per-row .loc on the (unsorted)
        # MultiIndex, which pandas warns about for every row.
        if stain_complete_df.index.isin(stain_setup_df.index).all():
            saved = stain_setup_df.reindex(stain_complete_df.index)
            for col in value_cols:
                stain_complete_df[col] = saved[col].to_numpy()
        else:
            use_setup = False

    if not use_setup or not setup_exists:
        stain_complete_df = stain_initial_df.copy()

        if automatic_contrast:
            for _, idx in _progress_iter(
                enumerate(stain_complete_df.index),
                progress,
                desc="Step 07B - Automatic Contrast",
                total=len(stain_complete_df.index),
                leave=False,
            ):
                im_view = _normalize_channel_to_uint8_view(im_in[:, :, :, _])
                cont_min, cont_max = _compute_automatic_contrast_limits(im_view, auto_method)
                stain_complete_df.loc[idx, "Cont_min"] = cont_min
                stain_complete_df.loc[idx, "Cont_max"] = cont_max
                stain_complete_df.loc[idx, "Gamma"] = 1.0

            rule = ("background - 0.25 noise to background + 2 x (p99.95 - background), "
                    "calibrated on the nuclei ground truths"
                    if auto_method == "calibrated"
                    else "histogram peak + 1 to the 99th percentile intensity")
            print(
                ("calibrated_contrast is True" if automatic_contrast == "calibrated"
                 else f"automatic_contrast is {automatic_contrast!r}")
                + " — skipped the napari viewer and "
                f"picked Cont_min/Cont_max per channel automatically ({rule}); Gamma left at 1.0."
            )
        else:
            if settings is None:
                from napari.settings import get_settings
                settings = get_settings()
            if napari_module is None:
                import napari as napari_module

            _close_all_napari_viewers(napari_module)
            settings.application.ipy_interactive = False
            viewer = napari_module.Viewer(title="Channels setup - adjust contrast and gamma, then close viewer to continue", ndisplay=3)

            for _, idx in _progress_iter(
                enumerate(stain_complete_df.index),
                progress,
                desc="Step 07B - Prepare Setup Viewer",
                total=len(stain_complete_df.index),
                leave=False,
            ):
                im_view = _normalize_channel_to_uint8_view(im_in[:, :, :, _])
                layer = viewer.add_image(
                    im_view,
                    name=f"{idx[0]} ({idx[1]})",
                    colormap=_napari_colormap(stain_initial_df.loc[idx]["Color"]),
                    blending="additive",
                    scale=(r_xyz[2], r_xyz[1], r_xyz[0]) if r_xyz is not None else None,
                )
                if suggest_contrast:  # v1.6.4: start from the calibrated suggestion
                    layer.contrast_limits_range = (0, 255)
                    layer.contrast_limits = _compute_automatic_contrast_limits(
                        im_view, "calibrated")
                    layer.gamma = 1.0

            print(
                "A napari window has opened — check it, adjust the contrast and gamma "
                "for each channel, then close the window to continue."
                + (" Each channel starts at the suggested (calibrated) contrast."
                   if suggest_contrast else "")
            )
            _run_blocking_viewer(viewer, napari_module)
            settings.application.ipy_interactive = True
            image_layers = [layer for layer in viewer.layers if isinstance(layer, napari_module.layers.Image)]
            contrast_limits = {layer.name: layer.contrast_limits for layer in image_layers}
            gamma_values = {layer.name: layer.gamma for layer in image_layers}

            stain_complete_df.sort_index(inplace=True)
            for _, idx in _progress_iter(
                enumerate(stain_complete_df.index),
                progress,
                desc="Step 07C - Collect Setup Values",
                total=len(stain_complete_df.index),
                leave=False,
            ):
                name = f"{idx[0]} ({idx[1]})"
                stain_complete_df.loc[idx, "Cont_min"] = int(contrast_limits[name][0])
                stain_complete_df.loc[idx, "Cont_max"] = int(contrast_limits[name][1])
                stain_complete_df.loc[idx, "Gamma"] = gamma_values[name]

        current_setup_df = stain_complete_df[value_cols]
        if setup_exists:
            # Keep the saved rows of other conditions/markers, replace the
            # current ones (concat, not row-by-row .loc enlargement, which
            # pandas 2.x rejects on a MultiIndex with int columns).
            saved_df = _read_setup_csv().reindex(columns=value_cols)
            saved_df = saved_df[~saved_df.index.isin(current_setup_df.index)]
            stain_setup_df = pd.concat([saved_df, current_setup_df])
        else:
            stain_setup_df = current_setup_df.copy()

        stain_csv_setup_df = stain_setup_df.reset_index().sort_values(by="Condition")
        stain_csv_setup_df = stain_csv_setup_df[["Condition", "Marker", "Laser", "Cont_min", "Cont_max", "Gamma"]]
        stain_csv_setup_df.to_csv(setup_path, index=False)

    stain_df = stain_df.set_index("Condition")
    stain_complete_df = stain_complete_df.reset_index().set_index("Condition")
    stain_complete_df = stain_complete_df.loc[stain_df.index]
    stain_complete_df = stain_complete_df[["Marker", "Laser", "Color", "Cont_min", "Cont_max", "Gamma"]]
    original_stain_complete_df = stain_complete_df.copy()
    if suggest_contrast:  # v1.6.4: check the chosen values against the suggestion
        check = suggest_contrast_settings(im_in, stain_complete_df, current=stain_complete_df)
        flagged = check[check["Check"] != "ok"]
        print("Contrast check against the calibrated suggestion: "
              + ("all channels ok." if flagged.empty else f"{len(flagged)} channel(s) to look at:"))
        for cond, row in flagged.iterrows():
            print(f"  {cond}: {row['Check']}")
    return stain_df, stain_complete_df, original_stain_complete_df


def export_channel_histograms(
    im_final_stack,
    stain_complete_df,
    input_file,
    ROI_print=None,
    lazy_loading_used=None,
    name_setup='',
    nuclei_diameter=None,
    cell_diameter=None,
    cyto_factor=None,
    pcm_thickness_factor=None,
    zoom_factors=None,
    scale_factor=1.0,
    trig_cellpose=False,
    trig_stardist=False,
    trig_cellpose_cyto=False,
    multilabel=False,
    cyto_markers=None,
    nuclei_split_config=None,
    sigma=0.5,
    num_plateaus=2,
    plateau_factor=0.7,
    threshold_method='otsu',
    global_threshold_weight=0.15,
    sauvola_weight=0.60,
    aggregate_distance_factor=0.05,
    progress=None,
    output_dir=None,
    membrane_markers=None,
    extra_params=None,
):
    """
    Export per-channel intensity histograms for every processing stage in
    im_final_stack to a single Excel workbook.

    One sheet is written per stage (named by the stage abbreviation).  Each
    sheet contains a 'Pixel_Value' column followed by four columns per channel
    (Count, Percentage, Cumulative Count, Cumulative %) grouped under a merged,
    colour-coded title cell matching the acquisition colour of that marker.
    White channels are displayed as light grey.  A final 'Parameters' sheet
    records the stain settings and any supplied image processing parameters.

    Parameters
    ----------
    im_final_stack : dict
        Ordered dict mapping stage names (e.g. 'Adjusted image') to
        (Z, Y, X, C) ndarray image stacks.
    stain_complete_df : DataFrame
        Staining metadata; provides per-channel marker names, colors, and
        contrast/gamma settings.
    output_path : str or Path
        Destination Excel file path.
    processing_params : dict, optional
        Key-value pairs of image processing parameters to record on the
        Parameters sheet (e.g. {'sigma': 0.5, 'threshold_method': 'otsu'}).
    progress : callable, optional
        Progress wrapper (e.g. tqdm).
    output_dir : str or Path, optional
        Folder to write the workbook to (default: current working directory).
    extra_params : dict, optional
        More parameter/value pairs for the Parameters sheet (e.g. the
        LIVE/DEAD settings).

    Returns
    -------
    output_path : str or Path
    """
    from helpers.notebook_setup_helpers import stardist_available

    output_path = _stem_output_path(input_file, '_histograms.xlsx', output_dir)
    if nuclei_split_config is None:
        nuclei_split_config = get_nuclei_split_config(profile="balanced")
    # Record the method actually used (segment_nuclei warns when it falls back).
    trig_stardist = bool(trig_stardist) and stardist_available()

    processing_params = {
        'Input file':                     str(input_file),
        'ROI [x0,x1,y0,y1,z0,z1]':        str(ROI_print),
        'Lazy (chunked) loading used':    str(lazy_loading_used),
        'Experiment name':                str(name_setup),
        'Nuclei diameter (um)':           str(nuclei_diameter),
        'Cell diameter (um)':             str(cell_diameter),
        'Cyto factor':                    str(cyto_factor),
        'PCM thickness factor (x cell diameter)': str(pcm_thickness_factor),
        'Zoom factors [Z,Y,X]':           str(zoom_factors),
        'Scale factor':                   str(scale_factor),
        'Use Cellpose':                   str(trig_cellpose),
        'Use StarDist':                   str(trig_stardist),
        'Use Cellpose (cytoplasm shape)': str(trig_cellpose_cyto),
        'Multilabel':                     str(multilabel),
        'Cyto markers':                   str(cyto_markers),
        'Nuclei split config':            str(nuclei_split_config),
        'Sigma (Gaussian smoothing)':     str(sigma),
        'Num plateaus (histogram eq.)':   str(num_plateaus),
        'Plateau factor (histogram eq.)': str(plateau_factor),
        'Threshold method':               str(threshold_method),
        'Global threshold weight':        str(global_threshold_weight),
        'Sauvola (local) threshold weight': str(sauvola_weight),
        'Aggregate distance factor (x cell diameter)': str(aggregate_distance_factor),
    }
    if membrane_markers is not None:
        processing_params['Membrane markers'] = str(membrane_markers)
    for key, value in (extra_params or {}).items():
        processing_params[str(key)] = str(value)

    _stage_abbrev = {
        'Original image':   'Orig',
        'Normalized image': 'Norm',
        'Zoomed image':     'Zoom',
        'Denoised image':   'Denoise',
        'Adjusted image':   'Adj',
        'Filtered image':   'Filt',
        'Equalized image':  'Eq',
        'Threshold image':  'Thresh',
    }

    # Pastel background colours for channel title rows (napari colour names → hex)
    _channel_bg = {
        'BLUE':    '#BDD7EE',
        'GREEN':   '#C6EFCE',
        'RED':     '#FFC7CE',
        'CYAN':    '#CCFFFF',
        'MAGENTA': '#FFB3FF',
        'YELLOW':  '#FFFF99',
        'WHITE':   '#D9D9D9',
        'GREY':    '#D9D9D9',
        'GRAY':    '#D9D9D9',
        'ORANGE':  '#FFD966',
    }
    _default_bg = '#F2F2F2'

    _sub_cols = ['Count', 'Percentage', 'Cum. Count', 'Cum. %']
    n_sub = len(_sub_cols)  # 4 columns per channel

    _skip_stages = {'Original image', 'Threshold image'}

    # Keep only stages that contain 4-D image arrays, excluding non-informative stages
    stages = [
        (name, arr)
        for name, arr in im_final_stack.items()
        if isinstance(arr, np.ndarray) and arr.ndim == 4 and name not in _skip_stages
    ]
    # Skipped processing steps (Cells 9-14) only hold their input again: no sheet.
    steps_run = im_final_stack.get(_STEPS_RUN_KEY)
    if steps_run is not None:
        stages = [(name, arr) for name, arr in stages
                  if name not in _STAGE_NAMES or name in steps_run]

    n_channels = stain_complete_df.shape[0]

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Common formats (created once, shared across all sheets)
        _hdr = {'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'}
        pv_fmt  = workbook.add_format({**_hdr, 'bg_color': '#D9E1F2'})
        sub_fmt = workbook.add_format({**_hdr, 'bg_color': '#EBF0F9', 'text_wrap': True, 'font_size': 9})
        num_fmt = workbook.add_format({'num_format': '0.00'})
        int_fmt = workbook.add_format({})

        for stage_name, arr in _progress_iter(stages, progress, desc="Export Histograms"):
            abbrev = _stage_abbrev.get(stage_name, stage_name.split()[0][:8])
            ws = workbook.add_worksheet(abbrev[:31])

            TITLE_ROW = 0
            HDR_ROW   = 1
            DATA_ROW  = 2

            # Pixel_Value header: merges across both header rows for a clean look
            ws.merge_range(TITLE_ROW, 0, HDR_ROW, 0, 'Pixel_Value', pv_fmt)
            ws.set_column(0, 0, 11)

            for c in range(n_channels):
                condition = stain_complete_df.index[c]
                marker    = stain_complete_df.loc[condition, 'Marker']
                color_key = str(stain_complete_df.loc[condition, 'Color']).upper()
                bg = _channel_bg.get(color_key, _default_bg)

                # Per-channel title format using the channel's pastel colour
                ch_fmt = workbook.add_format({**_hdr, 'bg_color': bg, 'font_size': 11})

                col_start = 1 + c * n_sub
                col_end   = col_start + n_sub - 1

                # Coloured merged title spanning the 4 sub-columns
                ws.merge_range(TITLE_ROW, col_start, TITLE_ROW, col_end, marker, ch_fmt)

                # Sub-column headers
                for j, sub in enumerate(_sub_cols):
                    ws.write(HDR_ROW, col_start + j, sub, sub_fmt)

                # Histogram data for this channel
                im3d = arr[:, :, :, c].copy()
                values, counts = np.unique(im3d.astype(int), return_counts=True)
                hist = np.zeros(256, dtype=int)
                valid = (values >= 0) & (values <= 255)
                hist[values[valid]] = counts[valid]

                total = hist.sum()
                pct   = (hist / total * 100) if total > 0 else np.zeros(256)
                cum_c = np.cumsum(hist)
                cum_p = np.cumsum(pct)

                for pv in range(256):
                    ws.write_number(DATA_ROW + pv, col_start,     int(hist[pv]),    int_fmt)
                    ws.write_number(DATA_ROW + pv, col_start + 1, float(pct[pv]),   num_fmt)
                    ws.write_number(DATA_ROW + pv, col_start + 2, int(cum_c[pv]),   int_fmt)
                    ws.write_number(DATA_ROW + pv, col_start + 3, float(cum_p[pv]), num_fmt)

                ws.set_column(col_start, col_end, 13)

            # Pixel_Value data column (0–255)
            for pv in range(256):
                ws.write_number(DATA_ROW + pv, 0, pv, int_fmt)

            ws.freeze_panes(DATA_ROW, 0)
            ws.set_row(TITLE_ROW, 18)
            ws.set_row(HDR_ROW, 30)

        # --- Parameters sheet ---
        params_ws  = workbook.add_worksheet('Parameters')
        bold       = workbook.add_format({'bold': True})
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})

        row = 0

        # Section 1: Stain settings
        params_ws.write(row, 0, 'STAIN SETTINGS', bold)
        row += 1
        stain_reset = stain_complete_df.reset_index()
        for col_idx, col_name in enumerate(stain_reset.columns):
            params_ws.write(row, col_idx, str(col_name), header_fmt)
        row += 1
        for _, df_row in stain_reset.iterrows():
            for col_idx, val in enumerate(df_row):
                params_ws.write(row, col_idx, str(val))
            row += 1

        row += 1  # blank separator

        # Section 2: Image processing parameters
        params_ws.write(row, 0, 'IMAGE PROCESSING PARAMETERS', bold)
        row += 1
        params_ws.write(row, 0, 'Parameter', header_fmt)
        params_ws.write(row, 1, 'Value', header_fmt)
        row += 1
        if processing_params:
            for k, v in processing_params.items():
                params_ws.write(row, 0, str(k))
                params_ws.write(row, 1, str(v))
                row += 1

        params_ws.set_column(0, 0, 35)
        params_ws.set_column(1, 10, 20)

    print(f"Histograms saved to: {output_path}")
    return output_path


def build_labels_dict(
    im_segmentation_stack,
    filtered_img,
    stain_complete_df,
    stain_df,
    r_xyz,
    zooms,
    multilabel=False,
    percell_mean_df=None,
    percell_std_df=None,
    progress=None,
):
    """Build the compact per-marker quantification dictionary used by the notebook.

    ``percell_mean_df``/``percell_std_df`` (optional, see
    ``compute_percell_marker_intensity_df``) add the "(all cells)" avg/std
    columns per single-marker row -- the whole segmented population's
    intensity for that channel, not just cells that passed threshold.
    """
    labels_dict = {}
    nuc_positions, nuc_sizes, cyto_positions, cyto_sizes = compute_nuclei_cytoplasm_stats(
        im_segmentation_stack,
        r_xyz,
        zooms,
        progress=progress,
    )

    if "NUCLEI" in stain_complete_df.index:
        nuc_row = stain_complete_df.loc["NUCLEI"]
        labels_dict[nuc_row["Marker"]] = _make_labels_record(
            "NUCLEI",
            nuc_row["Laser"],
            nuc_row["Color"],
            np.max(im_segmentation_stack["Nuclei"]),
            nucleus_positions=nuc_positions,
            nucleus_sizes=nuc_sizes,
        )

    if "CYTOPLASM" in stain_complete_df.index:
        cyto_row = stain_complete_df.loc["CYTOPLASM"]
        labels_dict[cyto_row["Marker"]] = _make_labels_record(
            "CYTOPLASM",
            cyto_row["Laser"],
            cyto_row["Color"],
            np.max(im_segmentation_stack["Cytoplasm"]),
            cytoplasm_positions=cyto_positions,
            cytoplasm_sizes=cyto_sizes,
        )

    num_channels = filtered_img.shape[3]
    for c in _progress_iter(range(num_channels), progress, desc="Step 23A - Quantify Marker Stats"):
        condition = stain_complete_df.index[c]
        if condition in ["NUCLEI", "CYTOPLASM", "PCM"]:
            continue

        marker = stain_complete_df.loc[condition, "Marker"]
        (
            shared_labels,
            m_sizes,
            m_avg,
            m_std,
            m_cyto_sizes,
            m_cyto_avg,
            m_cyto_std,
            m_pcm_sizes,
            m_pcm_avg,
            m_pcm_std,
        ) = compute_marker_stats_for_marker(c, im_segmentation_stack, filtered_img, r_xyz, zooms, progress=progress)

        avg_marker_all = ()
        std_marker_all = ()
        if percell_mean_df is not None and condition in percell_mean_df.columns:
            avg_marker_all = percell_mean_df[condition].to_numpy()
        if percell_std_df is not None and condition in percell_std_df.columns:
            std_marker_all = percell_std_df[condition].to_numpy()

        labels_dict[marker] = _make_labels_record(
            condition,
            stain_complete_df.loc[condition, "Laser"],
            stain_complete_df.loc[condition, "Color"],
            len(shared_labels),
            shared_labels=sorted(shared_labels),
            nucleus_positions=(nuc_positions[i - 1] for i in shared_labels),
            cytoplasm_positions=(cyto_positions[i - 1] for i in shared_labels),
            nucleus_sizes=(nuc_sizes[i - 1] for i in shared_labels),
            cytoplasm_sizes=m_cyto_sizes,
            marker_sizes=m_sizes,
            avg_marker=m_avg,
            std_marker=m_std,
            marker_cyto_sizes=m_cyto_sizes,
            avg_cyto_marker=m_cyto_avg,
            std_cyto_marker=m_cyto_std,
            marker_pcm_sizes=m_pcm_sizes,
            avg_pcm_marker=m_pcm_avg,
            std_pcm_marker=m_pcm_std,
            avg_marker_all=avg_marker_all,
            std_marker_all=std_marker_all,
        )

    if multilabel:
        non_nuc_channels = [
            i for i in range(num_channels) if stain_complete_df.index[i] not in ["NUCLEI", "CYTOPLASM", "PCM"]
        ]
        max_combo_size = min(3, max(2, len(non_nuc_channels)))
        marker_index_to_shared = {}
        for c in _progress_iter(non_nuc_channels, progress, desc="Step 23B - Prepare Multilabel Sets"):
            marker_name = stain_complete_df.iloc[c]["Marker"]
            marker_index_to_shared[c] = set(labels_dict.get(marker_name, [(), (), (), (), ()])[4])

        from itertools import combinations

        for combo_size in _progress_iter(range(2, max_combo_size + 1), progress, desc="Step 23C - Build Multilabel Combos"):
            for comb in combinations(non_nuc_channels, combo_size):
                combo_markers = tuple(stain_complete_df.iloc[i]["Marker"] for i in comb)
                combo_sets = [marker_index_to_shared.get(i, set()) for i in comb]
                if not combo_sets:
                    continue
                combo_labels = sorted(set.intersection(*combo_sets))
                if not combo_labels:
                    continue

                labels_dict[combo_markers] = _make_labels_record(
                    tuple(stain_complete_df.index[i] for i in comb),
                    (),
                    (),
                    len(combo_labels),
                    shared_labels=combo_labels,
                    nucleus_positions=(nuc_positions[i - 1] for i in combo_labels),
                    cytoplasm_positions=(cyto_positions[i - 1] for i in combo_labels),
                    nucleus_sizes=(nuc_sizes[i - 1] for i in combo_labels),
                    cytoplasm_sizes=(cyto_sizes[i - 1] for i in combo_labels),
                )

    return labels_dict


def build_full_labels_dict(
    im_segmentation_stack,
    im_final_stack,
    filtered_img,
    stain_complete_df,
    r_xyz,
    zooms,
    percell_mean_df=None,
    percell_std_df=None,
    progress=None,
):
    """Build the full per-marker quantification dictionary for exports and meshes.

    ``percell_mean_df``/``percell_std_df`` (optional, see
    ``compute_percell_marker_intensity_df``) add the "(all cells)" avg/std
    columns per single-marker row -- see ``build_labels_dict``.
    """
    labels_full_dict = {}
    nuc_positions, nuc_sizes, cyto_positions, cyto_sizes = compute_nuclei_cytoplasm_stats(
        im_segmentation_stack,
        r_xyz,
        zooms,
        progress=progress,
    )

    if "NUCLEI" in stain_complete_df.index:
        nuc_row = stain_complete_df.loc["NUCLEI"]
        labels_full_dict[nuc_row["Marker"]] = _make_labels_record(
            "NUCLEI",
            nuc_row["Laser"],
            nuc_row["Color"],
            np.max(im_segmentation_stack["Nuclei"]),
            nucleus_positions=nuc_positions,
            nucleus_sizes=nuc_sizes,
        )

    if "CYTOPLASM" in stain_complete_df.index:
        cyto_row = stain_complete_df.loc["CYTOPLASM"]
        labels_full_dict[cyto_row["Marker"]] = _make_labels_record(
            "CYTOPLASM",
            cyto_row["Laser"],
            cyto_row["Color"],
            np.max(im_segmentation_stack["Cytoplasm"]),
            cytoplasm_positions=cyto_positions,
            cytoplasm_sizes=cyto_sizes,
        )

    num_channels = filtered_img.shape[3]
    for c in _progress_iter(range(num_channels), progress, desc="Step 25 - Quantify Full Marker Stats"):
        condition = stain_complete_df.index[c]
        if condition in ["NUCLEI", "CYTOPLASM", "PCM"]:
            continue

        marker = stain_complete_df.iloc[c]["Marker"]
        (
            full_labels,
            m_full_sizes,
            m_full_avg,
            m_full_std,
            m_full_cyto_sizes,
            m_full_cyto_avg,
            m_full_cyto_std,
            m_full_pcm_sizes,
            m_full_pcm_avg,
            m_full_pcm_std,
        ) = compute_full_marker_stats_for_marker(c, im_final_stack, im_segmentation_stack, filtered_img, r_xyz, zooms, progress=progress)

        avg_marker_all = ()
        std_marker_all = ()
        if percell_mean_df is not None and condition in percell_mean_df.columns:
            avg_marker_all = percell_mean_df[condition].to_numpy()
        if percell_std_df is not None and condition in percell_std_df.columns:
            std_marker_all = percell_std_df[condition].to_numpy()

        labels_full_dict[marker] = _make_labels_record(
            condition,
            stain_complete_df.iloc[c]["Laser"],
            stain_complete_df.iloc[c]["Color"],
            len(full_labels),
            shared_labels=sorted(full_labels),
            nucleus_positions=(nuc_positions[i - 1] for i in full_labels),
            cytoplasm_positions=(cyto_positions[i - 1] for i in full_labels),
            nucleus_sizes=(nuc_sizes[i - 1] for i in full_labels),
            cytoplasm_sizes=m_full_cyto_sizes,
            marker_sizes=m_full_sizes,
            avg_marker=m_full_avg,
            std_marker=m_full_std,
            marker_cyto_sizes=m_full_cyto_sizes,
            avg_cyto_marker=m_full_cyto_avg,
            std_cyto_marker=m_full_cyto_std,
            marker_pcm_sizes=m_full_pcm_sizes,
            avg_pcm_marker=m_full_pcm_avg,
            std_pcm_marker=m_full_pcm_std,
            avg_marker_all=avg_marker_all,
            std_marker_all=std_marker_all,
        )

    return labels_full_dict


def build_labels_df(
    im_segmentation_stack,
    im_final_stack,
    stain_complete_df,
    stain_df,
    r_zX,
    r_zY,
    r_zZ,
    zooms,
    multilabel=False,
    percell_mean_df=None,
    percell_std_df=None,
    progress=None,
    is_2d=False,
):
    """Build the compact quantification dictionary, display a preview, and return the DataFrame.

    ``percell_mean_df``/``percell_std_df`` (optional, see
    ``compute_percell_marker_intensity_df``) add the "(all cells)" avg/std
    columns per single-marker row. ``is_2d`` labels the sizes as areas.
    """
    filtered_img = im_final_stack['Filtered image']
    r_xyz = (r_zX, r_zY, r_zZ)
    labels_dict = build_labels_dict(
        im_segmentation_stack,
        filtered_img,
        stain_complete_df=stain_complete_df,
        stain_df=stain_df,
        r_xyz=r_xyz,
        zooms=zooms,
        multilabel=multilabel,
        percell_mean_df=percell_mean_df,
        percell_std_df=percell_std_df,
        progress=progress,
    )
    labels_df, truncated_df = labels_dict_to_dataframe(labels_dict, truncate=True, progress=progress,
                                                       is_2d=is_2d)
    try:
        from IPython.display import display as _display
        _display(truncated_df)
    except Exception:
        pass
    return labels_df


def build_full_labels_df(
    im_segmentation_stack,
    im_final_stack,
    stain_complete_df,
    r_zX,
    r_zY,
    r_zZ,
    zooms,
    percell_mean_df=None,
    percell_std_df=None,
    progress=None,
    is_2d=False,
):
    """Build the full quantification dictionary and convert it to a DataFrame.

    Uses zoomed voxel sizes (r_zX, r_zY, r_zZ) because the segmentation arrays
    are in zoomed pixel space — consistent with build_labels_df (Cell 24).

    ``percell_mean_df``/``percell_std_df`` (optional, see
    ``compute_percell_marker_intensity_df``) add the "(all cells)" avg/std
    columns per single-marker row. ``is_2d`` labels the sizes as areas.
    """
    filtered_img = im_final_stack['Filtered image']
    r_xyz = (r_zX, r_zY, r_zZ)
    labels_full_dict = build_full_labels_dict(
        im_segmentation_stack,
        im_final_stack,
        filtered_img,
        stain_complete_df=stain_complete_df,
        r_xyz=r_xyz,
        zooms=zooms,
        percell_mean_df=percell_mean_df,
        percell_std_df=percell_std_df,
        progress=progress,
    )
    return labels_dict_to_dataframe(labels_full_dict, is_2d=is_2d)


def _area_column_names(columns):
    """2D images: the sizes are areas, so '[um3]' columns become '[um2]'."""
    return {c: c.replace("[um3]", "[um2]") for c in columns if "[um3]" in c}


def labels_dict_to_dataframe(labels_dict, truncate=False, progress=None, is_2d=False):
    """Convert a quantification dictionary to the standard notebook DataFrame.

    With ``is_2d`` the size columns are labelled as areas ('[um2]'): for a
    single-plane image the Z voxel size is 1 um, so the values are areas.
    """
    labels_df = pd.DataFrame.from_dict(labels_dict, orient="index", columns=LABELS_TABLE_COLUMNS)
    labels_df.index.name = "Combination"

    if not truncate:
        return labels_df.rename(columns=_area_column_names(labels_df.columns)) if is_2d else labels_df

    truncated_df = labels_df.copy()
    truncate_columns = [
        "Shared labels",
        "Mean nuclei positions [um]",
        "Mean cytoplasm positions [um]",
        "Nuclei size [um3]",
        "Cytoplasm size [um3]",
        "Marker size [um3]",
        "Avg. marker intensity",
        "STD marker intensity",
        "Marker size cytoplasm [um3]",
        "Avg. marker intensity cytoplasm",
        "STD marker intensity cytoplasm",
        "Marker size PCM [um3]",
        "Avg. marker intensity PCM",
        "STD marker intensity PCM",
        "Avg. marker intensity (all cells)",
        "STD marker intensity (all cells)",
    ]
    for column in _progress_iter(truncate_columns, progress, desc="Step 23D - Truncate Display Columns"):
        truncated_df[column] = truncated_df[column].apply(lambda value: truncate_cell(value))

    if is_2d:
        labels_df = labels_df.rename(columns=_area_column_names(labels_df.columns))
        truncated_df = truncated_df.rename(columns=_area_column_names(truncated_df.columns))
    return labels_df, truncated_df


def _volume_column_names(df):
    """``(df, is_2d)``: a labels table built with ``is_2d=True`` has its size
    columns as '[um2]' (areas). They are renamed back to the '[um3]' names
    the readers below use, and ``is_2d`` tells them to label areas."""
    is_2d = any("[um2]" in str(c) for c in df.columns)
    if is_2d:
        df = df.rename(columns={c: str(c).replace("[um2]", "[um3]")
                                for c in df.columns if "[um2]" in str(c)})
    return df, is_2d


def print_population_summary(labels_df, stain_complete_df, stain_df, progress=None):
    """Print the compact summary block used in the analysis section.

    Sizes are areas (um²) when ``labels_df`` comes from a 2D image
    (``build_labels_df(..., is_2d=True)``), volumes (um³) otherwise.

    When ``labels_df`` was built with ``percell_mean_df`` (see
    ``build_labels_df`` / ``compute_percell_marker_intensity_df``), prints a
    second "Marker intensity (all cells)" line per marker -- the whole
    segmented population's mean intensity, positive or not -- next to the
    existing "Marker intensity (positive cells)" line, which only covers
    cells where that marker's threshold mask fired. Skipped for multi-marker
    combination rows, since that column only exists for single marker
    channels.
    """
    labels_df, is_2d = _volume_column_names(labels_df)
    size_unit = "um²" if is_2d else "um³"
    size_word = "AREA" if is_2d else "SIZE"
    nuclei_rows = labels_df[labels_df["Condition"] == "NUCLEI"]
    cyto_rows = labels_df[labels_df["Condition"] == "CYTOPLASM"]
    total_nuclei = float(nuclei_rows.iloc[0]["Number"]) if not nuclei_rows.empty else float(labels_df.iloc[0]["Number"])
    # Percentages are of the analysed cells: nuclei without a cytoplasm are
    # excluded (see segment_cytoplasm).
    total_cells = total_nuclei
    if not cyto_rows.empty and float(cyto_rows.iloc[0]["Number"]) > 0:
        total_cells = float(cyto_rows.iloc[0]["Number"])

    # --- Channel overview ---
    if is_2d:
        print("2D IMAGE: nucleus and cytoplasm sizes are areas [um²].")
    print("CHANNELS:")
    for i, cond in enumerate(stain_complete_df.index):
        row = stain_complete_df.loc[cond]
        print(f"  [{i}] {cond:<20} {row['Marker']:<16} ({row['Laser']})")

    print("_" * 80)
    print("TOT CELLS =", int(total_cells))
    if total_nuclei > total_cells:
        print(f"  ({int(total_nuclei)} nuclei, {int(total_nuclei - total_cells)} "
              "excluded: no cytoplasm beyond the nucleus)")
    print(" ")
    for _, marker in _progress_iter(
        enumerate(labels_df.index),
        progress,
        desc="Step 24A - Summary Percentages",
        total=len(labels_df.index),
    ):
        condition = labels_df.iloc[_]["Condition"]
        if condition not in ["NUCLEI", "CYTOPLASM", "NUCLEI + CYTOPLASM"]:
            count = int(labels_df.iloc[_]["Number"])
            perc = 100.0 * count / total_cells
            print(f"  PERC {condition} ({marker}) = {perc:.1f} %  ({count} cells)")

    print("_" * 80)

    # --- Nuclei and cytoplasm population size statistics ---
    def _size_stats_line(sizes_tuple, label, unit=size_unit):
        arr = np.array(sizes_tuple, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return
        print(
            f"{label}:  mean = {np.mean(arr):.2f} {unit}"
            f"  |  std = {np.std(arr):.2f} {unit}"
            f"  |  median = {np.median(arr):.2f} {unit}"
            f"  |  [min = {np.min(arr):.2f},  max = {np.max(arr):.2f}]"
        )

    if not nuclei_rows.empty:
        _size_stats_line(nuclei_rows.iloc[0]["Nuclei size [um3]"], f"NUCLEI {size_word}")
    if "CYTOPLASM" in stain_df.index and not cyto_rows.empty:
        _size_stats_line(cyto_rows.iloc[0]["Cytoplasm size [um3]"], f"CYTOPLASM {size_word}")

    print("_" * 80)

    # --- Per-condition detailed stats ---
    for _, marker in _progress_iter(
        enumerate(labels_df.index),
        progress,
        desc="Step 24B - Summary Per-Condition",
        total=len(labels_df.index),
    ):
        condition = labels_df.iloc[_]["Condition"]
        if condition in ["NUCLEI", "CYTOPLASM", "NUCLEI + CYTOPLASM"]:
            continue

        row = labels_df.iloc[_]
        count = int(row["Number"])
        perc = 100.0 * count / total_cells
        print(f"\n {condition} ({marker})  —  {count} cells  ({perc:.1f} %)")

        # Only marker intensity is printed per condition -- nucleus/cytoplasm
        # SIZE belongs to the segmentation, not to any one marker, and is
        # already reported once above under NUCLEI SIZE / CYTOPLASM SIZE.
        if "CYTOPLASM" in stain_df.index:
            _size_stats_line(row["Avg. marker intensity cytoplasm"], "   Cytoplasm intensity", unit="a.u.")

        # Marker intensity (mean/median/min/max across cells)
        _size_stats_line(row["Avg. marker intensity"], "   Marker intensity (positive cells)", unit="a.u.")

        # Whole-population intensity, including cells that didn't pass
        # threshold for this marker -- only present for single-marker rows
        # (see build_labels_df's percell_mean_df), not multi-marker combos.
        if "Avg. marker intensity (all cells)" in labels_df.columns:
            _size_stats_line(row["Avg. marker intensity (all cells)"], "   Marker intensity (all cells)", unit="a.u.")

    print("_" * 80)


def build_histogram_report(
    im_segmentation_stack,
    im_final_stack,
    stain_df,
    stain_complete_df,
    input_file,
    pad=20,
    thumb_size=None,
    progress=None,
    output_dir=None,
    keep_png=True,
    show_plot=True,
):
    """Collect histogram data, plot KDEs, and generate the per-nucleus PDF report.

    ``output_dir`` sets where the PDF is written (default: current working
    directory). With ``keep_png=False`` the per-nucleus PNGs embedded in the
    PDF are written to a temporary folder that is deleted afterwards, instead
    of the crop_png/merged_png/density_png subfolders. With
    ``show_plot=False`` the KDE overview figure is not drawn (``fig`` and
    ``axes`` are returned as None).
    """
    import tempfile
    from reportlab.lib.units import inch

    if thumb_size is None:
        thumb_size = (2.0 * inch, 2.0 * inch)

    filtered_img = im_final_stack['Filtered image']

    hist_data, intensity_ranges = collect_histogram_data(
        im_segmentation_stack,
        filtered_img,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        progress=progress,
    )

    if show_plot:
        fig, axes, x_grid = plot_nucleus_kdes(
            hist_data,
            stain_complete_df=stain_complete_df,
            progress=progress,
        )
    else:
        fig, axes, x_grid = None, None, np.linspace(0, 255, 400)

    set_notebook_context(
        seg_stack=im_segmentation_stack,
        filtered_img=filtered_img,
        hist_data=hist_data,
        x_grid=x_grid,
        stain_complete_df=stain_complete_df,
        stain_df=stain_df,
    )

    output_pdf = _stem_output_path(input_file, "_nuclei_marker.pdf", output_dir)
    if keep_png:
        create_row_pdf(
            output_pdf=output_pdf,
            pad=pad,
            thumb_size=thumb_size,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="napari_seg_png_", ignore_cleanup_errors=True) as png_dir:
            create_row_pdf(
                output_pdf=output_pdf,
                pad=pad,
                thumb_size=thumb_size,
                png_dir=png_dir,
            )

    return hist_data, intensity_ranges, fig, axes, x_grid


def collect_histogram_data(im_segmentation_stack, filtered_img, stain_df, stain_complete_df, progress=None):
    """Collect per-nucleus intensity values for all non-nuclear markers.

    Uses the assigned segmentation label image (from assign_channel_labels) to
    determine which voxels belong to a positive cell — matching the mask used by
    compute_full_marker_stats_for_marker so that avg/STD values in the PDF report
    are identical to those in the Excel quantification table.
    """
    hist_data = {}
    intensity_ranges = {}
    cell_slices = None

    for c in _progress_iter(range(filtered_img.shape[3]), progress, desc="Step 26A - Collect Histogram Data"):
        seg_key = stain_df.index[c]
        if seg_key in ("NUCLEI", "CYTOPLASM"):
            continue

        assigned = im_segmentation_stack.get(seg_key)
        if assigned is None:
            continue

        condition = stain_complete_df.index[c]
        intensity_img = filtered_img[:, :, :, c]
        intensity_ranges[condition] = (float(intensity_img.min()), float(intensity_img.max()))
        max_n = _n_cells(im_segmentation_stack)
        if cell_slices is None:  # each cell in its bounding box (same values)
            cell_slices = _union_label_slices(
                [im_segmentation_stack["Nuclei"], im_segmentation_stack["Cytoplasm"],
                 im_segmentation_stack["PCM"]], max_n)

        for nucleus_id in _progress_iter(range(1, max_n + 1), progress, desc=f"Step 26B - {condition} Nuclei"):
            hist_data.setdefault(nucleus_id, {})
            hist_data[nucleus_id].setdefault(condition, [])
            sl = cell_slices[nucleus_id - 1]
            if sl is None:
                continue

            nuc_mask  = im_segmentation_stack["Nuclei"][sl]    == nucleus_id
            cyto_mask = im_segmentation_stack["Cytoplasm"][sl] == nucleus_id
            pcm_mask  = im_segmentation_stack["PCM"][sl]       == nucleus_id
            cell_mask = (nuc_mask | cyto_mask | pcm_mask)
            mask_marker = (assigned[sl] > 0) & cell_mask

            if np.any(mask_marker):
                values = intensity_img[sl][mask_marker]
                if values.size > 0:
                    hist_data[nucleus_id][condition].extend(values.tolist())

    return hist_data, intensity_ranges


def plot_nucleus_kdes(hist_data, stain_complete_df, progress=None, max_subplots=20):
    """Plot one KDE row per nucleus for the collected intensity values."""
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    nuclei = sorted(hist_data.keys())
    if len(nuclei) == 0:
        raise ValueError("hist_data is empty. Fill hist_data before plotting.")

    all_conditions = sorted({condition for nucleus_data in hist_data.values() for condition in nucleus_data.keys()})
    max_subplots = min(max(len(nuclei), 1), max_subplots)
    height = min(3 * max_subplots, 60)

    fig, axes = plt.subplots(max_subplots, 1, figsize=(10, height), sharex=True)
    if max_subplots == 1:
        axes = [axes]

    x_grid = np.linspace(0, 255, 400)
    nuclei_to_plot = nuclei[:max_subplots]

    for idx, nucleus_id in _progress_iter(
        enumerate(nuclei_to_plot),
        progress,
        desc="Step 26C - Plot Nucleus KDEs",
        total=len(nuclei_to_plot),
    ):
        ax = axes[idx]
        ax.set_title(f"Nucleus {nucleus_id}")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

        for condition in all_conditions:
            vals = np.asarray(hist_data.get(nucleus_id, {}).get(condition, []))
            color = stain_complete_df.loc[condition, "Color"] if condition in stain_complete_df.index else "GRAY"
            if color == "WHITE":
                color = "GRAY"
            if vals.size == 0:
                continue

            try:
                kde = gaussian_kde(vals)
                y_grid = kde(x_grid)
            except Exception:
                mean_val = vals.mean()
                y_grid = np.exp(-0.5 * ((x_grid - mean_val) / 1.0) ** 2)

            y_norm = y_grid / y_grid.max() if y_grid.max() > 0 else y_grid
            ax.plot(x_grid, y_norm, linewidth=2, color=color)

            mean_val = float(vals.mean())
            std_val = float(vals.std())
            ax.axvline(mean_val, linestyle="--", linewidth=1.5, color=color)
            y_at_mean = np.interp(mean_val, x_grid, y_norm)
            ax.hlines(y_at_mean, max(0.0, mean_val - std_val), min(255.0, mean_val + std_val), linewidth=2, color=color)

        legend_handles = [
            Line2D([0], [0], color=(stain_complete_df.loc[c, "Color"] if stain_complete_df.loc[c, "Color"] != "WHITE" else "GRAY"), lw=2)
            for c in all_conditions
        ]
        ax.legend(legend_handles, all_conditions, loc="upper right", framealpha=0.9)
        if nucleus_id == nuclei_to_plot[-1]:
            ax.set_xlabel("Intensity (0–255)")

    axes[0].set_ylabel("Relative Density (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, axes, x_grid


def plot_spatial_distributions(labels_df, stain_complete_df, stain_df, im_final_stack, r_X, r_Y, r_Z, zoom_factors,
                               percell_mean_df=None, progress=None):
    """Plot X/Y/Z spatial distributions for each population.

    With ``percell_mean_df`` (see ``compute_percell_marker_intensity_df``),
    every analysed cell is also drawn as a dot per marker channel: its
    nucleus position on X and its mean intensity (all cells) on a secondary
    y axis, so the intensity can be followed along each direction. A 2D
    image (single plane) has X and Y only.
    """
    im_in = im_final_stack['Filtered image']
    n_axes = 2 if im_in.shape[0] == 1 else 3
    fig, axs = plt.subplots(n_axes, 1, figsize=(15, 5 * n_axes))

    for idx, marker in _progress_iter(
        enumerate(labels_df.index),
        progress,
        desc="Step 28 - Plot Spatial Distributions",
        total=len(labels_df.index),
    ):
        xcoor = [t[0] for t in labels_df.iloc[idx]["Mean nuclei positions [um]"]]
        ycoor = [t[1] for t in labels_df.iloc[idx]["Mean nuclei positions [um]"]]
        zcoor = [t[2] for t in labels_df.iloc[idx]["Mean nuclei positions [um]"]]

        xcount, xbins = np.histogram(xcoor, range=(0, im_in.shape[2] * r_X / zoom_factors[2]), bins=30)
        ycount, ybins = np.histogram(ycoor, range=(0, im_in.shape[1] * r_Y / zoom_factors[1]), bins=30)
        zcount, zbins = np.histogram(zcoor, range=(0, im_in.shape[0] * r_Z / zoom_factors[0]), bins=30)
        xbin_centers = (xbins[:-1] + xbins[1:]) / 2
        ybin_centers = (ybins[:-1] + ybins[1:]) / 2
        zbin_centers = (zbins[:-1] + zbins[1:]) / 2

        condition = labels_df.iloc[idx]["Condition"]
        if condition == "CYTOPLASM":
            continue
        color = _condition_color(condition, stain_complete_df, stain_df=stain_df)
        linestyle = '-' if np.size(marker) == 1 else (0, (2, max(np.size(marker) - 1, 1)))
        curves = [(xbin_centers, xcount), (ybin_centers, ycount), (zbin_centers, zcount)]
        for ax, (centers, counts) in zip(axs, curves[:n_axes]):
            ax.plot(centers, counts, label=str(condition), linestyle=linestyle, color=color)

    # Per-cell intensity vs position, on a secondary y axis. percell_mean_df
    # rows are cell labels 1..K, matching the first K nuclei positions.
    twin_axs = [None] * n_axes
    if percell_mean_df is not None and len(percell_mean_df):
        nuclei_rows = labels_df[labels_df["Condition"] == "NUCLEI"]
        pos_row = nuclei_rows.iloc[0] if not nuclei_rows.empty else labels_df.iloc[0]
        positions = np.asarray(list(pos_row["Mean nuclei positions [um]"]), dtype=float)
        labels = np.asarray(percell_mean_df.index, dtype=int)
        labels = labels[labels <= len(positions)]
        cell_pos = positions[labels - 1]
        twin_axs = [ax.twinx() for ax in axs]
        for condition in percell_mean_df.columns:
            if condition in ("NUCLEI", "CYTOPLASM", "PCM"):
                continue
            values = percell_mean_df.loc[labels, condition].to_numpy(dtype=float)
            keep = ~np.isnan(values)
            color = _condition_color(condition, stain_complete_df, stain_df=stain_df)
            for k, tax in enumerate(twin_axs):
                tax.scatter(cell_pos[keep, k], values[keep], s=10, alpha=0.5, color=color,
                            edgecolors="none", label=f"{condition} intensity (cells)")
        for tax in twin_axs:
            tax.set_ylabel("Mean intensity, all cells [a.u.]")

    for ax, tax, axis_name in zip(axs, twin_axs, "XYZ"):
        ax.set_title(f"NUCLEI {axis_name} DISTRIBUTION")
        ax.set_xlabel("[μm]")
        ax.set_ylabel("Cell count")
        ax.set_facecolor("black")
        handles, names = ax.get_legend_handles_labels()
        if tax is not None:
            t_handles, t_names = tax.get_legend_handles_labels()
            handles, names = handles + t_handles, names + t_names
        (tax if tax is not None else ax).legend(handles, names, loc="upper right", fontsize="small")
    return fig, axs


def plot_size_distributions(labels_df, stain_complete_df, stain_df, progress=None):
    """Plot nuclei and cytoplasm size histograms for each population (areas
    for a 2D image, see ``build_labels_df(..., is_2d=True)``)."""
    labels_df, is_2d = _volume_column_names(labels_df)
    size_word, size_unit = ("AREA", "[μm²]") if is_2d else ("SIZE", "[μm3]")
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    nuclei_max_size = max(x for values in labels_df["Nuclei size [um3]"] for x in values)
    cytoplasm_max_size = max(x for values in labels_df["Cytoplasm size [um3]"] for x in values)

    for idx, marker in _progress_iter(
        enumerate(labels_df.index),
        progress,
        desc="Step 29 - Plot Size Distributions",
        total=len(labels_df.index),
    ):
        nuclei_sizes = list(labels_df.iloc[idx]["Nuclei size [um3]"])
        cell_sizes = list(labels_df.iloc[idx]["Cytoplasm size [um3]"])
        condition = labels_df.iloc[idx]["Condition"]
        color = _condition_color(condition, stain_complete_df, stain_df=stain_df)

        if condition != "CYTOPLASM":
            axs[0].hist(
                nuclei_sizes,
                range=(0, nuclei_max_size),
                bins=30,
                label=str(condition),
                alpha=1 / max(len(labels_df), 1),
                color=color,
            )
        if condition != "NUCLEI":
            axs[1].hist(
                cell_sizes,
                range=(0, cytoplasm_max_size),
                bins=30,
                label=str(condition),
                alpha=1 / max(len(labels_df) - 1, 1),
                color=color,
            )

    axs[0].set_title(f"NUCLEI {size_word} DISTRIBUTION")
    axs[0].set_xlabel(size_unit)
    axs[0].legend(loc="upper right")
    axs[1].set_title(f"CELL {size_word} DISTRIBUTION")
    axs[1].set_xlabel(size_unit)
    axs[1].legend(loc="upper right")
    return fig, axs


def compute_percell_marker_intensity_df(
    im_segmentation_stack,
    im_final_stack,
    stain_complete_df,
    conditions=None,
    progress=None,
):
    """Build per-cell x per-marker mean/std intensity tables (one row per cell).

    Unlike ``labels_df``'s "Avg./STD marker intensity" columns -- which only
    cover cells where that marker's threshold mask fired at all -- this
    measures every segmented cell's intensity over its full volume (nucleus +
    cytoplasm + PCM), whether or not it passed threshold. That is what a
    flow-cytometry-style intensity cloud needs: the negative/dim population
    has to stay visible next to the positive one instead of being filtered
    out before plotting.

    Returns
    -------
    (percell_mean_df, percell_std_df) : tuple of DataFrame
        Both indexed by nucleus label, one column per condition in
        ``conditions`` (default: every real image channel except NUCLEI).
        ``percell_mean_df`` holds each cell's mean intensity over its whole
        volume; ``percell_std_df`` holds that same cell's own voxel-to-voxel
        std (signal heterogeneity within that one cell) -- mirroring the
        existing "Avg./STD marker intensity" pair, just unthresholded.
    """
    filtered_img = im_final_stack['Filtered image']
    nuclei_img = im_segmentation_stack['Nuclei']
    cyto_img = im_segmentation_stack.get('Cytoplasm')
    pcm_img = im_segmentation_stack.get('PCM')
    max_label = _n_cells(im_segmentation_stack)

    # stain_complete_df can carry extra rows with no matching image channel
    # (e.g. segment_cytoplasm appends a placeholder 'CYTOPLASM' row when
    # cytoplasm is derived from cyto_markers instead of a dedicated channel)
    # -- so channel indices must come positionally from the real channels
    # only, the same way build_full_labels_dict does it.
    num_channels = filtered_img.shape[3]
    real_channel_conditions = list(stain_complete_df.index[:num_channels])
    channel_idx = {c: i for i, c in enumerate(real_channel_conditions)}

    if conditions is None:
        conditions = [c for c in real_channel_conditions if c != "NUCLEI"]
    else:
        conditions = [c for c in conditions if c in channel_idx]

    mean_data = {condition: np.full(max_label, np.nan) for condition in conditions}
    std_data = {condition: np.full(max_label, np.nan) for condition in conditions}

    # Each cell in its bounding box (one whole-image pass per cell before):
    # the same voxels in the same order, so the same values.
    cell_slices = _union_label_slices([nuclei_img, cyto_img, pcm_img], max_label)
    for label_id in _progress_iter(
        range(1, max_label + 1), progress, desc="Step Q1 - Per-Cell Marker Intensity"
    ):
        sl = cell_slices[label_id - 1]
        if sl is None:
            continue
        cell_mask = nuclei_img[sl] == label_id
        if cyto_img is not None:
            cell_mask = cell_mask | (cyto_img[sl] == label_id)
        if pcm_img is not None:
            cell_mask = cell_mask | (pcm_img[sl] == label_id)
        if not np.any(cell_mask):
            continue
        filtered_c = filtered_img[sl]

        for condition in conditions:
            c = channel_idx.get(condition)
            if c is None:
                continue
            values = filtered_c[:, :, :, c][cell_mask]
            if values.size:
                mean_data[condition][label_id - 1] = float(np.mean(values))
                std_data[condition][label_id - 1] = float(np.std(values))

    index = pd.RangeIndex(1, max_label + 1, name="Cell label")
    percell_mean_df = pd.DataFrame(mean_data, index=index)
    percell_std_df = pd.DataFrame(std_data, index=index)
    return percell_mean_df, percell_std_df


def plot_marker_intensity_clouds(
    percell_mean_df,
    stain_complete_df,
    im_segmentation_stack,
    r_xyz=(1.0, 1.0, 1.0),
    stain_df=None,
    conditions=None,
    log_scale=False,
    point_size=10,
    alpha=0.4,
    bins=40,
    progress=None,
    input_file=None,
    output_dir=None,
    show=True,
    is_2d=False,
):
    """Flow-cytometry-style XY cloud: per-cell marker intensity vs cytoplasm size.

    With ``is_2d`` (single-plane image, Z voxel size 1 um) the size is
    labelled as an area in um2.

    Every segmented cell is one point per marker channel, all channels drawn
    in the same axes (one colour per marker) so their positivity can be
    compared directly. X is the cell's mean intensity for that channel (see
    ``compute_percell_marker_intensity_df``); Y is the cell's cytoplasm
    volume in um3 (from ``im_segmentation_stack['Cytoplasm']`` and the voxel
    size ``r_xyz``; falls back to the nuclei labels when there is no
    cytoplasm label image). Marginal histograms show the intensity
    distribution per marker (top) and the cytoplasm size distribution
    (right). In the main plot, dashed lines mark each marker's median
    intensity (and the median cytoplasm size, in grey). In the marginal
    histograms, dashed lines mark the mean and dotted lines mean +/- 1
    standard deviation (across cells), with the values printed as text.

    If *input_file* is given, the figure is also saved as
    ``<input stem>_marker_intensity.png`` in *output_dir* (default: current
    working directory). ``show=False`` skips displaying it (batch mode).
    """
    if conditions is None:
        conditions = list(percell_mean_df.columns)

    # Per-cell cytoplasm volume, indexed like percell_mean_df (label 1..N).
    seg_img = im_segmentation_stack.get("Cytoplasm")
    if seg_img is None:
        seg_img = im_segmentation_stack["Nuclei"]
    max_label = len(percell_mean_df)
    counts = np.bincount(np.asarray(seg_img).ravel(), minlength=max_label + 1)[1:max_label + 1]
    cell_size = pd.Series(
        counts * float(r_xyz[0]) * float(r_xyz[1]) * float(r_xyz[2]),
        index=percell_mean_df.index,
    )
    cell_size = cell_size.where(cell_size > 0)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    size_vals = cell_size.dropna()
    if not size_vals.empty:
        ax_right.hist(size_vals, bins=bins, orientation="horizontal", color="0.5", alpha=0.7)
        ax.axhline(float(size_vals.median()), color="0.3", linestyle="--", linewidth=1.4)
        size_mean = float(size_vals.mean())
        size_std = float(size_vals.std(ddof=0))
        ax_right.axhline(size_mean, color="black", linestyle="--", linewidth=1.4)
        for edge in (size_mean - size_std, size_mean + size_std):
            ax_right.axhline(edge, color="black", linestyle=":", linewidth=1.1)
        ax_right.text(
            0.95, 0.97, f"mean\n{size_mean:.0f}\n± {size_std:.0f}",
            transform=ax_right.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
        )
    top_stats = []

    # Shared bins so the per-marker histograms are comparable.
    all_vals = np.concatenate(
        [percell_mean_df[c].dropna().to_numpy() for c in conditions] or [np.array([])]
    )
    x_bins = bins
    if all_vals.size:
        x_bins = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    for condition in _progress_iter(conditions, progress, desc="Step Q2 - Plot Marker Intensity Clouds"):
        data = pd.DataFrame({"x": percell_mean_df[condition], "y": cell_size}).dropna()
        if data.empty:
            continue
        color = _condition_color(condition, stain_complete_df, stain_df=stain_df)
        ax.scatter(data["x"], data["y"], s=point_size, alpha=alpha, color=color,
                   edgecolors="none", label=str(condition))
        ax_top.hist(data["x"], bins=x_bins, histtype="stepfilled", color=color, alpha=0.35)
        ax_top.hist(data["x"], bins=x_bins, histtype="step", color=color, linewidth=1.2)
        ax.axvline(float(data["x"].median()), color=color, linestyle="--", linewidth=1.4)

        mean_x = float(data["x"].mean())
        std_x = float(data["x"].std(ddof=0))
        ax_top.axvline(mean_x, color=color, linestyle="--", linewidth=1.4)
        for edge in (mean_x - std_x, mean_x + std_x):
            ax_top.axvline(edge, color=color, linestyle=":", linewidth=1.1)
        top_stats.append((f"{condition}: {mean_x:.1f} ± {std_x:.1f}", color))

    if log_scale:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Mean intensity per cell (a.u.)")
    ax.set_ylabel("Cytoplasm area [um2]" if is_2d else "Cytoplasm size [um3]")
    ax.grid(alpha=0.2)
    ax.legend(title="Marker", loc="upper right", markerscale=2)
    if top_stats:
        # One coloured line of text per marker, stacked in the top-left corner.
        for k, (text, color) in enumerate(top_stats):
            ax_top.text(
                0.01, 0.95 - 0.14 * k, text, transform=ax_top.transAxes,
                ha="left", va="top", fontsize=8, color=color, fontweight="bold",
            )
    ax_top.set_title("Per-cell marker intensity vs cytoplasm " + ("area" if is_2d else "size"))
    ax_top.set_ylabel("Cells")
    ax_right.set_xlabel("Cells")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    if input_file is not None:
        png_path = _stem_output_path(input_file, "_marker_intensity.png", output_dir)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Marker intensity plot saved to: {png_path}")
    if show:
        plt.show()
    return fig, ax


def export_quantification_to_excel(input_file, original_stain_complete_df, labels_full_df, progress=None,
                                   output_dir=None):
    """Write the main quantification report to Excel.

    Generates ``<stem>_segmentation.xlsx`` next to *input_file* (or inside
    *output_dir*, if given) with four sheets styled to match the histogram
    workbook:

    * **Setup** — channel stain table with colour-coded header per marker.
    * **Nuclei** — per-nucleus position and size.
    * **Cytoplasm** — per-cell position, size, and per-marker intensity stats.
    * **Summary** — population-level averages and percentages.

    Returns
    -------
    output_path : Path
    """
    from pathlib import Path as _Path

    if output_dir is not None:
        output_path = _stem_output_path(input_file, '_segmentation.xlsx', output_dir)
    else:
        output_path = (_Path(input_file).parent / _image_stem(input_file)).as_posix() + '_segmentation.xlsx'

    # 2D image (table built with is_2d=True): sizes are areas.
    labels_full_df, is_2d = _volume_column_names(labels_full_df)
    U = 'μm²' if is_2d else 'μm³'           # size unit in the headers
    SIZE = 'Area' if is_2d else 'Volume'
    size = 'area' if is_2d else 'vol.'

    # ── Colour palette (matches histogram workbook) ──────────────────────
    _channel_bg = {
        'BLUE':    '#BDD7EE', 'GREEN':   '#C6EFCE', 'RED':     '#FFC7CE',
        'CYAN':    '#CCFFFF', 'MAGENTA': '#FFB3FF', 'YELLOW':  '#FFFF99',
        'WHITE':   '#D9D9D9', 'GREY':    '#D9D9D9', 'GRAY':    '#D9D9D9',
        'ORANGE':  '#FFD966',
    }
    _default_bg = '#F2F2F2'

    _COL_HDR   = '#D9E1F2'  # blue-grey header
    _COL_SUBHD = '#EBF0F9'  # light blue sub-header
    _COL_ALT   = '#F5F5F5'  # alternating row tint
    _COL_NUC   = '#C6EFCE'  # green — nuclei
    _COL_CYTO  = '#BDD7EE'  # blue — cytoplasm
    _COL_SUM   = '#FFD966'  # amber — summary

    def _marker_bg(condition, stain_df):
        try:
            color_key = str(stain_df.loc[condition, 'Color']).upper()
            return _channel_bg.get(color_key, _default_bg)
        except Exception:
            return _default_bg

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Shared formats ────────────────────────────────────────────────
        def _fmt(**kw):
            return wb.add_format(kw)

        f_hdr   = _fmt(bold=True, border=1, align='center', valign='vcenter',
                       bg_color=_COL_HDR, text_wrap=True)
        f_subhd = _fmt(bold=True, border=1, align='center', valign='vcenter',
                       bg_color=_COL_SUBHD, text_wrap=True, font_size=9)
        f_num      = _fmt(num_format='0.00')
        f_num_alt  = _fmt(num_format='0.00', bg_color=_COL_ALT)
        f_int      = _fmt()
        f_int_alt  = _fmt(bg_color=_COL_ALT)
        f_pct      = _fmt(num_format='0.00"%"')
        f_pct_alt  = _fmt(num_format='0.00"%"', bg_color=_COL_ALT)
        f_str_alt  = _fmt(bg_color=_COL_ALT)

        def _write_sheet_title(ws, row, col, text, width_cols, bg):
            fmt = _fmt(bold=True, font_size=12, border=1, align='center',
                       valign='vcenter', bg_color=bg)
            ws.merge_range(row, col, row, col + width_cols - 1, text, fmt)
            ws.set_row(row, 22)
            return row + 1

        def _write_row(ws, row, values, fmts=None, alt=False):
            for c, v in enumerate(values):
                explicit = fmts[c] if fmts else None
                if explicit:
                    fmt = explicit
                elif isinstance(v, float):
                    fmt = f_num_alt if alt else f_num
                elif isinstance(v, int):
                    fmt = f_int_alt if alt else f_int
                else:
                    fmt = f_str_alt if alt else None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ws.write_number(row, c, v, fmt)
                else:
                    ws.write(row, c, v if v is not None else '', fmt)

        # ══════════════════════════════════════════════════════════════════
        # SHEET 1 — SETUP
        # ══════════════════════════════════════════════════════════════════
        ws_setup = wb.add_worksheet('Setup')
        r = 0
        r = _write_sheet_title(ws_setup, r, 0, 'EXPERIMENT SETUP', 7, _COL_HDR)

        # Stain table header
        stain_cols = ['Channel', 'Marker', 'Laser', 'Color',
                      'Cont_min', 'Cont_max', 'Gamma']
        for c, col in enumerate(stain_cols):
            ws_setup.write(r, c, col, f_hdr)
        r += 1

        for i, (idx, srow) in enumerate(original_stain_complete_df.iterrows()):
            color_key = str(srow.get('Color', '')).upper()
            bg = _channel_bg.get(color_key, _default_bg)
            ch_fmt     = _fmt(bg_color=bg, border=1)
            ch_fmt_num = _fmt(bg_color=bg, border=1, num_format='0.00')
            ws_setup.write(r, 0, str(idx),                     ch_fmt)
            ws_setup.write(r, 1, str(srow.get('Marker', '')),  ch_fmt)
            ws_setup.write(r, 2, str(srow.get('Laser',  '')),  ch_fmt)
            ws_setup.write(r, 3, str(srow.get('Color',  '')),  ch_fmt)
            for j, col in enumerate(['Cont_min', 'Cont_max', 'Gamma']):
                raw = srow.get(col, '')
                try:
                    ws_setup.write_number(r, 4 + j, float(raw), ch_fmt_num)
                except (ValueError, TypeError):
                    ws_setup.write(r, 4 + j, str(raw), ch_fmt)
            r += 1

        ws_setup.set_column(0, 0, 14)
        ws_setup.set_column(1, 1, 18)
        ws_setup.set_column(2, 6, 12)
        ws_setup.freeze_panes(2, 0)

        # ══════════════════════════════════════════════════════════════════
        # SHEET 2 — NUCLEI
        # ══════════════════════════════════════════════════════════════════
        ws_nuc = wb.add_worksheet('Nuclei')
        nuclei_rows = labels_full_df[labels_full_df["Condition"] == "NUCLEI"]

        r = 0
        r = _write_sheet_title(ws_nuc, r, 0, 'NUCLEI — Position and Size', 5, _COL_NUC)

        nuc_cols = ['Nucleus ID', 'X position [μm]', 'Y position [μm]',
                    'Z position [μm]', f'{SIZE} [{U}]']
        for c, col in enumerate(nuc_cols):
            ws_nuc.write(r, c, col, f_hdr)
        r += 1

        if not nuclei_rows.empty:
            nuclei_row = nuclei_rows.iloc[0]
            n_nuc = int(nuclei_row["Number"])
            for k in _progress_iter(range(1, n_nuc + 1), progress, desc="Step 32A - Write Nuclei Sheet"):
                pos  = nuclei_row["Mean nuclei positions [um]"][k - 1]
                size = nuclei_row["Nuclei size [um3]"][k - 1]
                alt  = (k % 2 == 0)
                _write_row(ws_nuc, r, [k, pos[0], pos[1], pos[2], size],
                           fmts=[f_int, f_num, f_num, f_num, f_num], alt=alt)
                r += 1

        for c, w in enumerate([12, 18, 18, 18, 16]):
            ws_nuc.set_column(c, c, w)
        ws_nuc.freeze_panes(2, 0)

        # ══════════════════════════════════════════════════════════════════
        # SHEET 3 — CYTOPLASM
        # ══════════════════════════════════════════════════════════════════
        ws_cyto = wb.add_worksheet('Cytoplasm')
        cyto_rows = labels_full_df[labels_full_df["Condition"] == "CYTOPLASM"]

        r = 0

        # Collect single-condition marker rows
        single_markers = []
        for idx2, marker_name in enumerate(labels_full_df.index):
            cond = labels_full_df.iloc[idx2]["Condition"]
            if cond not in ["NUCLEI", "CYTOPLASM"] and np.size(cond) == 1:
                single_markers.append((idx2, marker_name, cond))

        # Build column groups: base + one group per marker
        # Title row: merged spanning each group
        base_cols  = ['Cell ID',
                      'X pos [μm]', 'Y pos [μm]', 'Z pos [μm]',
                      f'Nuc. {size} [{U}]', f'Cyto {size} [{U}]']
        SZ = 'Area' if is_2d else 'Vol.'
        marker_sub = [f'{SZ} total [{U}]', f'{SZ} cyto [{U}]', f'{SZ} PCM [{U}]',
                      'Avg int. total', 'STD total',
                      'Avg int. cyto', 'STD cyto',
                      'Avg int. PCM', 'STD PCM',
                      'Avg int. all cells', 'STD int. all cells']
        n_positive_sub = 9  # first 9 sub-columns above: only cells sharing this marker's threshold mask
        n_base = len(base_cols)
        n_msub = len(marker_sub)

        total_cols = n_base + len(single_markers) * n_msub

        r = _write_sheet_title(ws_cyto, r, 0,
                               'CYTOPLASM — Per-cell Position, Size and Marker Intensities',
                               total_cols, _COL_CYTO)

        # Title row: base header + per-marker colour blocks
        base_fmt = _fmt(bold=True, border=1, align='center', valign='vcenter',
                        bg_color=_COL_CYTO, text_wrap=True)
        for c, col in enumerate(base_cols):
            ws_cyto.merge_range(r, c, r + 1, c, col, base_fmt)

        for m_i, (m_idx, m_name, m_cond) in enumerate(single_markers):
            bg = _marker_bg(m_cond, original_stain_complete_df)
            mfmt = _fmt(bold=True, border=1, align='center', valign='vcenter',
                        bg_color=bg, font_size=11)
            col_start = n_base + m_i * n_msub
            ws_cyto.merge_range(r, col_start, r, col_start + n_msub - 1, m_name, mfmt)
        r += 1

        # Sub-header row for marker columns
        for c, col in enumerate(base_cols):
            pass  # already merged above
        for m_i, _ in enumerate(single_markers):
            col_start = n_base + m_i * n_msub
            for j, sub in enumerate(marker_sub):
                ws_cyto.write(r, col_start + j, sub, f_subhd)
        r += 1

        if not cyto_rows.empty:
            cytoplasm_row = cyto_rows.iloc[0]
            n_cyto = int(cytoplasm_row["Number"])
            for k in _progress_iter(range(1, n_cyto + 1), progress, desc="Step 32B - Write Cytoplasm Sheet"):
                cyto_pos  = cytoplasm_row["Mean cytoplasm positions [um]"][k - 1]
                cyto_size = cytoplasm_row["Cytoplasm size [um3]"][k - 1]
                nuc_size  = (nuclei_rows.iloc[0]["Nuclei size [um3]"][k - 1]
                             if not nuclei_rows.empty else 0.0)
                row_vals  = [k,
                             cyto_pos[0], cyto_pos[1], cyto_pos[2],
                             nuc_size, cyto_size]
                alt = (k % 2 == 0)

                for m_i, (m_idx, m_name, m_cond) in enumerate(single_markers):
                    m_row = labels_full_df.iloc[m_idx]
                    shared = m_row["Shared labels"]
                    if k in shared:
                        si = list(shared).index(k)
                        row_vals.extend([
                            m_row["Marker size [um3]"][si],
                            m_row["Marker size cytoplasm [um3]"][si],
                            m_row["Marker size PCM [um3]"][si],
                            m_row["Avg. marker intensity"][si],
                            m_row["STD marker intensity"][si],
                            m_row["Avg. marker intensity cytoplasm"][si],
                            m_row["STD marker intensity cytoplasm"][si],
                            m_row["Avg. marker intensity PCM"][si],
                            m_row["STD marker intensity PCM"][si],
                        ])
                    else:
                        row_vals.extend([''] * n_positive_sub)

                    # Whole-cell intensity, positive or not (dense, indexed
                    # directly by cell ID -- not gated by "shared").
                    all_mean = m_row.get("Avg. marker intensity (all cells)", ())
                    all_std = m_row.get("STD marker intensity (all cells)", ())
                    mean_val = all_mean[k - 1] if len(all_mean) >= k else np.nan
                    std_val = all_std[k - 1] if len(all_std) >= k else np.nan
                    row_vals.append('' if pd.isna(mean_val) else mean_val)
                    row_vals.append('' if pd.isna(std_val) else std_val)

                _write_row(ws_cyto, r, row_vals, alt=alt)
                r += 1

        ws_cyto.set_column(0, 0, 10)
        ws_cyto.set_column(1, 5, 16)
        for m_i in range(len(single_markers)):
            col_start = n_base + m_i * n_msub
            ws_cyto.set_column(col_start, col_start + n_msub - 1, 14)
        ws_cyto.freeze_panes(3, 1)

        # ══════════════════════════════════════════════════════════════════
        # SHEET 4 — SUMMARY
        # ══════════════════════════════════════════════════════════════════
        ws_sum = wb.add_worksheet('Summary')
        r = 0
        r = _write_sheet_title(ws_sum, r, 0, 'POPULATION SUMMARY', 9, _COL_SUM)

        sum_cols = ['Condition', 'Marker', 'Laser', 'Color',
                    'Count', '% of total',
                    f'Mean nuc. {size} [{U}]', f'SD nuc. {size} [{U}]',
                    f'Mean cyto. {size} [{U}]', f'SD cyto. {size} [{U}]']

        # Add marker-specific mean columns
        for _, m_name, _ in single_markers:
            sum_cols += [f'{m_name} mean int. (positive cells)', f'{m_name} SD int. (positive cells)',
                         f'{m_name} mean {size} [{U}]',
                         f'{m_name} mean int. (all cells)', f'{m_name} SD int. (all cells)']

        for c, col in enumerate(sum_cols):
            ws_sum.write(r, c, col, f_hdr)
        r += 1

        nuclei_total = (float(nuclei_rows.iloc[0]["Number"])
                        if not nuclei_rows.empty else 1.0)

        # ── Single-condition rows ─────────────────────────────────────────
        for t in range(len(labels_full_df)):
            row_data = labels_full_df.iloc[t]
            cond = row_data["Condition"]
            if np.size(cond) != 1 or cond == "CYTOPLASM":
                continue

            color_key = str(row_data.get("Color", "")).upper()
            bg = _channel_bg.get(color_key, _default_bg)
            row_fmt = _fmt(bg_color=bg)

            nuc_sizes  = row_data["Nuclei size [um3]"]
            cyto_sizes = row_data["Cytoplasm size [um3]"]
            count      = int(row_data["Number"])
            pct        = 100.0 * count / nuclei_total if cond != "NUCLEI" else 100.0

            vals = [
                str(cond),
                str(row_data.get("Marker", "")),
                str(row_data.get("Laser", "")),
                str(row_data.get("Color", "")),
                count,
                round(pct, 1),
                round(float(np.mean(nuc_sizes)),  2) if len(nuc_sizes)  > 0 else '',
                round(float(np.std(nuc_sizes)),   2) if len(nuc_sizes)  > 0 else '',
                round(float(np.mean(cyto_sizes)), 2) if len(cyto_sizes) > 0 else '',
                round(float(np.std(cyto_sizes)),  2) if len(cyto_sizes) > 0 else '',
            ]

            for m_idx2, m_name, m_cond in single_markers:
                m_row = labels_full_df.iloc[m_idx2]
                if cond == m_cond:
                    ints = m_row.get("Avg. marker intensity", [])
                    vols = m_row.get("Marker size [um3]", [])
                    ints_all = np.array(m_row.get("Avg. marker intensity (all cells)", []), dtype=float)
                    ints_all = ints_all[~np.isnan(ints_all)]
                    vals += [
                        round(float(np.mean(ints)), 2) if len(ints) > 0 else '',
                        round(float(np.std(ints)),  2) if len(ints) > 0 else '',
                        round(float(np.mean(vols)), 2) if len(vols) > 0 else '',
                        round(float(np.mean(ints_all)), 2) if ints_all.size > 0 else '',
                        round(float(np.std(ints_all)),  2) if ints_all.size > 0 else '',
                    ]
                else:
                    vals += ['', '', '', '', '']

            for c, v in enumerate(vals):
                num_fmt_row = _fmt(bg_color=bg, num_format='0.00') if c > 4 else _fmt(bg_color=bg)
                if isinstance(v, float):
                    ws_sum.write_number(r, c, v, num_fmt_row)
                elif isinstance(v, int):
                    ws_sum.write_number(r, c, v, row_fmt)
                else:
                    ws_sum.write(r, c, v if v != '' else '', row_fmt)
            r += 1

        # ── Multilabel combination rows ───────────────────────────────────
        _COL_MULTI = '#FFFFFF'
        has_multi = any(
            np.size(labels_full_df.iloc[t]["Condition"]) != 1
            for t in range(len(labels_full_df))
        )
        if has_multi:
            sec_fmt = _fmt(bold=True, bg_color=_COL_SUM, border=1)
            ws_sum.write(r, 0, 'MULTILABEL COMBINATIONS', sec_fmt)
            for c in range(1, len(sum_cols)):
                ws_sum.write(r, c, '', sec_fmt)
            ws_sum.set_row(r, 16)
            r += 1

        for t in range(len(labels_full_df)):
            row_data = labels_full_df.iloc[t]
            cond = row_data["Condition"]
            if np.size(cond) == 1:
                continue

            row_fmt     = _fmt(bg_color=_COL_MULTI)
            row_fmt_num = _fmt(bg_color=_COL_MULTI, num_format='0.00')

            cond_str   = ' + '.join(str(c) for c in cond) if hasattr(cond, '__iter__') else str(cond)
            nuc_sizes  = row_data["Nuclei size [um3]"]
            cyto_sizes = row_data["Cytoplasm size [um3]"]
            count      = int(row_data["Number"])
            pct        = 100.0 * count / nuclei_total

            vals = [
                cond_str, '', '', '',
                count,
                round(pct, 1),
                round(float(np.mean(nuc_sizes)),  2) if len(nuc_sizes)  > 0 else '',
                round(float(np.std(nuc_sizes)),   2) if len(nuc_sizes)  > 0 else '',
                round(float(np.mean(cyto_sizes)), 2) if len(cyto_sizes) > 0 else '',
                round(float(np.std(cyto_sizes)),  2) if len(cyto_sizes) > 0 else '',
            ] + [''] * (len(single_markers) * 5)

            for c, v in enumerate(vals):
                fmt = row_fmt_num if c > 4 and isinstance(v, float) else row_fmt
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ws_sum.write_number(r, c, v, fmt)
                else:
                    ws_sum.write(r, c, v if v != '' else '', row_fmt)
            r += 1

        ws_sum.set_column(0, 0, 14)
        ws_sum.set_column(1, 3, 14)
        ws_sum.set_column(4, 4, 8)
        ws_sum.set_column(5, 5, 10)
        ws_sum.set_column(6, len(sum_cols) - 1, 18)
        ws_sum.freeze_panes(2, 0)

    print(f"Segmentation report saved to: {output_path}")
    return output_path


class ImageProcessing:
    """Minimal wrapper used by the reporting helpers."""

    def __init__(self, filename):
        self.filename = filename
        self.img = PILImage.open(filename)

    def as_np(self):
        return np.array(self.img)


def save_single_channel_png(img2d, fname):
    """Save a 2D image with percentile-based scaling for visualization."""
    if img2d is None or img2d.size == 0:
        return None

    arr = np.asarray(img2d, dtype=float)
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    arr_vis = (arr - vmin) / (vmax - vmin)
    arr_vis = np.clip(arr_vis * 255, 0, 255).astype(np.uint8)
    PILImage.fromarray(arr_vis).save(fname)
    return fname


def get_stain_name(stain_df, key):
    """Return a readable stain name from a dict-like object or DataFrame."""
    try:
        return str(stain_df[key])
    except Exception:
        try:
            return str(stain_df.loc[key, "stain_name"])
        except Exception:
            return key


def create_row_pdf(output_pdf="nuclei_row_pages.pdf", pad=20, thumb_size=None, png_dir=None):
    """Create the nuclei report PDF using notebook context previously registered.

    The per-nucleus PNGs embedded in the PDF are written to crop_png/,
    merged_png/ and density_png/ inside *png_dir* (default: current working
    directory).
    """
    from matplotlib.lines import Line2D
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from scipy.stats import gaussian_kde

    hist_data = _context("hist_data")
    stain_complete_df = _context("stain_complete_df")
    stain_df = _context("stain_df")
    seg_stack = _context("seg_stack")
    filtered_img = _context("filtered_img")
    x_grid = _context("x_grid")
    styles = globals().get("styles") or getSampleStyleSheet()

    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []
    nuclei = sorted(hist_data.keys())
    if not nuclei:
        raise ValueError("hist_data is empty.")

    all_conditions = sorted({condition for nucleus_data in hist_data.values() for condition in nucleus_data.keys()})
    marker_conditions = [condition for condition in all_conditions if condition.lower() != "nuclei"]

    marker_labels = {
        condition: (
            str(stain_complete_df.loc[condition, "Marker"])
            if condition in stain_complete_df.index and "Marker" in stain_complete_df.columns
            else condition
        )
        for condition in all_conditions
    }

    # 'Filtered image' already has each channel's stored Cont_min/Cont_max/Gamma
    # baked in (applied back in run_contrast_gamma), and that is what Napari's
    # viewer displays with no further contrast/gamma adjustment. Re-derive the
    # display range from the actual data here instead of re-applying the stored
    # settings, otherwise contrast/gamma gets stacked twice and the PNGs render
    # much darker than what is shown in Napari for the same image.
    display_ranges = {}
    for condition in marker_conditions:
        channel_indices = np.where(stain_complete_df.index == condition)[0]
        if len(channel_indices):
            channel_data = filtered_img[:, :, :, channel_indices[0]]
            display_ranges[condition] = (float(channel_data.min()), float(channel_data.max()))

    # Compute column geometry from the page so images never overflow their cells.
    _cell_pad = 4
    _left_margin = _right_margin = inch
    _page_w, _ = A4
    _usable_w = _page_w - _left_margin - _right_margin
    _n_cols = len(marker_conditions) + 1
    _col_w = _usable_w / _n_cols
    _img_w = _col_w - 2 * _cell_pad
    condition_colors = {
        condition: (
            stain_complete_df.loc[condition, "Color"]
            if condition in stain_complete_df.index and "Color" in stain_complete_df.columns
            else "gray"
        )
        for condition in marker_conditions
    }
    condition_colors = {key: ("gray" if value == "WHITE" else value) for key, value in condition_colors.items()}
    nucleus_color = (
        stain_df.loc["NUCLEI", "Color"]
        if "NUCLEI" in stain_df.index and "Color" in stain_df.columns
        else "blue"
    )

    png_dir = png_dir or ""
    crop_dir = os.path.join(png_dir, "crop_png")
    merged_dir = os.path.join(png_dir, "merged_png")
    density_dir = os.path.join(png_dir, "density_png")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)

    for nucleus_id in nuclei:
        full_stack = {}
        for condition in marker_conditions:
            channel_indices = np.where(stain_complete_df.index == condition)[0]
            if len(channel_indices):
                full_stack[condition] = filtered_img[:, :, :, channel_indices[0]]

        nucleus_mask = seg_stack["Nuclei"] == nucleus_id
        if np.sum(nucleus_mask) == 0:
            story.append(Paragraph(f"<b>Nucleus {nucleus_id}</b>: no pixels found", styles["Heading2"]))
            story.append(Spacer(1, 0.3 * inch))
            continue

        crop_dict, best_z, _, crop_shape = crop_nucleus_with_padding(nucleus_mask, full_stack, pad=pad)
        if crop_dict is None or crop_shape is None:
            story.append(Paragraph(f"<b>Nucleus {nucleus_id}</b>: no pixels found", styles["Heading2"]))
            story.append(Spacer(1, 0.3 * inch))
            continue

        min_h, min_w = crop_shape
        channel_pngs = []
        for condition in marker_conditions:
            img = crop_dict.get(condition)
            arr = np.zeros((min_h, min_w)) if img is None or img.size == 0 else img[:min_h, :min_w]
            fname = os.path.join(crop_dir, f"n{nucleus_id}_{condition}.png")
            color = stain_complete_df.loc[condition, "Color"] if (
                condition in stain_complete_df.index and "Color" in stain_complete_df.columns
            ) else "gray"
            if color == "WHITE":
                color = "GRAY"
            clim = display_ranges.get(condition)
            if clim is not None:
                try:
                    save_raw_png(arr, fname, contrast_limits=clim, gamma=1.0, color=color)
                except Exception:
                    save_raw_png(arr, fname, color=color)
            else:
                save_raw_png(arr, fname, color=color)
            channel_pngs.append(fname)

        merged_png = save_merged_figure(
            nucleus_mask,
            full_stack,
            condition_colors,
            nucleus_id,
            seg_stack=seg_stack,
            display_ranges=display_ranges,
            nucleus_color=nucleus_color,
            cytoplasm_color="green",
            pcm_color="magenta",
            pad=pad,
            out_dir=merged_dir,
        )
        if merged_png is None:
            merged_png = os.path.join(merged_dir, f"n{nucleus_id}_merged_placeholder.png")
            save_raw_png(np.zeros((min_h, min_w)), merged_png)

        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

        for condition in all_conditions:
            vals = np.asarray(hist_data.get(nucleus_id, {}).get(condition, []))
            color = stain_complete_df["Color"][condition] if stain_complete_df["Color"][condition] != "WHITE" else "GRAY"
            if vals.size == 0:
                continue
            try:
                kde = gaussian_kde(vals)
                y_grid = kde(x_grid)
            except Exception:
                mean_val = vals.mean()
                y_grid = np.exp(-0.5 * ((x_grid - mean_val) / 1.0) ** 2)

            y_norm = y_grid / y_grid.max() if y_grid.max() > 0 else y_grid
            ax.plot(x_grid, y_norm, linewidth=2, color=color)
            mean_val = float(vals.mean())
            std_val = float(vals.std())
            ax.axvline(mean_val, linestyle="--", linewidth=1.5, color=color)
            y_at_mean = np.interp(mean_val, x_grid, y_norm)
            ax.hlines(y_at_mean, max(0.0, mean_val - std_val), min(255.0, mean_val + std_val), linewidth=2, color=color)

        legend_handles = [
            Line2D([0], [0], color=stain_complete_df["Color"][condition] if stain_complete_df["Color"][condition] != "WHITE" else "GRAY", lw=2)
            for condition in all_conditions
        ]
        ax.legend(legend_handles, [marker_labels[condition] for condition in all_conditions], loc="upper right", framealpha=0.9)
        plt.tight_layout()
        density_png = os.path.join(density_dir, f"n{nucleus_id}_density.png")
        fig.savefig(density_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Compute thumbnail height that preserves the actual crop aspect ratio.
        _aspect = (min_w / min_h) if min_h > 0 else 1.0
        _img_h = _img_w / _aspect

        label_style = ParagraphStyle(
            "ch_label",
            parent=styles["Normal"],
            fontSize=7,
            alignment=1,
            textColor=rl_colors.black,
            spaceAfter=0,
            spaceBefore=0,
        )
        label_row = [Paragraph(marker_labels[cond], label_style) for cond in marker_conditions] + [Paragraph("Merged", label_style)]
        label_height = 0.18 * inch

        data = [
            [Image(density_png, width=_usable_w, height=2.0 * inch)],
            label_row,
            [Image(path, width=_img_w, height=_img_h) for path in channel_pngs] + [Image(merged_png, width=_img_w, height=_img_h)],
        ]
        table = Table(
            data,
            colWidths=[_col_w] * _n_cols,
            rowHeights=[2.0 * inch, label_height, _img_h],
        )
        table.setStyle(TableStyle([
            ("SPAN", (0, 0), (_n_cols - 1, 0)),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("VALIGN", (0, 1), (_n_cols - 1, 1), "BOTTOM"),
            # padding around image cells to create visible gaps
            ("LEFTPADDING",  (0, 1), (-1, 2), _cell_pad),
            ("RIGHTPADDING", (0, 1), (-1, 2), _cell_pad),
            ("TOPPADDING",    (0, 2), (-1, 2), 2),
            ("BOTTOMPADDING", (0, 2), (-1, 2), 2),
        ]))

        story.append(Paragraph(f"<b>Nucleus {nucleus_id} (Z {best_z})</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.05 * inch))
        story.append(table)
        story.append(PageBreak())

    doc.build(story)
    print(f"PDF saved to: {output_pdf}")


def watershed_nuclei(im_in, channel_index, nuclei_diameter, r_zZ, r_zY, r_zX):
    """Run a watershed-oriented nuclei marker step with anisotropic peaks."""
    distance = ndi.distance_transform_edt(im_in[:, :, :, channel_index], sampling=[r_zZ, r_zY, r_zX])
    radius_x = int((nuclei_diameter / 2.0) / r_zX)
    radius_y = int((nuclei_diameter / 2.0) / r_zY)
    radius_z = int((nuclei_diameter / 2.0) / r_zZ)
    footprint = make_anisotropic_footprint(int(0.6 * radius_z), int(0.6 * radius_y), int(0.6 * radius_x))
    min_dist_um = nuclei_diameter * 0.75
    min_dist_vox = int(np.min([min_dist_um / r_zZ, min_dist_um / r_zY, min_dist_um / r_zX]))
    coords = peak_local_max(distance, footprint=footprint, min_distance=min_dist_vox, labels=im_in[:, :, :, channel_index].astype(np.int32))

    mask = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    return markers


def grow_markers_within_islands_limited(markers: np.ndarray, islands: np.ndarray, max_distance: float) -> np.ndarray:
    """Expand labels by a limited distance while staying inside a binary mask."""
    from skimage.segmentation import expand_labels

    if markers.shape != islands.shape:
        raise ValueError("markers and islands must have the same shape")

    expanded = expand_labels(markers, distance=max_distance)
    expanded[~islands.astype(bool)] = 0
    return expanded

def detect_peaks_xy_with_best_z(
    local_distance,
    peak_min_distance_xy,
    peak_threshold_fraction,
):
    """
    Detect multiple peaks in a distance-transform volume using 2D XY projection.
    
    Useful for identifying multiple seed points in dumbbell-shaped or multi-lobed
    nuclei where standard erosion fails to create separate seeds. Prevents 
    over-fragmentation across Z slices by projecting to 2D for peak detection,
    then mapping each XY peak to its best Z coordinate.
    
    Parameters
    ----------
    local_distance : ndarray, shape (Z, Y, X)
        Distance transform (EDT) computed for a single nucleus or connected component.
        Typically obtained from scipy.ndimage.distance_transform_edt.
    
    peak_min_distance_xy : int
        Minimum pixel distance between detected peaks in the XY plane.
        Typical values: 45% of nucleus diameter in voxels.
    
    peak_threshold_fraction : float
        Fraction of the maximum distance value to use as threshold.
        Peaks below this threshold are excluded.
        Typical values: 0.45 (45% of max EDT value).
    
    Returns
    -------
    peak_coords_3d : list of tuples
        List of (z, y, x) coordinates for detected peaks.
        Each tuple represents a 3D location where a peak was found.
        Returns empty list if fewer than 2 peaks are detected or if
        all peaks are below threshold.
    
    Notes
    -----
    - Projects distance transform to 2D via max along Z: xy_distance = np.max(local_distance, axis=0)
    - Detects peaks only in the 2D projection
    - For each XY peak, finds the Z coordinate with maximum distance (z_i = argmax(z_line))
    - Only includes peaks where max distance along Z exceeds the threshold
    - Prevents Z-layer fragmentation by assigning one seed per unique XY location
    
    Example
    -------
    >>> distance = scipy.ndimage.distance_transform_edt(binary_mask, sampling=[0.2, 0.1, 0.1])
    >>> local_dist = np.where(component_mask, distance, 0.0)
    >>> peaks = detect_peaks_xy_with_best_z(local_dist, peak_min_distance_xy=10, peak_threshold_fraction=0.45)
    >>> print(f"Found {len(peaks)} peaks")
    """
    if not np.any(local_distance):
        return []
    
    # Step 1: Project distance to 2D via max along Z
    xy_distance = np.max(local_distance, axis=0)
    
    # Step 2: Compute threshold
    peak_threshold = max(
        0.0,
        peak_threshold_fraction * float(local_distance[local_distance > 0].max()),
    )
    
    # Step 3: Find peaks in XY plane
    peak_coords_xy = peak_local_max(
        xy_distance,
        min_distance=peak_min_distance_xy,
        threshold_abs=peak_threshold,
        exclude_border=False,
    )
    
    # Step 4: For each XY peak, find best Z and validate
    peak_coords_3d = []
    for yx in peak_coords_xy:
        y_i, x_i = int(yx[0]), int(yx[1])
        z_line = local_distance[:, y_i, x_i]
        z_max = float(np.max(z_line))
        
        # Only include if peak value in Z direction exceeds threshold
        if z_max < peak_threshold:
            continue
        
        z_i = int(np.argmax(z_line))
        peak_coords_3d.append((z_i, y_i, x_i))
    
    return peak_coords_3d


def detect_peaks_xy_intensity_aware(
    local_distance,
    local_intensity,
    peak_min_distance_xy,
    peak_threshold_fraction,
    distance_weight=1.0,
    intensity_weight=1.0,
):
    """
    Detect multiple nucleus seed points within one connected component using
    both the EDT distance transform and the raw (smoothed) intensity signal.

    Two touching, individually round nuclei often merge into a single
    threshold "island" whose distance transform has no clean second maximum
    (the neck isn't deep enough). Combining distance with intensity — most
    nuclear stains dip in brightness right where two nuclei abut — restores
    a local minimum/saddle at that boundary, so each nucleus's own bright,
    bulging center still shows up as a separate maximum in the combined map.

    Parameters
    ----------
    local_distance : ndarray, shape (Z, Y, X)
        Distance transform (EDT), zeroed outside the connected component.
    local_intensity : ndarray, shape (Z, Y, X)
        Smoothed intensity image, zeroed outside the connected component.
    peak_min_distance_xy : int
        Minimum pixel distance between detected peaks in the XY plane.
    peak_threshold_fraction : float
        Fraction of the combined map's maximum used as the peak threshold.
    distance_weight, intensity_weight : float
        Relative weight of each (locally 0-1 normalized) term in the combined
        seed-strength map that peaks are detected in.

    Returns
    -------
    peak_coords_3d : list of tuples
        (z, y, x) coordinates for detected peaks. Empty if fewer than 2
        distinct peaks are found.
    """
    mask = local_distance > 0
    if not np.any(mask):
        return []

    def _norm(arr):
        values = arr[mask]
        if values.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    combined = (
        distance_weight * _norm(local_distance)
        + intensity_weight * _norm(np.where(mask, local_intensity, 0))
    )
    combined[~mask] = 0.0

    xy_combined = np.max(combined, axis=0)
    peak_threshold = max(0.0, peak_threshold_fraction * float(xy_combined.max()))

    peak_coords_xy = peak_local_max(
        xy_combined,
        min_distance=peak_min_distance_xy,
        threshold_abs=peak_threshold,
        exclude_border=False,
    )

    peak_coords_3d = []
    for yx in peak_coords_xy:
        y_i, x_i = int(yx[0]), int(yx[1])
        z_line = combined[:, y_i, x_i]
        z_max = float(np.max(z_line))
        if z_max < peak_threshold:
            continue
        z_i = int(np.argmax(z_line))
        peak_coords_3d.append((z_i, y_i, x_i))

    return peak_coords_3d


def _split_labels_by_z_consistency(label_img, min_fragment_vox=8,
                                   aggressive=False):
    """Split labels whose voxels form multiple disconnected 3D regions.

    Two modes are available, selected by *aggressive*:

    **topology** (``aggressive=False``, default) — only true spatial
    disconnections in the z-stack cause a split.  Slice-by-slice 2D
    connected-component tracking links overlapping components across
    adjacent z-slices; each independently connected group becomes its
    own label.  Fast and conservative.

    **watershed** (``aggressive=True``) — if a label has multiple
    significant 2D components at ANY z-slice, a local 3D watershed
    splits it along the natural constriction, even when the components
    reconnect at other z-levels.  Slower and more aggressive.

    Parameters
    ----------
    label_img : ndarray (Z, Y, X), int
        Integer label volume (0 = background).
    min_fragment_vox : int
        Groups / fragments smaller than this are absorbed back into the
        largest piece rather than becoming a new label.
    aggressive : bool
        If False use topology-only splitting; if True use watershed
        reinforced splitting.

    Returns
    -------
    out : ndarray (Z, Y, X), int
        Updated label volume with split labels.
    total_splits : int
        Number of new labels created.
    """
    if aggressive:
        return _split_z_watershed(label_img, min_fragment_vox)
    return _split_z_topology(label_img, min_fragment_vox)


def _split_z_topology(label_img, min_fragment_vox, min_z_overlap=2):
    """Topology-only z-split: union-find across adjacent z-slices.

    Uses 4-connectivity within each XY slice so that even a 1-pixel
    diagonal gap counts as a disconnection.  Two components in adjacent
    z-slices are linked only when they share at least *min_z_overlap*
    pixels, so a thin 1-pixel bridge across z is not enough to keep
    two nuclei merged.

    Parameters
    ----------
    label_img : ndarray (Z, Y, X), int
    min_fragment_vox : int
    min_z_overlap : int
        Minimum number of overlapping pixels between components in
        adjacent z-slices to consider them connected (default 2).
    """
    out = np.asarray(label_img).copy()
    current_labels = np.unique(out)
    current_labels = current_labels[current_labels > 0]
    next_label = int(out.max()) + 1
    total_splits = 0
    # Each label in its bounding box: same result as on the whole image.
    slices = ndi.find_objects(out) if current_labels.size else []

    for label_id in current_labels:
        sl = slices[int(label_id) - 1]
        if sl is None:
            continue
        out_c = out[sl]  # view: writes go to out
        label_mask = out_c == label_id
        z_slices = np.where(np.any(label_mask, axis=(1, 2)))[0]
        if len(z_slices) <= 1:
            continue

        slice_components = {}
        nodes = []
        node_index = {}
        for z in z_slices:
            labeled_2d, num = ndi.label(label_mask[z])
            slice_components[z] = labeled_2d
            for c in range(1, num + 1):
                idx = len(nodes)
                nodes.append((z, c))
                node_index[(z, c)] = idx

        if len(nodes) <= 1:
            continue

        parent = list(range(len(nodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for i in range(len(z_slices) - 1):
            z_curr = z_slices[i]
            z_next = z_slices[i + 1]
            if z_next != z_curr + 1:
                continue
            lab_curr = slice_components[z_curr]
            lab_next = slice_components[z_next]
            num_curr = int(lab_curr.max())
            num_next = int(lab_next.max())
            for c1 in range(1, num_curr + 1):
                mask_c1 = lab_curr == c1
                for c2 in range(1, num_next + 1):
                    overlap = int(np.count_nonzero(mask_c1 & (lab_next == c2)))
                    if overlap >= min_z_overlap:
                        union(node_index[(z_curr, c1)],
                              node_index[(z_next, c2)])

        groups = {}
        for idx_n, (z, c) in enumerate(nodes):
            groups.setdefault(find(idx_n), []).append((z, c))

        if len(groups) <= 1:
            continue

        group_sizes = {}
        for root, members in groups.items():
            size = 0
            for z, c in members:
                size += int(np.count_nonzero(slice_components[z] == c))
            group_sizes[root] = size

        largest_root = max(group_sizes, key=group_sizes.get)

        for root, members in groups.items():
            if root == largest_root:
                continue
            if group_sizes[root] < min_fragment_vox:
                continue
            new_label = next_label
            next_label += 1
            total_splits += 1
            for z, c in members:
                out_c[z][slice_components[z] == c] = new_label

    return out, total_splits


def _split_z_watershed(label_img, min_fragment_vox):
    """Watershed-reinforced z-split: any z with multiple significant 2D
    components seeds a local watershed that carves the full 3D label."""
    from skimage.segmentation import watershed

    out = np.asarray(label_img).copy()
    next_label = int(out.max()) + 1
    total_splits = 0
    min_component_area_2d = max(4, min_fragment_vox // 2)

    for _pass in range(3):
        current_labels = np.unique(out)
        current_labels = current_labels[current_labels > 0]
        pass_splits = 0
        # Each label in its bounding box (+1 voxel for the distance
        # transform): same result as on the whole image. A label only loses
        # voxels during the pass, so its box stays valid.
        slices = ndi.find_objects(out) if current_labels.size else []

        for label_id in current_labels:
            sl = slices[int(label_id) - 1]
            if sl is None:
                continue
            out_c = out[_padded_slice(sl, out.shape, 1)]  # view: writes go to out
            label_mask = out_c == label_id
            z_slices = np.where(np.any(label_mask, axis=(1, 2)))[0]
            if len(z_slices) <= 1:
                continue

            best_z = None
            best_num_significant = 1
            for z in z_slices:
                labeled_2d_z, num_z = ndi.label(label_mask[z])
                if num_z <= best_num_significant:
                    continue
                sizes_z = np.bincount(labeled_2d_z.ravel())[1:]
                n_significant = int(np.sum(sizes_z >= min_component_area_2d))
                if n_significant > best_num_significant:
                    best_num_significant = n_significant
                    best_z = z

            if best_z is None:
                continue

            labeled_2d, num_components = ndi.label(label_mask[best_z])
            comp_sizes = np.bincount(labeled_2d.ravel())
            valid_components = [
                c for c in range(1, num_components + 1)
                if comp_sizes[c] >= min_component_area_2d
            ]

            if len(valid_components) <= 1:
                continue

            markers = np.zeros_like(label_mask, dtype=np.int32)
            for i, c in enumerate(valid_components, start=1):
                markers[best_z][labeled_2d == c] = i

            distance = ndi.distance_transform_edt(label_mask)
            sub_labels = watershed(-distance, markers, mask=label_mask)

            ws_sizes = np.bincount(sub_labels.ravel(),
                                   minlength=len(valid_components) + 1)
            largest_sid = int(np.argmax(ws_sizes[1:]) + 1)

            for sid in range(1, len(valid_components) + 1):
                if sid == largest_sid:
                    continue
                if ws_sizes[sid] < min_fragment_vox:
                    continue
                out_c[sub_labels == sid] = next_label
                next_label += 1
                pass_splits += 1

        total_splits += pass_splits
        if pass_splits == 0:
            break

    return out, total_splits


def _merge_labels_by_z_consistency(label_img):
    """Merge adjacent labels that together form a z-consistent object.

    If two neighbouring labels, when combined, produce exactly one
    connected 2D component (8-connectivity) at every z-slice, they were
    over-split and should be a single nucleus.

    Uses union-find to batch all valid merges in one pass, then repeats
    (at most twice) in case merges reveal new adjacencies.

    Parameters
    ----------
    label_img : ndarray (Z, Y, X), int
        Integer label volume (0 = background).

    Returns
    -------
    out : ndarray (Z, Y, X), int
        Updated label volume with merged labels.
    total_merges : int
        Number of merges performed.
    """
    out = np.asarray(label_img).copy()
    struct_2d = np.ones((3, 3), dtype=int)

    def _adjacent_pairs_fast(labels):
        pairs = set()
        max_l = int(labels.max()) + 1
        for axis in range(3):
            slc_a = [slice(None)] * 3
            slc_b = [slice(None)] * 3
            slc_a[axis] = slice(1, None)
            slc_b[axis] = slice(None, -1)
            l1 = labels[tuple(slc_a)]
            l2 = labels[tuple(slc_b)]
            mask = (l1 > 0) & (l2 > 0) & (l1 != l2)
            if not np.any(mask):
                continue
            a_vals = l1[mask].ravel().astype(np.int64)
            b_vals = l2[mask].ravel().astype(np.int64)
            lo = np.minimum(a_vals, b_vals)
            hi = np.maximum(a_vals, b_vals)
            for code in np.unique(lo * max_l + hi):
                pairs.add((int(code // max_l), int(code % max_l)))
        return pairs

    def _is_z_consistent_pair(labels, a, b, slices):
        # Only the bounding box of the two labels (same result as the whole image).
        sa, sb = slices[a - 1], slices[b - 1]
        sl = tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(sa, sb))
        crop = labels[sl]
        mask_a = crop == a
        mask_b = crop == b
        zs_a = set(np.where(np.any(mask_a, axis=(1, 2)))[0])
        zs_b = set(np.where(np.any(mask_b, axis=(1, 2)))[0])
        for z in zs_a | zs_b:
            combined_z = mask_a[z] | mask_b[z]
            _, num = ndi.label(combined_z, structure=struct_2d)
            if num > 1:
                return False
        return True

    total_merges = 0

    for _pass in range(3):
        pairs = _adjacent_pairs_fast(out)
        if not pairs:
            break

        uf_parent = {}
        def _find(x):
            while uf_parent.setdefault(x, x) != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x
        def _union(x, y):
            rx, ry = _find(x), _find(y)
            if rx != ry:
                uf_parent[ry] = rx

        pass_merges = 0
        slices = ndi.find_objects(out)  # out is not changed until the end of the pass
        for a, b in pairs:
            if _find(a) == _find(b):
                continue
            if _is_z_consistent_pair(out, a, b, slices):
                _union(a, b)
                pass_merges += 1

        if pass_merges == 0:
            break

        remap = {}
        for lbl in np.unique(out):
            if lbl <= 0:
                continue
            root = _find(int(lbl))
            if root != int(lbl):
                remap[int(lbl)] = root

        if remap:
            lut = np.arange(int(out.max()) + 1, dtype=out.dtype)
            for old, new in remap.items():
                lut[old] = new
            out = lut[out]

        total_merges += pass_merges

    return out, total_merges


def _padded_slice(sl, shape, pad):
    """Bounding box *sl* (from `ndi.find_objects`) grown by *pad* voxels on
    every side, clipped to the image *shape*.

    Speed: the per-label steps work on this crop
    instead of the whole image, so their cost follows the label's size, not
    the image's (with many cells, a whole-image pass per label made them
    grow as cells x voxels). The padding gives every filter the same
    neighbourhood it has in the whole image, so the results are unchanged:
    1 voxel for the distance transform, erosion, contours and connected
    components; the filter radius for a Gaussian; ``min_distance`` for a
    peak search.
    """
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
                 for s, n in zip(sl, shape))


def _gaussian_radius(sigma):
    """Radius (voxels) of `scipy.ndimage.gaussian_filter` /
    `gaussian_gradient_magnitude` with the default ``truncate=4.0``."""
    return int(4.0 * float(sigma) + 0.5)


def _union_label_slices(label_images, n_labels):
    """For labels 1..*n_labels*: the bounding box covering the label in all
    *label_images* (None where it is in none of them)."""
    per_image = [ndi.find_objects(np.asarray(img), max_label=n_labels)
                 for img in label_images if img is not None]
    out = []
    for i in range(n_labels):
        boxes = [s[i] for s in per_image if i < len(s) and s[i] is not None]
        if not boxes:
            out.append(None)
            continue
        out.append(tuple(slice(min(b[d].start for b in boxes), max(b[d].stop for b in boxes))
                         for d in range(len(boxes[0]))))
    return out


def _label_roundness_xy(mask):
    """Circularity (4*pi*area/perimeter^2) of a 3D boolean mask's largest XY slice."""
    z_counts = np.count_nonzero(mask, axis=(1, 2))
    if not np.any(z_counts):
        return 0.0
    z_best = int(np.argmax(z_counts))
    slice_mask = mask[z_best]
    area = int(np.count_nonzero(slice_mask))
    if area <= 2:
        return 0.0
    contours, _ = cv2.findContours(
        slice_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    if perimeter <= 0.0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter ** 2))


def _filter_labels_by_roundness_xy(label_img, min_roundness, protect_min_vox=None):
    """Remove 3D labels whose largest XY slice is less circular than threshold.

    Parameters
    ----------
    protect_min_vox : float, optional
        Labels with at least this many voxels are never deleted here, even
        if non-round. A label this large is implausible as a single
        nucleus — it's an unsplit multi-nucleus cluster, and deleting it
        would erase multiple real nuclei outright. Leaving it merged (still
        one label) loses far less signal than removing it entirely. (See
        ``_split_nonround_clusters_by_intensity_peaks`` /
        ``_split_nonround_clusters_by_rethreshold``, which run first and
        try to properly separate these instead of just protecting them.)

    Returns
    -------
    out : ndarray
    removed : int
        Non-round labels below ``protect_min_vox`` that were deleted.
    protected : int
        Non-round labels at/above ``protect_min_vox`` that were left
        merged instead of deleted — i.e. clusters both upstream re-split
        strategies failed to separate. Zero means every non-round label was
        small enough to just delete outright (or there were none at all).
    """
    threshold = max(0.0, float(min_roundness))
    if threshold <= 0.0:
        return label_img, 0, 0

    out = np.asarray(label_img).copy()
    max_label = int(np.max(out))
    removed = 0
    protected = 0
    slices = ndi.find_objects(out) if max_label > 0 else []

    for label_id in range(1, max_label + 1):
        if slices[label_id - 1] is None:
            continue
        crop = out[_padded_slice(slices[label_id - 1], out.shape, 1)]  # view
        mask = crop == label_id
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            continue

        if _label_roundness_xy(mask) >= threshold:
            continue

        if protect_min_vox is not None and voxel_count >= protect_min_vox:
            protected += 1
            continue

        crop[mask] = 0
        removed += 1

    return out, removed, protected


def _split_nonround_clusters_by_intensity_peaks(
    label_img,
    intensity_img,
    spacing,
    min_roundness,
    expected_nucleus_vox,
    peak_min_distance_xy,
    peak_threshold_fraction,
    distance_weight=1.0,
    intensity_weight=1.0,
    gradient_weight=1.0,
    smooth_sigma=1.0,
    min_piece_vox_factor=0.35,
    max_piece_vox_factor=1.7,
):
    """Re-split non-round, multi-nucleus-sized labels using DAPI intensity peaks.

    Geometry/distance-only watershed can leave several touching nuclei as
    one label when the neck between them is too shallow to erode apart. For
    any label that is both (a) non-round and (b) large enough to plausibly
    hold more than one nucleus, this looks for distinct nucleus centers
    directly in the (smoothed) intensity signal — round nuclei that are
    individually distinguishable by brightness even when their combined
    outline isn't — and re-splits along the intensity dip/gradient between
    them (same elevation surface as the main watershed).

    Each resulting piece is judged independently against the roundness
    threshold and a plausible single-nucleus size window
    (``min_piece_vox_factor``-``max_piece_vox_factor`` x
    ``expected_nucleus_vox``) and only the pieces that pass are peeled off
    as new labels — a cluster where most pieces look like real nuclei and
    one or two don't isn't discarded wholesale; the failing voxels simply
    stay under the original label id, available for
    ``_split_nonround_clusters_by_rethreshold`` or another round of this
    same function to resolve further. Nothing is kept unless at least 2
    pieces pass (a single valid piece isn't useful — there'd be nothing
    left to compare it against).

    Parameters
    ----------
    label_img : ndarray (Z, Y, X), int
        Integer label volume (0 = background).
    intensity_img : ndarray (Z, Y, X)
        Intensity image the labels were segmented from (e.g. 'Filtered image'
        for the NUCLEI channel).
    spacing : tuple of float (Z, Y, X)
        Physical voxel size, used for the distance transform and elevation.
    min_roundness : float
        Same threshold used by ``_filter_labels_by_roundness_xy``; a label
        already at or above it is left alone (nothing to fix), and every
        candidate split piece must also meet it.
    expected_nucleus_vox : float
        Volume (voxels) of one typical full-diameter nucleus.
    peak_min_distance_xy, peak_threshold_fraction : float
        Passed to ``detect_peaks_xy_intensity_aware`` for locating nucleus
        centers within the label. Use tighter/more permissive values than
        the main seeding pass (e.g. half the distance, ~55% the threshold)
        since this is specifically targeting a cluster that already failed
        to separate at the default sensitivity.
    distance_weight, intensity_weight, gradient_weight, smooth_sigma : float
        Passed through to ``_gradient_watershed_elevation`` for the local
        re-split.
    min_piece_vox_factor, max_piece_vox_factor : float
        A candidate split piece must fall within
        [``min_piece_vox_factor``, ``max_piece_vox_factor``] x
        ``expected_nucleus_vox`` to be accepted (default 0.35-1.7x) — too
        small suggests a spurious noise peak, too large means the piece
        still isn't a single nucleus.

    Returns
    -------
    out : ndarray, same shape as label_img
        Updated label volume with valid splits applied.
    split_count : int
        Number of labels successfully split into >= 2 valid pieces.
    """
    from skimage.segmentation import watershed

    out = np.asarray(label_img).copy()
    max_label = int(out.max())
    next_label = max_label + 1
    split_count = 0

    # Absolute floor only — just enough to reject a stray noise-sized
    # sliver. The real size check below is relative to what THIS cluster's
    # own peak count implies, not a fixed nucleus-size assumption, since
    # ``expected_nucleus_vox`` (derived from Cell 3's ``nuclei_diameter``)
    # is a rough population-level estimate that individual clusters —
    # especially unusually large or dense ones — can legitimately deviate
    # from.
    absolute_min_piece_vox = 0.1 * expected_nucleus_vox
    piece_roundness_floor = max(0.0, min_roundness * 0.8)

    # Each candidate is handled in its bounding box, padded by the smoothing
    # radius and the peak search distance (+1): the same result as on the
    # whole image. Labels too small to be a cluster are skipped without
    # building their mask (their volume only changes when they are split).
    volumes = np.bincount(out.ravel(), minlength=max_label + 1)
    slices = ndi.find_objects(out) if max_label > 0 else []
    pad = max(_gaussian_radius(max(float(smooth_sigma), 1e-3)), int(peak_min_distance_xy)) + 1

    for label_id in range(1, max_label + 1):
        if volumes[label_id] < 1.5 * expected_nucleus_vox or slices[label_id - 1] is None:
            continue
        sl = _padded_slice(slices[label_id - 1], out.shape, pad)
        out_c = out[sl]  # view: writes go to out
        mask = out_c == label_id
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count < 1.5 * expected_nucleus_vox:
            continue
        if _label_roundness_xy(mask) >= min_roundness:
            continue

        intensity_c = intensity_img[sl]
        local_intensity = np.where(mask, intensity_c, 0.0).astype(np.float32)
        local_distance = ndi.distance_transform_edt(mask, sampling=spacing)
        peak_coords_3d = detect_peaks_xy_intensity_aware(
            local_distance,
            local_intensity,
            peak_min_distance_xy,
            peak_threshold_fraction,
            distance_weight=distance_weight,
            intensity_weight=intensity_weight,
        )
        if len(peak_coords_3d) < 2:
            continue

        seeds = np.zeros(mask.shape, dtype=np.int32)
        for i, coord in enumerate(peak_coords_3d, start=1):
            seeds[coord] = i

        elevation = _gradient_watershed_elevation(
            intensity_c,
            mask,
            spacing=spacing,
            smooth_sigma=smooth_sigma,
            w_distance=distance_weight,
            w_intensity=intensity_weight,
            w_gradient=gradient_weight,
        )
        sub_labels = watershed(elevation, seeds, mask=mask)

        # Piece size is judged against the average piece size THIS split
        # implies (cluster volume / number of peaks) rather than the
        # global expected_nucleus_vox — so a cluster of unusually large (or
        # small) nuclei still validates correctly as long as its own peaks
        # divide it reasonably evenly.
        avg_piece_vox = voxel_count / len(peak_coords_3d)
        local_min_piece_vox = max(min_piece_vox_factor * avg_piece_vox, absolute_min_piece_vox)
        local_max_piece_vox = max_piece_vox_factor * avg_piece_vox

        # Partial accept: each watershed piece is judged independently. A
        # cluster with e.g. 8 peaks where 7 pieces look like real nuclei and
        # 1 is still an oversized unresolved remainder should keep those 7,
        # not be discarded wholesale for the one bad piece.
        valid_pieces = []
        for sub_id in range(1, int(sub_labels.max()) + 1):
            piece_mask = sub_labels == sub_id
            piece_vox = int(np.count_nonzero(piece_mask))
            if piece_vox == 0:
                continue
            if (local_min_piece_vox <= piece_vox <= local_max_piece_vox
                    and _label_roundness_xy(piece_mask) >= piece_roundness_floor):
                valid_pieces.append(piece_mask)

        if len(valid_pieces) < 2:
            continue

        # Peel off the valid pieces only; anything left over (rejected
        # pieces, gaps) keeps the original label id so it's still available
        # for the rethreshold pass or a later round of this same pass.
        for piece_mask in valid_pieces:
            out_c[piece_mask] = next_label
            next_label += 1
        split_count += 1

    return out, split_count


def _split_nonround_clusters_by_rethreshold(
    label_img,
    intensity_img,
    spacing,
    min_roundness,
    expected_nucleus_vox,
    min_piece_vox_factor=0.35,
    max_piece_vox_factor=1.7,
    percentiles=(55, 65, 75, 85),
    smooth_sigma=1.0,
    distance_weight=1.0,
    intensity_weight=1.0,
    gradient_weight=1.0,
):
    """Re-split non-round, multi-nucleus-sized labels by raising the local
    intensity threshold until the dim bridge between touching nuclei
    disconnects, then growing each resulting core back out to the label's
    original extent with a gradient-aware watershed.

    Complements ``_split_nonround_clusters_by_intensity_peaks``: peak
    detection needs a resolvable local maximum per nucleus, which fails on
    heavily blurred/merged signal where the cores still differ in
    brightness from the connecting bridge but don't form distinct peaks.
    Raising the threshold only requires *some* brightness margin between
    core and bridge, which is a much lower bar — the same idea as the
    combined-threshold recipe in Cell 15, just applied locally to one
    cluster instead of the whole channel.

    For each qualifying label, intensity percentiles *within that label*
    (not the whole image) are tried from least to most aggressive; each
    percentile's hard core mask (after a 1-voxel erosion, to fully sever
    thin single-voxel bridges) is used as watershed markers, flooded back
    out to the label's full original extent, and every resulting piece is
    judged independently against the roundness and size checks (same
    partial-accept logic as ``_split_nonround_clusters_by_intensity_peaks``).
    Whichever percentile yields the most valid pieces (>= 2 required) wins;
    those pieces are peeled off as new labels and any rejected/leftover
    voxels stay under the original label id for another round. If no
    percentile ever reaches 2 valid pieces, the label is left untouched.

    Parameters
    ----------
    label_img : ndarray (Z, Y, X), int
    intensity_img : ndarray (Z, Y, X)
    spacing : tuple of float (Z, Y, X)
    min_roundness : float
    expected_nucleus_vox : float
        Volume (voxels) of one typical full-diameter nucleus.
    min_piece_vox_factor, max_piece_vox_factor : float
        Accepted single-nucleus size window, as a fraction of
        ``expected_nucleus_vox`` (default 0.35-1.7x).
    percentiles : sequence of float
        Intensity percentiles (within the label) tried in order, least to
        most aggressive, until one yields a valid split.
    smooth_sigma, distance_weight, intensity_weight, gradient_weight : float
        Passed through to ``_gradient_watershed_elevation`` for growing the
        cores back out to the label's original extent.

    Returns
    -------
    out : ndarray, same shape as label_img
    split_count : int
        Number of labels successfully split into >= 2 valid pieces.
    """
    from skimage.segmentation import watershed

    out = np.asarray(label_img).copy()
    max_label = int(out.max())
    next_label = max_label + 1
    split_count = 0

    # Absolute floor only — the real per-piece size check is relative to
    # what each percentile's own core count implies (see avg_piece_vox
    # below), not a fixed nucleus-size assumption. See the matching note in
    # ``_split_nonround_clusters_by_intensity_peaks``.
    absolute_min_piece_vox = 0.1 * expected_nucleus_vox
    piece_roundness_floor = max(0.0, min_roundness * 0.8)
    min_core_vox = max(4, int(0.15 * expected_nucleus_vox))

    # As `_split_nonround_clusters_by_intensity_peaks`: each candidate in its
    # bounding box padded by the smoothing radius (+1), which gives the
    # whole-image result; the smoothing is computed on that crop only
    # (it was recomputed on the whole image for every candidate).
    sigma = max(float(smooth_sigma), 1e-3)
    volumes = np.bincount(out.ravel(), minlength=max_label + 1)
    slices = ndi.find_objects(out) if max_label > 0 else []
    pad = _gaussian_radius(sigma) + 1

    for label_id in range(1, max_label + 1):
        if volumes[label_id] < 1.5 * expected_nucleus_vox or slices[label_id - 1] is None:
            continue
        sl = _padded_slice(slices[label_id - 1], out.shape, pad)
        out_c = out[sl]  # view: writes go to out
        mask = out_c == label_id
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count < 1.5 * expected_nucleus_vox:
            continue
        if _label_roundness_xy(mask) >= min_roundness:
            continue

        intensity_c = intensity_img[sl]
        smoothed = ndi.gaussian_filter(intensity_c.astype(np.float32), sigma=sigma)
        values = smoothed[mask]
        if values.size == 0:
            continue

        # Try every percentile (rather than stopping at the first that
        # clears the bar) and keep whichever gives the most valid pieces —
        # partial accept, same reasoning as the peak-based pass: a cluster
        # where most pieces look like real nuclei and a remainder doesn't
        # should keep the good pieces rather than be discarded wholesale.
        best_valid_pieces = []

        for pct in percentiles:
            cutoff = float(np.percentile(values, pct))
            core_mask = mask & (smoothed >= cutoff)
            core_mask = ndi.binary_erosion(core_mask, iterations=1)
            core_labels, n_cores = ndi.label(core_mask)

            sizes = np.bincount(core_labels.ravel(), minlength=n_cores + 1)
            keep_ids = [i for i in range(1, n_cores + 1) if sizes[i] >= min_core_vox]
            if len(keep_ids) < 2:
                continue

            seeds = np.zeros_like(core_labels)
            for new_id, old_id in enumerate(keep_ids, start=1):
                seeds[core_labels == old_id] = new_id

            elevation = _gradient_watershed_elevation(
                intensity_c,
                mask,
                spacing=spacing,
                smooth_sigma=smooth_sigma,
                w_distance=distance_weight,
                w_intensity=intensity_weight,
                w_gradient=gradient_weight,
            )
            sub_labels = watershed(elevation, seeds, mask=mask)

            avg_piece_vox = voxel_count / len(keep_ids)
            local_min_piece_vox = max(min_piece_vox_factor * avg_piece_vox, absolute_min_piece_vox)
            local_max_piece_vox = max_piece_vox_factor * avg_piece_vox

            valid_pieces = []
            for sub_id in range(1, int(sub_labels.max()) + 1):
                piece_mask = sub_labels == sub_id
                piece_vox = int(np.count_nonzero(piece_mask))
                if piece_vox == 0:
                    continue
                if (local_min_piece_vox <= piece_vox <= local_max_piece_vox
                        and _label_roundness_xy(piece_mask) >= piece_roundness_floor):
                    valid_pieces.append(piece_mask)

            if len(valid_pieces) > len(best_valid_pieces):
                best_valid_pieces = valid_pieces

        if len(best_valid_pieces) >= 2:
            for piece_mask in best_valid_pieces:
                out_c[piece_mask] = next_label
                next_label += 1
            split_count += 1

    return out, split_count


def segment_nuclei_watershed(
    binary_mask,
    r_zX,
    r_zY,
    r_zZ,
    nuclei_diameter,
    nuclei_z_anisotropy_factor=1.0,
    nuclei_bridge_shrink_factor=0.28,
    nuclei_split_diameter_min_factor=0.5,
    nuclei_split_diameter_max_factor=1.5,
    nuclei_split_diameter_scales=3,
    nuclei_seed_min_fraction=0.03,
    nuclei_min_roundness=0.45,
    z_split_aggressive=False,
    intensity_img=None,
    intensity_img_raw=None,
    split_by_intensity_gradient=True,
    nuclei_gradient_smooth_sigma=1.0,
    nuclei_distance_weight=1.0,
    nuclei_intensity_weight=1.0,
    nuclei_gradient_weight=1.0,
    progress=None,
):
    """
    Segment nuclei using watershed with multi-scale erosion and EDT peak fallback.

    Implements a robust watershed-based segmentation algorithm that:
    1. Handles Z-anisotropy by adjusting erosion depths
    2. Uses multi-scale erosion to create multiple seed points for dumbbell/multi-lobed nuclei
    3. Falls back to EDT-peak detection, blended with intensity when available
       (see ``split_by_intensity_gradient``), when erosion fails to split nuclei
    4. Detects boundary-touching components to apply gentler erosion
    5. Preserves isolated single-nuclei labels during cleanup

    Parameters
    ----------
    binary_mask : ndarray, shape (Z, Y, X), dtype bool
        Binary mask of foreground (nuclear) regions.

    r_zX, r_zY, r_zZ : float
        Voxel physical spacing in micrometers along X, Y, Z axes.

    nuclei_diameter : float
        Approximate nucleus diameter in micrometers (used for erosion scaling).

    nuclei_z_anisotropy_factor : float, optional
        Scaling factor for Z-axis weighting (default=1.0).
        Values > 1 reduce Z erosion; < 1 increase it.

    nuclei_bridge_shrink_factor : float, optional
        Fraction of nucleus diameter to erode for bridge-breaking (default=0.28).
        Range [0.1, 0.5]; higher values = more aggressive erosion.

    nuclei_split_diameter_min_factor : float, optional
        Minimum erosion scale as fraction of diameter (default=0.5).

    nuclei_split_diameter_max_factor : float, optional
        Maximum erosion scale as fraction of diameter (default=1.5).

    nuclei_split_diameter_scales : int, optional
        Number of erosion scales to evaluate (default=3).

    nuclei_seed_min_fraction : float, optional
        Minimum seed size as fraction of expected nucleus volume (default=0.03).

    nuclei_min_roundness : float, optional
        Minimum circularity on the largest XY slice for a nucleus to be kept (default=0.45).
        Uses circularity = 4*pi*area/perimeter^2. Typical range is [0.0, 1.0].
        Two (or more) touching round nuclei that fail to be split apart form
        one irregular blob; if that blob's volume is still nucleus-sized
        (< ~1.5x one expected nucleus) it gets removed entirely by this
        filter. The ``split_by_intensity_gradient`` seeding/watershed below
        exists to catch these cases upstream, before roundness is even
        checked. Larger unsplit clusters are never deleted by this filter at
        all — see the note under ``split_by_intensity_gradient``.

    intensity_img : ndarray, shape (Z, Y, X), optional
        Filtered/smoothed intensity image (same channel the mask was
        thresholded from — typically 'Filtered image'), used when
        ``split_by_intensity_gradient`` is True to find nucleus centers by
        their actual brightness peaks (not just mask shape) and to bias the
        main watershed boundary toward where the signal itself dips between
        two touching nuclei. If None, behaves exactly like the original
        shape/distance-only algorithm.

    intensity_img_raw : ndarray, shape (Z, Y, X), optional
        Same channel as ``intensity_img`` but from an earlier pipeline
        stage that hasn't been through the per-channel contrast/gamma clip
        (e.g. 'Denoised image', pre-Cell-12) — used instead of
        ``intensity_img`` for the two post-watershed cluster re-split
        strategies below. Dense, unusually bright clusters are exactly the
        regions most likely to have been clipped to the display ceiling by
        contrast limits chosen for normal-brightness nuclei, which ties
        many voxels to the identical max value and makes any percentile- or
        peak-based re-split blind to real internal structure that's still
        present in the less-processed data. Falls back to ``intensity_img``
        if not given.

    split_by_intensity_gradient : bool, optional
        If True (default) and ``intensity_img`` is given, intensity is used
        at three points instead of shape/distance alone: (1) when erosion
        leaves a connected component with only one seed, also search for
        seeds using a combined distance+intensity peak map, so two
        individually round, touching nuclei with only a shallow geometric
        neck but a real intensity dip between them still get two seeds; (2)
        run the watershed on a distance+intensity+gradient elevation
        surface (see ``_gradient_watershed_elevation``) instead of pure
        ``-distance``, so the split boundary snaps to where the marker
        signal actually fades between nuclei; (3) after that watershed and
        cleanup, any label that's still both non-round and clearly too big
        to be one nucleus (>= ~1.5x the expected single-nucleus volume) is
        run back through two dedicated local re-split attempts, in order:
        ``_split_nonround_clusters_by_intensity_peaks`` (a more permissive
        peak search than step 1), then, for whatever that still can't
        resolve, ``_split_nonround_clusters_by_rethreshold`` — which raises
        the *local* intensity threshold within just that label until the
        dim bridge connecting the nuclei disconnects into separate bright
        cores, then grows each core back out to the label's original
        extent. The re-threshold strategy only needs a brightness margin
        between core and bridge, not a resolvable peak, so it can succeed
        on heavily blurred/merged signal where peak detection can't. Both
        strategies use partial accept: each resulting piece is judged
        independently, so a cluster of e.g. 8 sub-nuclei where 7 pieces
        come out valid and 1 doesn't still keeps those 7, instead of the
        whole split being discarded for one bad piece — the failing
        remainder simply keeps the original label id. This pair of
        attempts is repeated for up to 3 rounds so a remainder that's still
        too big to be one nucleus gets picked up again, peeling off more
        nuclei each round until nothing further can be resolved. As a
        last-resort safety net, ``nuclei_min_roundness`` never deletes a
        component that's still >= ~1.5x one expected nucleus after all
        rounds — whatever's left merged is kept as one label covering
        several nuclei rather than erased outright, since losing it loses
        more real signal than keeping it merged does.

    nuclei_gradient_smooth_sigma : float, optional
        Smoothing (voxels) applied to ``intensity_img`` before measuring
        intensity/gradient for seeding and the final elevation surface
        (default=1.0). Only used when ``split_by_intensity_gradient`` is True.

    nuclei_distance_weight, nuclei_intensity_weight, nuclei_gradient_weight : float, optional
        Relative weights of the three elevation/seeding terms — geometric
        distance, absolute dimness, and steepness of intensity drop — each
        default 1.0. Only used when ``split_by_intensity_gradient`` is True.
        Raise ``nuclei_gradient_weight`` to make splitting more sensitive to
        a fast intensity drop between two closely touching, bright nuclei.

    Returns
    -------
    im_out : ndarray, shape (Z, Y, X), dtype int32
        Labeled nucleus segmentation (label value per voxel). Label 0 = background.

    debug_info : dict
        Debugging information with keys:
        - 'z_anisotropy': Computed Z anisotropy factor
        - 'z_split_weight': Effective Z-weighting applied to erosion
        - 'scales_with_seeds': Number of erosion scales that produced markers
        - 'added_peak_seed_count': Number of seeds added via EDT/intensity peak detection
        - 'added_seed_count': Number of fallback seeds added
        - 'erosion_triplets': List of (z, y, x) erosion radii attempted
        - 'boundary_components': Number of boundary-touching components
        - 'restored_isolated_count': Voxels restored after cleanup
        - 'dmin', 'dmax', 'n_scales': Diameter scaling parameters
        - 'min_seed_vox': Minimum seed size threshold in voxels
        - 'expected_nucleus_vox': Volume (voxels) of one typical full-diameter nucleus
        - 'multi_nucleus_vox_thresh': Volume (voxels) above which a component is
          treated as an unsplit multi-nucleus cluster (triggers the two re-split
          strategies below and is protected from roundness deletion if both fail)
        - 'peak_cluster_splits': Number of non-round, multi-nucleus-sized labels
          re-split by ``_split_nonround_clusters_by_intensity_peaks``
        - 'rethreshold_cluster_splits': Number of non-round, multi-nucleus-sized
          labels re-split by ``_split_nonround_clusters_by_rethreshold`` (tried
          on whatever the peak-based pass above couldn't resolve)
        - 'nonround_cluster_splits': Sum of the two counts above
        - 'removed_by_roundness': Number of (small) non-round labels dropped entirely
        - 'protected_nonround_clusters': Number of non-round, multi-nucleus-sized
          labels that BOTH re-split strategies failed to separate, so were left
          merged instead of deleted. If this is 0 and 'nonround_cluster_splits'
          is also 0, there simply were no non-round multi-nucleus-sized labels
          left at this point — nothing needed fixing.
        - 'used_intensity_gradient': Whether intensity-aware seeding/splitting ran

    Notes
    -----
    - Modifies parameter r_zZ by computing effective_r_zZ for Z-anisotropy handling
    - Uses peak_local_max from skimage.feature for EDT/intensity-based fallback seeding
    - Applies watershed transform with negative distance map (or, when
      ``split_by_intensity_gradient`` is active, a distance+intensity+gradient
      elevation surface) to favor markers
    - Removes small island labels but preserves those touching image boundaries

    Example
    -------
    >>> binary = im_threshold > 0  # Binary foreground mask
    >>> nuclei, debug = segment_nuclei_watershed(
    ...     binary,
    ...     r_zX=0.1, r_zY=0.1, r_zZ=0.2,
    ...     nuclei_diameter=30.0,
    ...     nuclei_bridge_shrink_factor=0.28,
    ...     intensity_img=filtered_nuclei_channel,
    ... )
    >>> print(f"Nuclei found: {int(nuclei.max())}, Peak-based seeds: {debug['added_peak_seed_count']}")
    """
    from skimage import morphology
    from skimage.segmentation import watershed, relabel_sequential

    use_intensity = split_by_intensity_gradient and intensity_img is not None
    smoothed_intensity = None
    if use_intensity:
        smoothed_intensity = ndi.gaussian_filter(
            intensity_img.astype(np.float32), sigma=max(float(nuclei_gradient_smooth_sigma), 1e-3)
        )

    # Derive Z weighting from the actual voxel anisotropy so depth
    # separation follows the measured Z versus XY resolution.
    xy_spacing = max(1e-6, float(np.mean([r_zY, r_zX])))
    z_anisotropy = max(1.0, float(r_zZ / xy_spacing))
    z_split_weight = z_anisotropy * max(0.25, float(nuclei_z_anisotropy_factor))
    effective_r_zZ = r_zZ / z_split_weight

    # Identify boundary-touching components to apply gentler erosion.
    boundary_mask = np.zeros_like(binary_mask, dtype=bool)
    boundary_margin = 3
    boundary_mask[0:boundary_margin, :, :] = True
    boundary_mask[-boundary_margin:, :, :] = True
    boundary_mask[:, 0:boundary_margin, :] = True
    boundary_mask[:, -boundary_margin:, :] = True
    boundary_mask[:, :, 0:boundary_margin] = True
    boundary_mask[:, :, -boundary_margin:] = True

    cc_labels_for_boundary, _ = ndi.label(binary_mask)
    boundary_components = set(
        np.unique(cc_labels_for_boundary[boundary_mask]).tolist()
    ) - {0}

    # Evaluate several size priors from strong to weak erosion.
    dmin = max(0.2, float(nuclei_split_diameter_min_factor))
    dmax = max(dmin, float(nuclei_split_diameter_max_factor))
    n_scales = max(2, int(nuclei_split_diameter_scales))
    diameter_factors = np.linspace(dmax, dmin, n_scales)

    min_diameter_um = nuclei_diameter * dmin
    min_radius_um = min_diameter_um * 0.5
    min_nucleus_volume_um3 = (4.0 * np.pi * (min_radius_um ** 3.0)) / 3.0
    voxel_um3 = r_zZ * r_zY * r_zX
    min_expected_nucleus_vox = max(1, int(min_nucleus_volume_um3 / voxel_um3))
    min_seed_vox = max(8, int(min_expected_nucleus_vox * nuclei_seed_min_fraction))
    min_seed_vox_boundary = max(4, int(min_seed_vox * 0.6))

    # Volume of one typical (full-diameter) nucleus, in voxels. Connected
    # components well above this are almost certainly multiple nuclei
    # merged together, not one oddly-shaped nucleus — used below both to
    # retry seeding more aggressively on stuck clusters and to stop the
    # roundness cleanup filter from deleting them outright if splitting
    # still fails.
    full_nucleus_vol_um3 = (4.0 * np.pi * ((nuclei_diameter / 2.0) ** 3)) / 3.0
    expected_nucleus_vox = max(1.0, full_nucleus_vol_um3 / voxel_um3)
    multi_nucleus_vox_thresh = 1.5 * expected_nucleus_vox

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    next_marker = 0
    erosion_triplets = []
    scales_with_seeds = 0

    # Voxels of a mask component touching the image border (whose seeds may
    # be smaller), computed once.
    boundary_cc_lut = np.zeros(int(cc_labels_for_boundary.max()) + 1, dtype=bool)
    if boundary_components:
        boundary_cc_lut[np.fromiter(boundary_components, dtype=np.int64)] = True
    in_boundary_cc = boundary_cc_lut[cc_labels_for_boundary]
    del boundary_cc_lut

    def _valid_seed_ids(mk, num_seeds):
        # Seeds big enough to keep: per-seed size and border contact counted
        # in one pass each, instead of one whole-image mask per seed.
        sizes = np.bincount(mk.ravel(), minlength=num_seeds + 1)
        touches = np.bincount(mk[in_boundary_cc], minlength=num_seeds + 1) > 0
        threshold = np.where(touches, min_seed_vox_boundary, min_seed_vox)
        ok = sizes >= threshold
        ok[0] = False
        return np.flatnonzero(ok)

    def _add_seeds(mk, num_seeds, seed_ids):
        # Seeds overlapping no marker yet get the next marker ids, in seed
        # order (seeds of one erosion are disjoint, so checking them all
        # against the markers before this erosion is the same as one by one).
        nonlocal next_marker
        taken = np.bincount(mk[markers > 0], minlength=num_seeds + 1) > 0
        new_ids = [int(s) for s in seed_ids if not taken[s]]
        if not new_ids:
            return
        lut = np.zeros(num_seeds + 1, dtype=np.int32)
        lut[new_ids] = np.arange(next_marker + 1, next_marker + 1 + len(new_ids), dtype=np.int32)
        added = lut[mk]
        sel = added > 0
        markers[sel] = added[sel]
        next_marker += len(new_ids)

    for d_factor in diameter_factors:
        shrink_um = nuclei_diameter * d_factor * nuclei_bridge_shrink_factor
        er_z_i = max(1, int(shrink_um / effective_r_zZ))
        er_y_i = max(1, int(shrink_um / r_zY))
        er_x_i = max(1, int(shrink_um / r_zX))
        erosion_fp_i = make_anisotropic_footprint(er_z_i, er_y_i, er_x_i)
        eroded_i = morphology.erosion(binary_mask, footprint=erosion_fp_i)
        erosion_triplets.append((er_z_i, er_y_i, er_x_i))

        if not np.any(eroded_i):
            continue

        mk_i, num_i = ndi.label(eroded_i)
        if num_i == 0:
            continue

        valid_ids = _valid_seed_ids(mk_i, num_i)
        if len(valid_ids) == 0:
            continue

        scales_with_seeds += 1
        _add_seeds(mk_i, num_i, valid_ids)

    num = int(markers.max())

    # If all scales are too strong, relax once with half radii.
    if num == 0:
        for er_z_i, er_y_i, er_x_i in erosion_triplets:
            er_z_r = max(1, er_z_i // 2)
            er_y_r = max(1, er_y_i // 2)
            er_x_r = max(1, er_x_i // 2)
            erosion_fp_r = make_anisotropic_footprint(er_z_r, er_y_r, er_x_r)
            eroded_r = morphology.erosion(binary_mask, footprint=erosion_fp_r)
            if not np.any(eroded_r):
                continue
            mk_r, num_r = ndi.label(eroded_r)
            if num_r == 0:
                continue
            _add_seeds(mk_r, num_r, _valid_seed_ids(mk_r, num_r))

        num = int(markers.max())
    del in_boundary_cc

    # Fallback: if shrinking removes everything, keep one seed per
    # connected component of the original mask rather than failing.
    if num == 0:
        markers, num = ndi.label(binary_mask)

    distance = ndi.distance_transform_edt(
        binary_mask, sampling=[effective_r_zZ, r_zY, r_zX]
    )

    cc_labels, num_cc = ndi.label(binary_mask)
    next_marker = int(markers.max())
    added_seed_count = 0
    added_peak_seed_count = 0
    peak_min_distance_xy = max(
        1,
        int((0.45 * nuclei_diameter) / max(1e-6, np.mean([r_zX, r_zY]))),
    )
    peak_threshold_fraction = 0.45

    # Each component is handled in its bounding box, padded by the peak
    # search distance (+1) so the peaks found are the whole-image ones.
    cc_slices = ndi.find_objects(cc_labels)
    cc_pad = peak_min_distance_xy + 1
    for cc_id in _progress_iter(
        range(1, num_cc + 1), progress, desc='Step 17 - Seed Nuclei Components', leave=False
    ):
        if cc_slices[cc_id - 1] is None:
            continue
        sl = _padded_slice(cc_slices[cc_id - 1], cc_labels.shape, cc_pad)
        markers_c = markers[sl]  # view: writes go to markers
        cc_mask = cc_labels[sl] == cc_id
        cc_marker_ids = np.unique(markers_c[cc_mask])
        cc_marker_ids = cc_marker_ids[cc_marker_ids > 0]
        local_distance = np.where(cc_mask, distance[sl], 0.0)

        # Recovery for dumbbell-like nuclei, or two touching-but-individually-
        # round nuclei with too shallow a geometric neck for erosion to
        # split: if erosion still leaves only one seed, try to split the
        # component using multiple peaks. When intensity is available, the
        # peaks are found in a combined distance+intensity map so a real
        # brightness dip between two nuclei (not just a mask constriction)
        # is enough to place two seeds; otherwise falls back to EDT-only
        # peaks. Uses 2D XY projection to prevent over-splitting across Z.
        if cc_marker_ids.size <= 1 and np.any(cc_mask):
            if use_intensity:
                local_intensity = np.where(cc_mask, smoothed_intensity[sl], 0.0)
                peak_coords_3d = detect_peaks_xy_intensity_aware(
                    local_distance,
                    local_intensity,
                    peak_min_distance_xy,
                    peak_threshold_fraction,
                    distance_weight=nuclei_distance_weight,
                    intensity_weight=nuclei_intensity_weight,
                )
            else:
                peak_coords_3d = detect_peaks_xy_with_best_z(
                    local_distance,
                    peak_min_distance_xy,
                    peak_threshold_fraction,
                )

            if len(peak_coords_3d) >= 2:
                if cc_marker_ids.size == 1:
                    markers_c[(markers_c == int(cc_marker_ids[0])) & cc_mask] = 0
                for peak_coord in peak_coords_3d:
                    if markers_c[peak_coord] != 0:
                        continue
                    next_marker += 1
                    markers_c[peak_coord] = next_marker
                    added_peak_seed_count += 1
                cc_marker_ids = np.unique(markers_c[cc_mask])
                cc_marker_ids = cc_marker_ids[cc_marker_ids > 0]

        if cc_marker_ids.size == 0:
            seed_pos = np.unravel_index(np.argmax(local_distance), local_distance.shape)
            next_marker += 1
            markers_c[seed_pos] = next_marker
            added_seed_count += 1

    if use_intensity:
        elevation = _gradient_watershed_elevation(
            intensity_img,
            binary_mask,
            spacing=(effective_r_zZ, r_zY, r_zX),
            smooth_sigma=nuclei_gradient_smooth_sigma,
            w_distance=nuclei_distance_weight,
            w_intensity=nuclei_intensity_weight,
            w_gradient=nuclei_gradient_weight,
        )
        im_out = watershed(elevation, markers, mask=binary_mask)
    else:
        im_out = watershed(-distance, markers, mask=binary_mask)

    # Preserve labels that are alone in their mask component, and all labels
    # of components touching the image border. Computed from the unique
    # (component, label) pairs in one pass instead of one scan per component.
    preserve_labels = set()
    both = (cc_labels > 0) & (im_out > 0)
    if both.any():
        n_labels = int(im_out.max()) + 1
        keys = np.unique(cc_labels[both].astype(np.int64) * n_labels + im_out[both])
        pair_cc, pair_label = keys // n_labels, keys % n_labels
        cc_ids, labels_per_cc = np.unique(pair_cc, return_counts=True)
        keep_cc = set(cc_ids[labels_per_cc == 1].tolist()) | boundary_components
        preserve_labels = {
            int(label) for cc_id, label in zip(pair_cc.tolist(), pair_label.tolist())
            if cc_id in keep_cc
        }

    min_cell_vox = max(8, int(0.3 * full_nucleus_vol_um3 / voxel_um3))

    im_out_before_cleanup = im_out.copy()
    _, im_out = remove_small_island_labels(
        im_out, connectivity=1, size_ratio_thresh=0.5,
        min_cell_size=min_cell_vox,
    )

    restored_isolated_count = 0
    if len(preserve_labels) > 0:
        preserve_mask = np.isin(im_out_before_cleanup, list(preserve_labels))
        restore_mask = preserve_mask & (im_out == 0)
        restored_isolated_count = int(np.count_nonzero(restore_mask))
        im_out[restore_mask] = im_out_before_cleanup[restore_mask]

    # Before falling back to "just don't delete it", actively try to
    # separate any label that's both non-round and large enough to
    # plausibly be several merged nuclei. Two strategies are tried in turn:
    # (1) find distinct DAPI intensity peaks per nucleus; (2) if that fails
    # (common on heavily blurred/merged signal with no resolvable peak),
    # raise the local intensity threshold until the dim bridge connecting
    # the nuclei disconnects into separate bright cores, then grow each
    # core back out to the label's original extent. Both use partial
    # accept — valid pieces are peeled off immediately and any oversized
    # remainder keeps its original label id — so this is repeated for a
    # few rounds: a remainder that's still too big to be one nucleus gets
    # picked up again next round, peeling off more nuclei each time until
    # nothing further can be resolved.
    peak_cluster_splits = 0
    rethreshold_cluster_splits = 0
    if use_intensity:
        # Prefer the pre-contrast-clip source for these two local re-split
        # strategies: a dense, unusually bright cluster is exactly the kind
        # of region likely to have been clipped to the display ceiling by
        # Cell 12's contrast limits, which ties many voxels to the same max
        # value and hides real internal structure that's still present
        # before that clip was applied.
        split_intensity_img = intensity_img_raw if intensity_img_raw is not None else intensity_img
        for _round in _progress_iter(
            range(3), progress, desc='Step 17 - Split Non-round Clusters', leave=False
        ):
            im_out, round_peak_splits = _split_nonround_clusters_by_intensity_peaks(
                im_out,
                split_intensity_img,
                spacing=(effective_r_zZ, r_zY, r_zX),
                min_roundness=nuclei_min_roundness,
                expected_nucleus_vox=expected_nucleus_vox,
                peak_min_distance_xy=max(1, peak_min_distance_xy // 2),
                peak_threshold_fraction=peak_threshold_fraction * 0.55,
                distance_weight=nuclei_distance_weight,
                intensity_weight=nuclei_intensity_weight,
                gradient_weight=nuclei_gradient_weight,
                smooth_sigma=nuclei_gradient_smooth_sigma,
            )
            im_out, round_rethreshold_splits = _split_nonround_clusters_by_rethreshold(
                im_out,
                split_intensity_img,
                spacing=(effective_r_zZ, r_zY, r_zX),
                min_roundness=nuclei_min_roundness,
                expected_nucleus_vox=expected_nucleus_vox,
                distance_weight=nuclei_distance_weight,
                intensity_weight=nuclei_intensity_weight,
                gradient_weight=nuclei_gradient_weight,
                smooth_sigma=nuclei_gradient_smooth_sigma,
            )
            peak_cluster_splits += round_peak_splits
            rethreshold_cluster_splits += round_rethreshold_splits
            if round_peak_splits == 0 and round_rethreshold_splits == 0:
                break
    nonround_cluster_splits = peak_cluster_splits + rethreshold_cluster_splits

    # Any cluster that still couldn't be split (e.g. no distinct peaks and
    # no clean brightness margin even after both attempts above) is
    # protected here rather than deleted: losing the whole cluster loses
    # far more real signal than leaving it merged as one label.
    im_out, removed_by_roundness, protected_nonround_clusters = _filter_labels_by_roundness_xy(
        im_out,
        nuclei_min_roundness,
        protect_min_vox=multi_nucleus_vox_thresh,
    )

    im_out, z_consistency_splits = _split_labels_by_z_consistency(
        im_out, min_fragment_vox=min_seed_vox,
        aggressive=z_split_aggressive,
    )

    im_out, z_consistency_merges = _merge_labels_by_z_consistency(im_out)

    im_out, _, _ = relabel_sequential(im_out)

    debug_info = {
        'z_anisotropy': z_anisotropy,
        'z_split_weight': z_split_weight,
        'scales_with_seeds': scales_with_seeds,
        'added_peak_seed_count': added_peak_seed_count,
        'peak_cluster_splits': peak_cluster_splits,
        'rethreshold_cluster_splits': rethreshold_cluster_splits,
        'nonround_cluster_splits': nonround_cluster_splits,
        'added_seed_count': added_seed_count,
        'erosion_triplets': erosion_triplets,
        'boundary_components': boundary_components,
        'restored_isolated_count': restored_isolated_count,
        'dmin': dmin,
        'dmax': dmax,
        'n_scales': n_scales,
        'min_seed_vox': min_seed_vox,
        'min_cell_vox': min_cell_vox,
        'expected_nucleus_vox': expected_nucleus_vox,
        'multi_nucleus_vox_thresh': multi_nucleus_vox_thresh,
        'nuclei_min_roundness': float(max(0.0, nuclei_min_roundness)),
        'removed_by_roundness': removed_by_roundness,
        'protected_nonround_clusters': protected_nonround_clusters,
        'z_consistency_splits': z_consistency_splits,
        'z_consistency_merges': z_consistency_merges,
        'used_intensity_gradient': use_intensity,
    }

    return im_out, debug_info


# ── Optional processing steps (Cells 9-14) ─────────────────────────────────
# Each processing cell (9-14) can be skipped: a step reads the most recent
# stage computed before it (e.g. Cell 12 reads the 'Zoomed image' when Cell 11
# was not run), and `complete_skipped_steps` (Cell 15) stands the stage of
# every skipped step in for its input, so the threshold and every later cell
# find the stage names they read. Skipping Cells 9-14 all together thresholds
# the original image of Cell 8 (only rescaled to 0-255, the range the
# contrast settings and the threshold work in).

PROCESSING_STAGES = (  # (stage, the step that makes it), in pipeline order
    ('Original image', 'Cell 8 - original image'),
    ('Normalized image', 'Cell 9 - normalization (and depth correction)'),
    ('Zoomed image', 'Cell 10 - resampling to isotropic voxels'),
    ('Denoised image', 'Cell 11 - denoising (and haze removal)'),
    ('Adjusted image', 'Cell 12 - contrast and gamma'),
    ('Filtered image', 'Cell 13 - Gaussian smoothing'),
    ('Equalized image', 'Cell 14 - histogram equalization'),
)
_STAGE_NAMES = tuple(name for name, _ in PROCESSING_STAGES)
_STEPS_RUN_KEY = '_processing_steps_run'  # not an image: skipped by .items() users


def reset_processing_steps(im_final_stack):
    """Drop every processing stage after the original image (Cell 8).

    Called at the end of Cell 8, so a re-run that skips some of Cells 9-14
    never picks up a stage left over from an earlier run.
    """
    for name in _STAGE_NAMES[1:] + ('Threshold image',):
        im_final_stack.pop(name, None)
    im_final_stack[_STEPS_RUN_KEY] = []  # none of Cells 9-14 run yet
    gc.collect()
    return im_final_stack


def _stage_input(im_final_stack, stage):
    """``(name, image)`` of the most recent stage before *stage*.

    The original image is rescaled to 0-255 per channel (as Cell 9 does) when
    it is the input of a later step: the contrast settings and the threshold
    work in that range.
    """
    i = _STAGE_NAMES.index(stage) if stage in _STAGE_NAMES else len(_STAGE_NAMES)
    for name in reversed(_STAGE_NAMES[:i]):
        if name in im_final_stack:
            img = im_final_stack[name]
            if name == 'Original image' and stage != 'Normalized image':
                img = normalize_image_channels(img)
            return name, img
    raise KeyError(f"No image stage before '{stage}' in im_final_stack: run Cells 7-8 first.")


def _store_stage(im_final_stack, stage, image):
    """Store *stage* and drop every later stage (now outdated)."""
    i = _STAGE_NAMES.index(stage)
    for name in _STAGE_NAMES[i + 1:] + ('Threshold image',):
        im_final_stack.pop(name, None)
    im_final_stack[stage] = image
    steps = im_final_stack.setdefault(_STEPS_RUN_KEY, [])
    steps[:] = [s for s in steps if _STAGE_NAMES.index(s) < i] + [stage]
    return im_final_stack


def _fill_skipped_stages(im_final_stack):
    """Give each skipped stage from 'Zoomed image' on the image of the stage
    before it (same array, no copy). Returns the names of the filled stages."""
    filled = []
    for name in _STAGE_NAMES[2:]:  # 'Zoomed image' ... 'Equalized image'
        if name not in im_final_stack:
            _, im_final_stack[name] = _stage_input(im_final_stack, name)
            filled.append(name)
    im_final_stack.pop('Normalized image', None)  # not read past resampling
    return filled


def complete_skipped_steps(im_final_stack, zoom_factors, r_xyz):
    """Cell 15: complete the stages of the processing steps that were skipped.

    Every skipped step (Cells 9-14) passes its input through unchanged, so
    the threshold works on the last stage computed (the original image of
    Cell 8, rescaled to 0-255, when all of them were skipped). Without
    resampling (Cell 10), the image keeps its original voxel size: the zoom
    factors become 1 and the voxel sizes are the original ones.

    Parameters
    ----------
    im_final_stack : dict
    zoom_factors : list of 3 float
        [Z, Y, X] resampling factors of Cell 8.
    r_xyz : tuple of float
        Original voxel size (r_X, r_Y, r_Z) in um.

    Returns
    -------
    im_final_stack : dict
    zoom_factors : list of 3 float
        Unchanged, or [1, 1, 1] if the resampling was skipped.
    r_zX, r_zY, r_zZ : float
        Voxel size of the processed image in um.
    skipped_steps : list of str
        The steps that were skipped (for the report).
    """
    r_X, r_Y, r_Z = r_xyz
    steps_run = im_final_stack.get(_STEPS_RUN_KEY, [])
    skipped = [label for name, label in PROCESSING_STAGES[1:] if name not in steps_run]
    input_stage = next(n for n in reversed(_STAGE_NAMES) if n in im_final_stack)

    if 'Zoomed image' in im_final_stack:
        zoom_factors = list(zoom_factors)
        r_zX, r_zY, r_zZ = r_X / zoom_factors[2], r_Y / zoom_factors[1], r_Z / zoom_factors[0]
    else:
        zoom_factors = [1.0, 1.0, 1.0]
        r_zX, r_zY, r_zZ = r_X, r_Y, r_Z
    _fill_skipped_stages(im_final_stack)

    if skipped:
        print("Skipped processing steps (their input is passed on unchanged):")
        for label in skipped:
            print(f"  - {label}")
        print(f"The threshold works on the '{input_stage}'"
              + (" (rescaled to 0-255)." if input_stage == 'Original image' else "."))
        if 'Zoomed image' not in steps_run:
            print(f"No resampling: the image keeps its original voxel size "
                  f"[{r_zX:.4g}, {r_zY:.4g}, {r_zZ:.4g}] um (zoom factors set to 1).")
    else:
        print("All processing steps (Cells 9-14) were run.")
    return im_final_stack, zoom_factors, r_zX, r_zY, r_zZ, skipped


# ── Pipeline wrappers (Cell-level helpers) ─────────────────────────────────
# Each function handles one processing step end-to-end:
# updates im_final_stack in-place, plots the resulting histogram, and returns
# the updated stack (plus any derived values). Each step reads the most
# recent stage before it, so any of them can be skipped (see above).

def run_normalize(im_final_stack, stain_complete_df):
    """Normalize each channel to [0, 255] and display histogram."""
    _store_stage(im_final_stack, 'Normalized image',
                 normalize_image_channels(im_final_stack['Original image']))
    hist_plot(im_final_stack['Normalized image'], stain_complete_df)
    return im_final_stack


def run_resample(im_final_stack, stain_complete_df, zoom_factors, meta, edge_mode="constant"):
    """Resample to isotropic voxel spacing and display histogram
    (``edge_mode``: see `resample_to_isotropic`).

    Returns
    -------
    im_final_stack : dict
    r_zX, r_zY, r_zZ : float  —  isotropic voxel sizes in µm
    """
    source, img = _stage_input(im_final_stack, 'Zoomed image')
    zoomed, r_zX, r_zY, r_zZ = resample_to_isotropic(img, zoom_factors, meta=meta,
                                                     edge_mode=edge_mode)
    _store_stage(im_final_stack, 'Zoomed image', zoomed)
    # 'Normalized image' is not used past this point; drop it to free RAM
    # before the (larger, isotropic) downstream stages accumulate.
    if source == 'Normalized image':
        del im_final_stack['Normalized image']
    del img
    gc.collect()
    hist_plot(im_final_stack['Zoomed image'], stain_complete_df)
    return im_final_stack, r_zX, r_zY, r_zZ


def run_denoise(im_final_stack, stain_complete_df):
    """Apply median filter for noise removal and display histogram."""
    _, img = _stage_input(im_final_stack, 'Denoised image')
    _store_stage(im_final_stack, 'Denoised image', apply_median_denoise(img))
    hist_plot(im_final_stack['Denoised image'], stain_complete_df)
    return im_final_stack


def run_contrast_gamma(im_final_stack, stain_complete_df):
    """Apply per-channel contrast and gamma adjustments and display histogram."""
    _, img = _stage_input(im_final_stack, 'Adjusted image')
    _store_stage(im_final_stack, 'Adjusted image',
                 apply_contrast_gamma_per_channel(img, stain_complete_df))
    hist_plot(im_final_stack['Adjusted image'], stain_complete_df)
    return im_final_stack


def run_smooth(im_final_stack, stain_complete_df, sigma=0.5):
    """Apply Gaussian smoothing and display histogram.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel in voxels.
        Controls noise suppression vs. detail preservation.
        0 = no smoothing; 0.5 (default) = mild; 1–2 = stronger (noisier data).
    """
    _, img = _stage_input(im_final_stack, 'Filtered image')
    _store_stage(im_final_stack, 'Filtered image', apply_gaussian_smoothing(img, sigma))
    hist_plot(im_final_stack['Filtered image'], stain_complete_df)
    return im_final_stack


def run_equalize(im_final_stack, stain_complete_df, num_plateaus=2, plateau_factor=0.7):
    """Apply histogram equalization and display histogram.

    Parameters
    ----------
    num_plateaus : int
        Number of clipping plateaus applied during equalization.
        Higher values redistribute intensity more aggressively.
        2 (default) suits most data; increase to 3–4 for dim channels.
    plateau_factor : float
        Clipping height relative to the mean histogram bin count.
        Lower = stronger clipping. Typical range 0.5–1.0.
    """
    _, img = _stage_input(im_final_stack, 'Equalized image')
    _store_stage(im_final_stack, 'Equalized image',
                 apply_histogram_equalization_per_channel(img, num_plateaus, plateau_factor))
    hist_plot(im_final_stack['Equalized image'], stain_complete_df)
    return im_final_stack


def run_threshold(im_final_stack, stain_complete_df,
                  nuclei_diameter, cell_diameter,
                  r_zX, r_zY, r_zZ,
                  threshold_method='otsu',
                  global_threshold_weight=0.15,
                  sauvola_weight=0.60,
                  progress=None,
                  nuclear_conditions=None):
    """Apply combined thresholding and store the result in im_final_stack.

    Parameters
    ----------
    nuclear_conditions : list of str, optional
        Conditions thresholded like the NUCLEI channel (nucleus-sized local
        window and smallest island). Used by the LIVE/DEAD notebook for the
        markers set as nuclear; the NUCLEI channel always is.
    threshold_method : str
        Global thresholding algorithm used as one component of the
        combined threshold (the rest is Sauvola local + statistical background).
        - ``'otsu'``   : maximises inter-class variance (default; bimodal data).
        - ``'median'`` : threshold at median non-zero intensity (sparse signal).
        - ``'huang'``  : fuzzy entropy method (dim or diffuse signal).
    global_threshold_weight : float
        Weight (0-1) of the global threshold in the combined threshold
        (default 0.15).
    sauvola_weight : float
        Weight (0-1) of the Sauvola local threshold in the combined threshold
        (default 0.60). The remainder (``1 - global_threshold_weight -
        sauvola_weight``) is given to the statistical background threshold.
    """
    _fill_skipped_stages(im_final_stack)  # skipped Cells 9-14 pass their input on
    im_final_stack["Threshold image"] = apply_threshold_per_channel(
        im_final_stack["Equalized image"],
        stain_complete_df=stain_complete_df,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        r_zxyz=(r_zX, r_zY, r_zZ),
        threshold_method=threshold_method,
        global_threshold_weight=global_threshold_weight,
        sauvola_weight=sauvola_weight,
        progress=progress,
        nuclear_conditions=nuclear_conditions,
    )
    return im_final_stack


def normalize_image_channels(image_stack):
    """
    Normalize each channel to [0, 255] range independently.
    
    Parameters
    ----------
    image_stack : ndarray, shape (Z, Y, X, C)
        Multi-channel 3D image stack.
    
    Returns
    -------
    normalized : ndarray, shape (Z, Y, X, C), dtype uint8
        Normalized image with each channel stretched to [0, 255].
    """
    im_out = image_stack.copy()
    for c in range(image_stack.shape[3]):
        im_channel = image_stack[:, :, :, c].copy()
        normalized = (im_channel - im_channel.min()) / (im_channel.max() - im_channel.min() + 1e-8) * 255
        im_out[:, :, :, c] = np.clip(normalized, 0, 255).astype('uint8')
    return im_out


def apply_per_channel_filter(image_stack, filter_func):
    """
    Apply a filter function independently to each channel.
    
    Parameters
    ----------
    image_stack : ndarray, shape (Z, Y, X, C)
        Multi-channel 3D image stack.
    
    filter_func : callable
        Function that takes a single 3D array and returns filtered 3D array.
    
    Returns
    -------
    filtered : ndarray, shape (Z, Y, X, C)
        Filtered image stack, same dtype as input.
    """
    im_out = np.zeros_like(image_stack)
    for c in range(image_stack.shape[3]):
        im_out[:, :, :, c] = filter_func(image_stack[:, :, :, c])
    return im_out


def _filter_plane_or_volume(ch, filter_func):
    """Apply *filter_func* to a (Z, Y, X) channel, in 2D when Z == 1 (a 3D
    filter would mix the single plane with its padded copies)."""
    if ch.shape[0] == 1:
        return filter_func(ch[0])[np.newaxis]
    return filter_func(ch)


def apply_median_denoise(image_stack):
    """Apply median filter to each channel independently."""
    from skimage import filters
    return apply_per_channel_filter(
        image_stack, lambda ch: _filter_plane_or_volume(ch, filters.median)
    )


def apply_gaussian_smoothing(image_stack, sigma=0.5):
    """Apply Gaussian filter to each channel independently."""
    from skimage import filters
    return apply_per_channel_filter(
        image_stack,
        lambda ch: _filter_plane_or_volume(
            ch, lambda arr: filters.gaussian(arr, sigma, preserve_range=True)
        ),
    )


def apply_contrast_gamma_per_channel(image_stack, stain_df):
    """
    Apply contrast and gamma adjustments per channel using stain_df settings.
    
    Parameters
    ----------
    image_stack : ndarray, shape (Z, Y, X, C)
        Multi-channel 3D image stack.
    
    stain_df : DataFrame
        Must have index matching channels and columns 'Cont_min', 'Cont_max', 'Gamma'.
    
    Returns
    -------
    adjusted : ndarray, shape (Z, Y, X, C), dtype uint8
        Contrast and gamma adjusted image.
    """
    im_out = image_stack.copy()
    for c in range(image_stack.shape[3]):
        idx = stain_df.index[c]
        cont_min = stain_df.loc[idx, 'Cont_min']
        cont_max = stain_df.loc[idx, 'Cont_max']
        gamma = stain_df.loc[idx, 'Gamma']
        im_out[:, :, :, c] = napari_contrast_gamma_uint8(
            image_stack[:, :, :, c],
            (cont_min, cont_max),
            gamma
        )
    return im_out


def apply_histogram_equalization_per_channel(image_stack, num_plateaus=2, plateau_factor=0.7):
    """Apply histogram equalization to each channel independently."""
    im_out = np.zeros_like(image_stack)
    for c in range(image_stack.shape[3]):
        equalized = double_plateau_hist_equalization_nd(
            image_stack[:, :, :, c].astype('uint8'), 
            num_plateaus=num_plateaus, 
            plateau_factor=plateau_factor
        )
        im_ori = equalized.copy()
        normalized = (im_ori - im_ori.min()) / (im_ori.max() - im_ori.min() + 1e-8) * 255
        im_out[:, :, :, c] = np.clip(normalized, 0, 255).astype('uint8')
    return im_out


def _slice_background(plane, radius_px, percentile):
    """Smooth background of one 2D plane: a low percentile over a window of
    ``2 * radius_px``, computed on a block-averaged copy (fast) and
    interpolated back to the plane's size."""
    h, w = plane.shape
    d = max(1, int(radius_px // 8))
    ph, pw = -h % d, -w % d
    padded = np.pad(plane, ((0, ph), (0, pw)), mode='edge') if (ph or pw) else plane
    small = padded.reshape(padded.shape[0] // d, d, padded.shape[1] // d, d).mean(axis=(1, 3))
    win = max(3, int(round(2 * radius_px / d)) | 1)
    bg = ndi.percentile_filter(small, percentile, size=win, mode='reflect')
    bg = ndi.uniform_filter(bg, size=win, mode='reflect')
    if d > 1:
        bg = ndi.zoom(bg, d, order=1, mode='nearest')
    return bg[:h, :w]


def correct_z_background(image_stack, radius_um, voxel_size_zyx, percentile=25.0,
                         channels=None):
    """Remove the background haze slice by slice (v1.6.4).

    Out-of-focus light makes some Z slices (typically the first ones, near
    the coverslip) brighter than the rest, and the threshold then takes the
    haze around the nuclei for nucleus: the nuclei of those slices come out
    too large and merged. For every Z slice the local background - the
    ``percentile``-th percentile of the intensity within ``radius_um`` - is
    estimated and subtracted, and the stack's median background level is
    added back, so the intensity scale (and the contrast settings) stay
    comparable.

    Parameters
    ----------
    image_stack : ndarray (Z, Y, X, C)
    radius_um : float
        Size of the background window, in um (several nuclei across, e.g.
        15-40). 0 = off (input returned unchanged). A value larger than the
        image (e.g. 1000) uses one background level per slice.
    voxel_size_zyx : tuple
        Voxel size (Z, Y, X) of ``image_stack`` in um.
    percentile : float
        Percentile of the intensity taken as background (default 25; must
        stay below the fraction of the window that is background).
    channels : iterable of int, optional
        Channels to correct (default all).
    """
    if not radius_um or radius_um <= 0:
        return image_stack
    im_out = image_stack.copy()
    dtype = image_stack.dtype
    hi = 255 if dtype == np.uint8 else None
    radius_px = float(radius_um) / float(voxel_size_zyx[1])
    whole_slice = radius_px >= max(image_stack.shape[1:3])
    for c in (range(image_stack.shape[3]) if channels is None else channels):
        ch = image_stack[:, :, :, c].astype(np.float32)
        bg = np.empty_like(ch)
        for z in range(ch.shape[0]):
            if whole_slice:
                bg[z] = np.percentile(ch[z], percentile)
            else:
                bg[z] = _slice_background(ch[z], radius_px, percentile)
        ref = float(np.median(bg))
        out = ch - (bg - ref)
        out = np.clip(out, 0, hi) if hi is not None else np.maximum(out, 0)
        im_out[:, :, :, c] = out.astype(dtype)
    return im_out


def _slice_signal_levels(ch, signal_percentile, bg_percentile, max_voxels=400_000):
    """Per-Z-slice background (``bg_percentile``) and signal above it
    (``signal_percentile`` - background) of a (Z, Y, X) channel. Large slices
    are subsampled on a regular grid (percentiles barely change)."""
    z_n = ch.shape[0]
    step = max(1, int(np.sqrt(ch.shape[1] * ch.shape[2] / max_voxels)))
    bg = np.empty(z_n, dtype=np.float64)
    sig = np.empty(z_n, dtype=np.float64)
    for z in range(z_n):
        sl = np.asarray(ch[z, ::step, ::step], dtype=np.float32)
        lo, hi = np.percentile(sl, [bg_percentile, signal_percentile])
        bg[z], sig[z] = lo, hi - lo
    return bg, sig


def _robust_depth_fit(z, y, model, n_iter=5):
    """Weighted least-squares line through (z, y) with Tukey bisquare
    re-weighting, so slices off the trend (the sample's first/last, nearly
    empty slices; a slice crossing a cluster) do not pull the fit.
    ``model`` 'exp' fits log(y), 'linear' fits y. Returns the fitted y at z."""
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 3:
        return None
    zz = z[ok].astype(float)
    t = np.log(y[ok]) if model == "exp" else y[ok].astype(float)
    w = np.ones_like(t)
    coef = np.polyfit(zz, t, 1)
    for _ in range(n_iter):
        r = t - np.polyval(coef, zz)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
        coef = np.polyfit(zz, t, 1, w=np.sqrt(w))
    fit = np.polyval(coef, z.astype(float))
    return np.exp(fit) if model == "exp" else fit


def estimate_z_attenuation(channel_zyx, model="exp", signal_percentile=99.9,
                           bg_percentile=25.0, max_gain=5.0, strength=1.0):
    """Depth profile of the signal of one (Z, Y, X) channel (v1.6.4).

    In confocal stacks the excitation and the emitted light are attenuated
    on their way through the sample, so the nuclei of the deep slices are
    dimmer above the same background (in the GT1 and GT2/GT4 ground truths
    the signal above background falls ~3x from the first to the last
    slice), and a threshold set on the whole stack loses or shrinks them.

    For every slice the background (``bg_percentile``) and the signal above
    it (``signal_percentile`` - background; a very high percentile, so that
    it measures the brightest nuclei of the slice whether the slice holds
    few or many of them) are measured, and a smooth trend is fitted through
    the signal along Z - robustly, so isolated slices off the trend are
    ignored. No ground truth is used.

    Parameters
    ----------
    model : {'exp', 'linear'}
        'exp' (Beer-Lambert-like attenuation, default) or 'linear'.
    max_gain : float
        Largest gain given to a slice (protects nearly empty slices).
    strength : float
        Fraction of the correction applied: gain = (ref / fit) ** strength
        (1 = full, 0 = none).

    Returns
    -------
    dict with per-slice arrays ``bg``, ``signal``, ``fit``, ``gain`` and the
    scalars ``model``, ``ratio_first_last`` (fitted first / last slice) and
    ``decay_per_slice`` (fractional signal loss per slice).
    """
    z_n = channel_zyx.shape[0]
    bg, sig = _slice_signal_levels(channel_zyx, signal_percentile, bg_percentile)
    z = np.arange(z_n)
    fit = _robust_depth_fit(z, sig, model) if z_n >= 3 else None
    if fit is None:
        fit = np.full(z_n, np.nanmax(sig) if np.isfinite(sig).any() else 1.0)
    floor = max(np.nanmax(fit) / max_gain, 1e-6)
    fit_c = np.maximum(fit, floor)
    gain = (np.nanmax(fit_c) / fit_c) ** float(strength)
    gain = np.clip(gain, 1.0, max_gain)
    ratio = float(fit_c[0] / fit_c[-1]) if z_n > 1 else 1.0
    decay = float(1 - ratio ** (-1.0 / (z_n - 1))) if z_n > 1 else 0.0
    return {"model": model, "bg": bg, "signal": sig, "fit": fit_c, "gain": gain,
            "ratio_first_last": ratio, "decay_per_slice": decay,
            "strength": float(strength), "signal_percentile": signal_percentile,
            "bg_percentile": bg_percentile}


def correct_z_attenuation(image_stack, model="exp", strength=1.0, channels=None,
                          signal_percentile=99.9, bg_percentile=25.0, max_gain=5.0,
                          inplace=False):
    """Compensate the loss of signal with depth, slice by slice (v1.6.4).

    For every channel the depth profile is estimated
    (:func:`estimate_z_attenuation`) and every Z slice is rescaled around its
    own background so that its signal matches the brightest slice::

        out[z] = bg[z] + (in[z] - bg[z]) * gain[z]

    The background is left where it is (out-of-focus haze is removed by
    :func:`correct_z_background`), and the result is clipped to the
    channel's original range, so the 0-255 normalization - and the contrast
    settings chosen on the original image - stay valid. Run it on the
    original image, before ``run_normalize`` (Cell 9): there the deep slices
    still have their full bit depth.

    Parameters
    ----------
    image_stack : ndarray (Z, Y, X, C)
    model : {'exp', 'linear', 'off'}
        Shape of the depth trend; 'off' (or None) returns the input unchanged.
    strength : float
        Fraction of the correction (0-1, or above 1 to over-correct).
    channels : iterable of int, optional
        Channels to correct (default all).
    inplace : bool
        Correct ``image_stack`` itself (saves a copy of a large stack).

    Returns
    -------
    corrected : ndarray (Z, Y, X, C), same dtype as the input
    profiles : dict {channel index: profile dict}
    """
    if not model or model == "off" or image_stack.shape[0] < 3 or strength == 0:
        return image_stack, {}
    if model not in ("exp", "linear"):
        raise ValueError(f"model must be 'exp', 'linear' or 'off', not {model!r}")
    out = image_stack if inplace else image_stack.copy()
    dtype = image_stack.dtype
    is_int = np.issubdtype(dtype, np.integer)
    profiles = {}
    for c in (range(image_stack.shape[3]) if channels is None else channels):
        ch = image_stack[:, :, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        prof = estimate_z_attenuation(ch, model=model, signal_percentile=signal_percentile,
                                      bg_percentile=bg_percentile, max_gain=max_gain,
                                      strength=strength)
        for z in range(ch.shape[0]):
            g = prof["gain"][z]
            if g == 1.0:
                continue
            sl = ch[z].astype(np.float32)
            sl = prof["bg"][z] + (sl - prof["bg"][z]) * g
            np.clip(sl, lo, hi, out=sl)
            out[z, :, :, c] = np.rint(sl).astype(dtype) if is_int else sl.astype(dtype)
        profiles[c] = prof
    return out, profiles


def plot_z_attenuation(profiles, names=None, voxel_z=None, show=True):
    """Diagnostic plot of :func:`correct_z_attenuation`: for every corrected
    channel the measured signal per slice and the fitted depth trend (top),
    and the gain applied to each slice (bottom)."""
    import matplotlib.pyplot as plt
    if not profiles:
        return None
    keys = list(profiles)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 5.2),
                             sharex="col", squeeze=False)
    for j, c in enumerate(keys):
        p = profiles[c]
        z = np.arange(len(p["signal"])) * (voxel_z or 1.0)
        name = names[j] if names is not None and j < len(names) else f"channel {c}"
        ax = axes[0, j]
        ax.plot(z, p["signal"], "o", ms=3, color="#9aa0a6", label="measured")
        ax.plot(z, p["fit"], "-", lw=2, color="#2a6fdb", label=f"{p['model']} fit")
        ax.set_title(f"{name}: first/last = {p['ratio_first_last']:.2f}x", fontsize=10)
        ax.set_ylabel("signal above background")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        ax = axes[1, j]
        ax.plot(z, p["gain"], "-", lw=2, color="#2a6fdb")
        ax.set_ylabel(f"gain (strength {p['strength']:g})")
        ax.set_xlabel("Z [um]" if voxel_z else "Z slice")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def trim_flared_slices(labels, factor):
    """Trim the Z slices where a nucleus flares out (v1.6.4).

    Blurred nuclei, in the first/last slices especially, can spread far
    wider in a few slices than in the rest of the nucleus. For every label,
    starting from its slice of typical (median) cross-section and moving up
    and down, a slice whose area is above ``factor`` x that median is cut
    back to the previous slice's footprint grown by one pixel. ``factor``
    0 = off.
    """
    if not factor or factor <= 0 or labels.ndim != 3 or labels.shape[0] < 3:
        return labels
    out = labels.copy()
    n_trimmed = 0
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        area = m.sum(axis=(1, 2))
        zs = np.nonzero(area)[0]
        if len(zs) < 3:
            continue
        ref = float(np.median(area[zs]))
        core = int(zs[np.argmin(np.abs(area[zs] - ref))])
        new = m.copy()
        for step in (1, -1):
            prev = new[core]
            z = core + step
            while 0 <= z < m.shape[0]:
                if area[z] > factor * ref:
                    new[z] = m[z] & ndi.binary_dilation(prev)
                    n_trimmed += 1
                if new[z].any():
                    prev = new[z]
                z += step
        out[sl][m & ~new] = 0
    if n_trimmed:
        print(f"Flared slices trimmed (area > {factor:g}x the nucleus median): {n_trimmed}")
    return out


def resample_to_isotropic(image_stack, zoom_factors, meta=None, edge_mode="constant"):
    """
    Resample image stack to isotropic voxel spacing.

    v1.6.41, ``edge_mode``: how `scipy.ndimage.zoom` samples past the last
    voxel. ``"constant"`` (default, as in v1.6.4) fills with 0, and
    because the last output plane maps a rounding error past the last input
    plane, the last Z slice (and the last row / column) came out empty
    whenever the stack was shrunk: the nuclei at the bottom of the stack were
    lost. ``"nearest"`` (the optimal settings v1.6.41) repeats the edge voxel.
    
    Parameters
    ----------
    image_stack : ndarray, shape (Z, Y, X, C)
        Original image stack.
    
    zoom_factors : list of 3 floats
        Scaling factors [Z, Y, X] for resampling.
    
    meta : optional
        Metadata object with physical_pixel_sizes. If provided, returns updated voxel sizes.
    
    Returns
    -------
    resampled : ndarray
        Resampled image stack.
    
    r_zX, r_zY, r_zZ : float (if meta provided)
        Updated physical voxel sizes in micrometers.
    """
    from scipy.ndimage import zoom as scipy_zoom
    
    new_shape = (
        round(image_stack.shape[0] * zoom_factors[0]),
        round(image_stack.shape[1] * zoom_factors[1]),
        round(image_stack.shape[2] * zoom_factors[2]),
        image_stack.shape[3]
    )
    im_out = np.zeros(new_shape, dtype=image_stack.dtype)
    
    for c in range(image_stack.shape[3]):
        resampled = scipy_zoom(image_stack[:, :, :, c], zoom=zoom_factors, order=1,
                               mode=edge_mode)
        resampled = resampled - np.min(resampled)
        normalized = (resampled - resampled.min()) / (resampled.max() - resampled.min() + 1e-8) * 255
        im_out[:, :, :, c] = np.clip(normalized, 0, 255).astype('uint8')
    
    if meta is not None:
        r_zX = meta.physical_pixel_sizes.X / zoom_factors[2]
        r_zY = meta.physical_pixel_sizes.Y / zoom_factors[1]
        r_zZ = meta.physical_pixel_sizes.Z / zoom_factors[0]
        return im_out, r_zX, r_zY, r_zZ
    
    return im_out


# Above this size, the file's full pixel array is read lazily via dask, with
# only the requested ROI ever computed into memory, instead of reading the
# whole file eagerly first. Kept conservative since later pipeline stages
# (Cells 9-15) each hold their own full-size copy of the array alongside it.
_LAZY_LOAD_THRESHOLD_BYTES = 2 * 1024 ** 3  # 2 GiB


def _estimate_full_image_bytes(meta):
    """Estimate the in-memory size (bytes) of the file's full pixel array."""
    n_voxels = int(np.prod(meta.shape))
    itemsize = np.dtype(getattr(meta, 'dtype', np.uint16)).itemsize
    return n_voxels * itemsize


def _lazy_image_data(meta):
    """``meta.get_image_dask_data("ZYXC")``, checked by reading one voxel.

    aicsimageio reads TIFF files lazily through zarr 2, which fails with
    ``ImportError: cannot import name 'cbuffer_sizes' from 'numcodecs.blosc'``
    when a newer numcodecs is installed. In that case the whole image is read
    directly instead (returned as a numpy array) and a warning is printed.
    """
    try:
        lazy = meta.get_image_dask_data("ZYXC")
        lazy[:1, :1, :1, :1].compute()
        return lazy
    except ImportError as exc:
        print(f"[lazy loading] Not available in this environment ({exc}); "
              "reading the whole image into memory instead.")
        return meta.get_image_data("ZYXC", T=0)


def extract_roi_from_metadata(meta, roi_coords):
    """
    Extract a region of interest from a microscopy file.

    Automatically chooses how to read the data: files whose full pixel array
    would exceed ``_LAZY_LOAD_THRESHOLD_BYTES`` are read lazily via dask, so
    only the requested ROI is ever computed into memory; smaller files are
    read directly since the dask overhead isn't worth it.

    Parameters
    ----------
    meta : AICSImage
        Metadata object from bioio.AICSImage.

    roi_coords : list of 6 ints
        [x0, x1, y0, y1, z0, z1] coordinates. Use 0 to keep full range.

    Returns
    -------
    image : ndarray, shape (Z, Y, X, C)
        Extracted image in ZYX order.

    roi_used : list
        Actual ROI coordinates used (with 0s replaced by full ranges).

    used_lazy_loading : bool
        Whether the dask/chunked read path was used.
    """
    x0, x1, y0, y1, z0, z1 = roi_coords

    # Replace 0s with full range
    if x1 == 0:
        x1 = meta.shape[4]
    if y1 == 0:
        y1 = meta.shape[3]
    if z1 == 0:
        z1 = meta.shape[2]

    used_lazy_loading = _estimate_full_image_bytes(meta) > _LAZY_LOAD_THRESHOLD_BYTES
    lazy = _lazy_image_data(meta) if used_lazy_loading else None
    if lazy is not None and not isinstance(lazy, np.ndarray):
        sub = lazy[z0:z1, y0:y1, x0:x1, :]
        image = sub.compute()
    else:
        used_lazy_loading = False
        full = lazy if lazy is not None else meta.get_image_data("ZYXC", T=0)
        image = full[z0:z1, y0:y1, x0:x1, :]

    return image, [x0, x1, y0, y1, z0, z1], used_lazy_loading


def fix_image_axes_order(original_image):
    """
    Fix image axes order and shape for multi-channel 3D stacks.
    
    Ensures image has shape (Z, Y, X, C) and channels are last.
    
    Parameters
    ----------
    original_image : ndarray
        Image array of shape (Z, Y, X, C[, T]) or similar.
    
    Returns
    -------
    fixed : ndarray
        Image with standard shape (Z, Y, X, C).
    """
    orig = original_image.copy()
    shape = orig.shape
    
    # Drop singleton time axis if present (common order Z,Y,X,C,T)
    if len(shape) == 5 and shape[4] == 1:
        orig = orig[..., 0]
    
    # Ensure channels are last
    if len(orig.shape) == 4:
        # If last axis looks large (likely X), detect small axis as channel
        if orig.shape[-1] > 50:
            chan_axis = next((i for i, s in enumerate(orig.shape) if s < 50), None)
            if chan_axis is not None and chan_axis != 3:
                orig = np.moveaxis(orig, chan_axis, -1)
    
    return orig


def prepare_image_stack(img, nuclei_diameter, cell_diameter):
    """Compute geometry parameters and build the initial image stack.

    Derives nuclei/cell radii and volumes from diameter inputs, casts the
    (already ROI-cropped) image to float32, and fixes axes order so the
    result is always (Z, Y, X, C).

    Parameters
    ----------
    img : ndarray
        ROI-cropped image array already loaded into memory, as returned by
        ``initialize_dataset``/``load_image_and_metadata`` (shape Z, Y, X, C
        or similar).
    nuclei_diameter : float
        Approximate nucleus diameter in µm.
    cell_diameter : float
        Approximate whole-cell diameter in µm.

    Returns
    -------
    im_final_stack : dict
        {'Original image': ndarray (Z, Y, X, C)}
    nuclei_radius : float
    cell_radius : float
    nuclei_volume : float
    cell_volume : float
    """
    nuclei_radius = nuclei_diameter * 0.5
    cell_radius = cell_diameter * 0.5
    nuclei_volume = np.ceil(4.0 * (nuclei_radius ** 3.0) * np.pi / 3.0)
    cell_volume = np.ceil(4.0 * (cell_radius ** 3.0) * np.pi / 3.0)

    orig = fix_image_axes_order(img.astype('float32'))

    print("Image stack ready.")
    return {'Original image': orig}, nuclei_radius, cell_radius, nuclei_volume, cell_volume


def _napari_colormap(color):
    """Map a stain colour to a napari colormap name.

    Stain colours are stored uppercase; napari does not recognise "GREEN" as its
    built-in colormap and falls back to the darker CSS green (#008000).
    """
    if isinstance(color, str) and color.upper() == "GREEN":
        return "green"
    return color


def build_stain_dataframe(stain_dict: dict, file_meta: dict) -> "pd.DataFrame":
    """Build and sort the staining metadata DataFrame.

    Normalises all string keys/values in *stain_dict* to uppercase, constructs
    a DataFrame with columns ``['Marker', 'Laser', 'Color']``, then sorts rows
    to match the channel order reported by *file_meta*.

    A ``'Channel_index'`` column is kept (rather than dropped) holding each
    condition's real position in *file_meta*'s channel list, so callers can
    align the raw image's channel axis to this table's row order — including
    dropping any real channel that has no entry in *stain_dict*. If a
    ``Laser`` value doesn't match any channel name in *file_meta*, that is a
    typo/misconfiguration (not a "missing channel") and raises rather than
    silently mis-sorting the row.

    Parameters
    ----------
    stain_dict : dict
        User-defined mapping of condition name to [marker, laser/channel, color].
    file_meta : dict
        Output of ``read_file_metadata``; must contain a ``'channels'`` list.

    Returns
    -------
    stain_df : pd.DataFrame
        Sorted staining table indexed by condition name, with a
        ``'Channel_index'`` column giving each row's real channel position.
    """
    norm = {
        k.upper(): [item.upper() if isinstance(item, str) else item for item in v]
        for k, v in stain_dict.items()
    }
    stain_df = pd.DataFrame.from_dict(norm, orient='index', columns=['Marker', 'Laser', 'Color'])
    stain_df.index.name = 'Condition'

    laser_order = file_meta.get("channels") or []
    order_map = {name.strip().upper(): i for i, name in enumerate(laser_order)}
    stain_df['Channel_index'] = stain_df['Laser'].map(order_map)

    unmatched = stain_df[stain_df['Channel_index'].isna()]
    if not unmatched.empty:
        raise ValueError(
            "stain_dict has Laser/channel name(s) that don't match any channel "
            f"reported by the file: {dict(zip(unmatched.index, unmatched['Laser']))}. "
            f"Available channels: {laser_order}"
        )

    stain_df['Channel_index'] = stain_df['Channel_index'].astype(int)
    stain_df = stain_df.sort_values('Channel_index')

    if 'NUCLEI' not in stain_df.index:
        print('[build_stain_dataframe] Warning: no NUCLEI condition found!')

    return stain_df


def view_original_channels(im_final_stack: dict, stain_df: "pd.DataFrame",
                            napari_module, r_xyz=None, progress=None) -> object:
    """Open a napari viewer and add each channel of the original image.

    Each channel is normalised to uint8 [0, 255] before display.

    Parameters
    ----------
    im_final_stack : dict
        Must contain ``'Original image'`` with shape (Z, Y, X, C).
    stain_df : pd.DataFrame
        Staining table from ``build_stain_dataframe``.
    napari_module : module
        The imported ``napari`` module.
    r_xyz : tuple of float (r_X, r_Y, r_Z), optional
        Physical voxel sizes in micrometers. Without this, napari treats
        every voxel as a 1x1x1 cube, which stretches or squashes the
        displayed volume along Z whenever the Z step differs from the
        XY pixel size.
    progress : callable, optional
        A ``tqdm``-compatible progress wrapper (e.g. ``tqdm``).

    Returns
    -------
    viewer : napari.Viewer
    """
    scale = (r_xyz[2], r_xyz[1], r_xyz[0]) if r_xyz is not None else None

    im_in = im_final_stack['Original image'].copy()
    _close_all_napari_viewers(napari_module)
    viewer = napari_module.Viewer(title="Original Image Channels")

    for c, c_name in _progress_iter(
        enumerate(stain_df['Marker']),
        progress,
        desc='Step 06 - Visualize Channels',
        total=len(stain_df['Marker']),
        leave=False,
    ):
        im_channel = im_in[:, :, :, c]
        ch_min, ch_max = im_channel.min(), im_channel.max()
        im_8b = ((im_channel - ch_min) / (ch_max - ch_min + 1e-8) * 255).clip(0, 255).astype('uint8')
        viewer.add_image(
            im_8b,
            name=f"{stain_df.index[c]} ({c_name})",
            colormap=_napari_colormap(stain_df['Color'].iloc[c]),
            blending='additive',
            scale=scale,
        )

    viewer.scale_bar.visible = True
    for layer in viewer.layers:
        layer.units = ('um', 'um', 'um')
    return viewer


def _threshold_sauvola_chunked(arr, window_size, memory_budget_bytes=1_000_000_000):
    """Memory-scoped equivalent of ``skimage.filters.threshold_sauvola`` for
    3D arrays.

    ``threshold_sauvola`` internally pads the whole volume and builds a
    float64 integral image over it, which can need several times the input
    array's size in one allocation (this is what raised ``MemoryError`` on
    large single-channel volumes). This processes the volume in Z-slabs,
    each padded with a halo of real neighbouring slices (matching
    ``skimage``'s own ``window_size // 2 + 1`` / ``window_size // 2`` pad
    widths) so every slab's result is identical to running the full-volume
    computation, but only one slab's padded copy/integral image is resident
    in memory at a time.

    Parameters
    ----------
    arr : ndarray, shape (Z, Y, X)
    window_size : int
        Odd window size, as passed to ``threshold_sauvola``.
    memory_budget_bytes : int
        Approximate cap on the float64 integral image for a single slab;
        the Z-chunk size is derived from this and the array's Y/X extent.

    Returns
    -------
    ndarray, same shape as ``arr``
    """
    from skimage.filters import threshold_sauvola

    z_dim, y_dim, x_dim = arr.shape
    if z_dim == 1:
        # 2D image: a 2D window over the single plane.
        return threshold_sauvola(arr[0], window_size=window_size)[np.newaxis]
    half_before = window_size // 2 + 1
    half_after = window_size // 2

    per_slice_bytes = 8 * y_dim * x_dim  # dominant cost: float64 integral image
    chunk_z = max(1, int(memory_budget_bytes / per_slice_bytes))

    if chunk_z >= z_dim:
        return threshold_sauvola(arr, window_size=window_size)

    out = np.empty(arr.shape, dtype=np.result_type(arr.dtype, np.float32))
    for start in range(0, z_dim, chunk_z):
        stop = min(start + chunk_z, z_dim)
        pad_start = max(0, start - half_before)
        pad_stop = min(z_dim, stop + half_after)
        slab_thr = threshold_sauvola(arr[pad_start:pad_stop], window_size=window_size)
        out[start:stop] = slab_thr[start - pad_start: start - pad_start + (stop - start)]
        del slab_thr
    return out


def _compute_channel_threshold_stats(
    arr,
    is_nuclei,
    nuclei_size,
    cell_size,
    threshold_method,
    marker_name="",
    verbose=True,
    global_weight=0.15,
    sauvola_weight=0.60,
):
    """Compute the global, statistical, and combined threshold values for one channel.

    Shared by `apply_threshold_per_channel` (Cell 15, which also builds the
    binary mask) and `run_equalize`'s threshold preview (Cell 14, which only
    needs the scalar values to draw histogram markers).

    Parameters
    ----------
    global_weight : float
        Weight of the global threshold in the combined threshold (default 0.15).
    sauvola_weight : float
        Weight of the Sauvola local threshold in the combined threshold
        (default 0.60). The remaining weight (``1 - global_weight -
        sauvola_weight``) is assigned to the statistical background threshold.

    Returns
    -------
    global_thr_value : float
    final_thr : ndarray or float
        Combined threshold map (Sauvola + statistical + global), after the
        oversaturation floor correction.
    statistical_thr : float
    gain_ass : float
    bg_mean : float
    """
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr)

    # Compute global threshold value based on chosen method
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    stretched = rescaler.Execute(img)

    if threshold_method == 'otsu':
        th_filter = sitk.OtsuThresholdImageFilter()
        th_filter.Execute(stretched)
        th_filter.Execute(img)
        global_thr_value = th_filter.GetThreshold()
    elif threshold_method == 'median':
        non_zero_tmp = arr[arr > 0]
        global_thr_value = float(np.median(non_zero_tmp)) if non_zero_tmp.size > 0 else 0.0
    elif threshold_method == 'huang':
        th_filter = sitk.HuangThresholdImageFilter()
        th_filter.Execute(stretched)
        th_filter.Execute(img)
        global_thr_value = th_filter.GetThreshold()
    else:
        raise ValueError(f"threshold_method must be one of ('otsu', 'median', 'huang'), got '{threshold_method}'")

    window_size = nuclei_size if is_nuclei else 4 * cell_size + 1
    if int(window_size) % 2 == 0:
        window_size += 1

    # Sauvola threshold map (processed in Z-slabs to cap peak memory — see
    # _threshold_sauvola_chunked docstring)
    sauvola_value = _threshold_sauvola_chunked(arr, window_size=int(window_size))

    # Global statistical background, excluding zeros
    non_zero = arr[arr > 0]
    if non_zero.size > 0:
        hist, bins = np.histogram(non_zero, bins=256, range=(0, non_zero.max()))
        mode_bin = bins[np.argmax(hist)]
        bg_mask = (arr >= mode_bin - 5) & (arr <= mode_bin + 5) & (arr > 0)
        gain_tot = 6.0
        gain_ass = gain_tot * (255.0 - 4.0 * mode_bin) / 255.0
        bg_vals = arr[bg_mask]
        if bg_vals.size < 50:
            p10 = np.percentile(non_zero, 10)
            bg_vals = non_zero[non_zero <= p10]
    else:
        bg_vals = arr
        gain_ass = 6.0

    bg_mean = bg_vals.mean()

    bg_mean_z = arr.mean()
    bg_std_z = arr.std() + 1e-6
    statistical_thr = bg_mean_z + 3.0 * bg_std_z

    # Clip Sauvola to avoid over-thresholding large/bright cells
    max_sauvola = bg_mean_z + 2.0 * bg_std_z
    sauvola_clipped = np.minimum(sauvola_value, max_sauvola)

    # Combined threshold map
    statistical_weight = 1.0 - global_weight - sauvola_weight
    final_thr = (
        sauvola_weight * sauvola_clipped +
        statistical_weight * statistical_thr +
        global_weight * global_thr_value
    )

    # Oversaturation correction: when many voxels are near the max
    # intensity, the halo around bright objects is also bright and passes
    # normal thresholds.  Detect this and raise the threshold floor so
    # only genuinely bright tissue is kept.
    if non_zero.size > 0:
        max_val = float(arr.max())
        saturation_level = max_val * 0.97
        saturated_count = int(np.sum(non_zero >= saturation_level))
        saturated_fraction = saturated_count / non_zero.size
        if saturated_fraction > 0.01:
            sat_pct = min(95.0, 75.0 + saturated_fraction * 500.0)
            saturation_floor = float(np.percentile(non_zero, sat_pct))
            final_thr = np.maximum(final_thr, saturation_floor)
            if verbose:
                print(
                    f"  [{marker_name}] oversaturation detected "
                    f"({saturated_fraction * 100:.1f}% near-max), "
                    f"threshold floor raised to P{sat_pct:.0f}={saturation_floor:.1f}"
                )

    return global_thr_value, final_thr, statistical_thr, gain_ass, bg_mean


def apply_threshold_per_channel(
    image_stack,
    stain_complete_df,
    nuclei_diameter,
    cell_diameter,
    r_zxyz,
    threshold_method='otsu',
    global_threshold_weight=0.15,
    sauvola_weight=0.60,
    progress=None,
    nuclear_conditions=None,
):
    """
    Threshold each channel using a combined global, Sauvola, and statistical
    background approach with gain-based pixel rescue.

    Parameters
    ----------
    image_stack : ndarray, shape (Z, Y, X, C)
        Equalized image stack (uint8).
    stain_complete_df : DataFrame
        Staining metadata; index is used to distinguish 'NUCLEI' from other channels.
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    cell_diameter : float
        Approximate cell diameter in micrometers.
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in micrometers per pixel.
    threshold_method : str, optional
        Global threshold algorithm used as one component of the combined threshold.
        One of 'otsu' (default), 'median', or 'huang'. The Sauvola local threshold
        and statistical background component are always applied regardless of this choice.
    global_threshold_weight : float, optional
        Weight (0-1) given to the global threshold in the combined threshold
        (default 0.15).
    sauvola_weight : float, optional
        Weight (0-1) given to the Sauvola local threshold in the combined
        threshold (default 0.60). The remaining weight
        (``1 - global_threshold_weight - sauvola_weight``) goes to the
        statistical background threshold. Raising ``sauvola_weight`` (more
        local adaptivity) helps recover nuclei buried inside bright
        aggregates, where the global/statistical components tend to push the
        threshold too high.
    progress : callable, optional
        Progress wrapper (e.g. tqdm).
    nuclear_conditions : list of str, optional
        Extra conditions thresholded like the NUCLEI channel (LIVE/DEAD
        markers set as nuclear).

    Returns
    -------
    im_out : ndarray, shape (Z, Y, X, C)
        Binary thresholded image stack (same dtype as input).
    """
    _valid_methods = ('otsu', 'median', 'huang')
    if threshold_method not in _valid_methods:
        raise ValueError(f"threshold_method must be one of {_valid_methods}, got '{threshold_method}'")

    if global_threshold_weight < 0 or sauvola_weight < 0 or global_threshold_weight + sauvola_weight > 1.0:
        raise ValueError(
            "global_threshold_weight and sauvola_weight must each be >= 0 and "
            f"sum to <= 1.0, got global_threshold_weight={global_threshold_weight}, "
            f"sauvola_weight={sauvola_weight}"
        )

    r_zX, r_zY, r_zZ = r_zxyz
    im_out = image_stack.copy()
    nuclei_size = int(nuclei_diameter / (np.mean([r_zX, r_zY])))
    cell_size = int(cell_diameter / (np.mean([r_zX, r_zY])))
    # One nucleus in voxels, the reference of the small-island filter: a
    # sphere for a 3D image (islands are counted in 3D), a disk for 2D.
    if image_stack.shape[0] > 1:
        nucleus_vox = _sphere_volume_um3(nuclei_diameter) / (r_zX * r_zY * r_zZ)
    else:
        nucleus_vox = np.pi * (nuclei_diameter / 2.0) ** 2 / (r_zX * r_zY)

    statistical_weight = 1.0 - global_threshold_weight - sauvola_weight
    global_thresholds = []
    combined_thresholds = []
    nuclear_conditions = {"NUCLEI"} | {str(c).upper() for c in (nuclear_conditions or [])}

    for c in _progress_iter(range(image_stack.shape[3]), progress, desc='Step 15 - Threshold Channels'):
        marker_name = stain_complete_df.loc[stain_complete_df.index[c], 'Marker']
        is_nuclei = str(stain_complete_df.index[c]).upper() in nuclear_conditions
        arr = image_stack[:, :, :, c]

        global_thr_value, final_thr, statistical_thr, gain_ass, bg_mean = _compute_channel_threshold_stats(
            arr, is_nuclei, nuclei_size, cell_size, threshold_method,
            marker_name=marker_name, verbose=True,
            global_weight=global_threshold_weight, sauvola_weight=sauvola_weight,
        )
        combined_thr_value = float(np.mean(final_thr))
        global_thresholds.append(global_thr_value)
        combined_thresholds.append(combined_thr_value)

        print(
            f"  [{marker_name}] threshold chosen: global ({threshold_method}, "
            f"w={global_threshold_weight:.2f}) = {global_thr_value:.1f}, combined "
            f"(Sauvola w={sauvola_weight:.2f} + statistical w={statistical_weight:.2f} "
            f"+ global) = {combined_thr_value:.1f}"
        )

        # Gain-based primary mask and rescue
        gain = arr / (bg_mean + 1e-6)
        primary = (arr > final_thr) & (gain > gain_ass)
        rescue = (gain > (gain_ass + 3.0)) & (arr > statistical_thr)
        arrayseg = primary | rescue

        min_size = np.ceil((0.4 if is_nuclei else 0.8) * nucleus_vox)

        im_out[:, :, :, c] = remove_small_islands(
            arrayseg, min_size, progress=progress,
            desc=f'Step 15 - Remove Small Islands [{marker_name}]',
        )

    hist_plot_with_thresholds(image_stack, stain_complete_df, global_thresholds, combined_thresholds)

    return im_out


# ── Cytoplasm from the stains (used by Cell 18) ───────────────────────────
# The cytoplasm is the union of the thresholded intracellular stains and the
# filled membrane stains (the run_threshold masks of Cell 15), with no growth
# beyond the stained voxels.

def _fill_small_holes_per_slice(mask, max_hole_area):
    """Fill XY holes smaller than *max_hole_area*, without closing larger
    gaps enclosed by a ring of objects."""
    out = mask.copy()
    for z in range(out.shape[0]):
        if not out[z].any():
            continue
        holes = ndi.binary_fill_holes(out[z]) & ~out[z]
        hole_labels, n = ndi.label(holes)
        if n == 0:
            continue
        is_small = np.bincount(hole_labels.ravel()) < max_hole_area
        is_small[0] = False
        out[z] |= is_small[hole_labels]
    return out


def _fill_membrane_rings(membrane_mask, max_cell_area, close_radius=2):
    """Turn a membrane mask (rings/shells around each cell) into the region it
    encloses, membrane included.

    Small gaps in the ring are closed first; then cavities closed in 3D are
    filled, and in each XY slice every enclosed hole up to *max_cell_area*
    voxels (a ring is often open at the top/bottom of the cell in Z, so a 3D
    fill alone misses it). Larger enclosed areas - e.g. empty space ringed by
    several cells - are left empty.
    """
    from skimage.morphology import ball, disk

    if membrane_mask.shape[0] == 1:
        # 2D image: close in the plane (a 3D ball would erode the single
        # plane away against the empty planes around it).
        closed = ndi.binary_closing(membrane_mask[0], structure=disk(close_radius))
        filled = _fill_small_holes_per_slice(closed[np.newaxis], max_cell_area)
        return filled | membrane_mask
    filled = ndi.binary_closing(membrane_mask, structure=ball(close_radius))
    filled = ndi.binary_fill_holes(filled)
    filled = _fill_small_holes_per_slice(filled, max_cell_area)
    return filled | membrane_mask


def _cytoplasm_stain_channels(stain_complete_df, n_channels, cyto_markers, membrane_markers):
    """Channel indices of the cytoplasm stains, as ``(intracellular,
    membrane)``. Intracellular = the CYTOPLASM channel and the markers ticked
    as intracellular; membrane = the markers ticked as membrane."""
    cyto_markers = list(cyto_markers or [])
    membrane_markers = list(membrane_markers or [])
    intra, memb = [], []
    for c, idx in enumerate(stain_complete_df.index[:n_channels]):
        if idx == 'NUCLEI':
            continue
        marker = stain_complete_df.loc[idx, 'Marker']
        if idx != 'CYTOPLASM' and marker in membrane_markers:
            memb.append(c)
        elif idx == 'CYTOPLASM' or marker in cyto_markers:
            intra.append(c)
    return intra, memb


def _rethreshold_cytoplasm_channels(equalized, stain_complete_df, channels, nuclei_diameter,
                                    cell_diameter, r_zxyz, settings):
    """'Threshold image'-like mask (Z, Y, X, C) in which only *channels* (the
    cytoplasm / membrane stains) are thresholded, with their own settings
    (``segment_cytoplasm(cyto_threshold=...)``); the other channels are empty."""
    import contextlib
    import io
    import matplotlib.pyplot as plt
    s = dict(settings)
    method = s.get("threshold_method", "otsu")
    g = float(s.get("global_threshold_weight", 0.15))
    w = float(s.get("sauvola_weight", 0.60))
    w = min(w, 1.0 - g)
    window = float(s.get("window_um") or cell_diameter)
    n_fig = len(plt.get_fignums())
    with contextlib.redirect_stdout(io.StringIO()):
        sub = apply_threshold_per_channel(
            np.asarray(equalized)[..., channels], stain_complete_df.iloc[channels],
            nuclei_diameter, window, r_zxyz, threshold_method=method,
            global_threshold_weight=g, sauvola_weight=w)
    for num in plt.get_fignums()[n_fig:]:
        plt.close(num)
    out = np.zeros(np.asarray(equalized).shape, dtype=bool)
    out[..., channels] = sub > 0
    names = ", ".join(str(stain_complete_df.iloc[c]["Marker"]) for c in channels)
    print(f"Cytoplasm stains thresholded with their own settings ({names}): {method}, global "
          f"weight {g:.2f}, Sauvola weight {w:.2f}, window {window:.1f} um")
    return out


def build_cytoplasm_mask(threshold_img, intra_channels, membrane_channels, max_cell_area,
                         close_radius=2):
    """Union of the thresholded intracellular stains and the filled membrane
    stains.

    *threshold_img* is the 'Threshold image' (Z, Y, X, C) of Cell 15. Nothing
    else is added: no nuclei, no hole filling of the union, no growth.
    Membrane rings are filled up to *max_cell_area* voxels per XY slice,
    after closing their gaps with a ball of *close_radius* voxels (see
    `_fill_membrane_rings`).
    """
    mask = np.zeros(threshold_img.shape[:3], dtype=bool)
    for c in intra_channels:
        mask |= threshold_img[:, :, :, c] > 0
    if membrane_channels:
        memb = np.zeros(threshold_img.shape[:3], dtype=bool)
        for c in membrane_channels:
            memb |= threshold_img[:, :, :, c] > 0
        mask |= _fill_membrane_rings(memb, max_cell_area, close_radius=close_radius)
    return mask


def segment_nuclei_cellpose(image_3d, nuclei_diameter, voxel_size, model_type='nuclei'):
    """Segment nuclei using Cellpose 3D.

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X)
        Single-channel intensity image (e.g. filtered NUCLEI channel).
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    voxel_size : tuple (Z, Y, X)
        Physical voxel spacing in micrometers, used for anisotropy.
    model_type : str
        Cellpose model name (default ``'nuclei'``).

    Returns
    -------
    labels : ndarray (Z, Y, X), int32
        Integer label volume (0 = background).
    """
    from cellpose import models

    model = models.Cellpose(model_type=model_type, gpu=True)

    xy_spacing = float(np.mean([voxel_size[1], voxel_size[2]]))
    diameter_px = nuclei_diameter / xy_spacing
    anisotropy = float(voxel_size[0] / xy_spacing)

    labels, _, _, _ = model.eval(
        image_3d,
        diameter=diameter_px,
        anisotropy=anisotropy,
        do_3D=True,
        channels=[0, 0],
    )

    print(f"Total nuclei found: {int(labels.max())}")
    return labels.astype(np.int32)


def segment_cytoplasm_cellpose(image_3d, cell_diameter, voxel_size, nuclei_labels, model_type='cyto3'):
    """Shape whole-cell (cytoplasm) instances using Cellpose 3D, seeded by nuclei labels.

    Cellpose segments the cell bodies independently of the nuclei labelling, so
    each resulting instance is relabelled to the nucleus label it overlaps the
    most with. Instances with no nucleus overlap are discarded (background);
    nuclei with no matching instance are left for the caller's gap-fill step.

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X)
        Single-channel intensity image (e.g. filtered CYTOPLASM channel).
    cell_diameter : float
        Approximate whole-cell diameter in micrometers.
    voxel_size : tuple (Z, Y, X)
        Physical voxel spacing in micrometers, used for anisotropy.
    nuclei_labels : ndarray (Z, Y, X), int
        Nuclei label volume used to reconcile Cellpose instance IDs with the
        rest of the pipeline's nuclei-driven labelling.
    model_type : str
        Cellpose model name (default ``'cyto3'``).

    Returns
    -------
    labels : ndarray (Z, Y, X), int32
        Integer label volume matching nuclei IDs (0 = background).
    """
    from cellpose import models

    model = models.Cellpose(model_type=model_type, gpu=True)

    xy_spacing = float(np.mean([voxel_size[1], voxel_size[2]]))
    diameter_px = cell_diameter / xy_spacing
    anisotropy = float(voxel_size[0] / xy_spacing)

    if image_3d.shape[0] == 1:
        # 2D image: Cellpose in 2D mode on the single plane.
        cell_labels_2d, _, _, _ = model.eval(
            image_3d[0], diameter=diameter_px, channels=[0, 0],
        )
        cell_labels = np.asarray(cell_labels_2d)[np.newaxis]
    else:
        cell_labels, _, _, _ = model.eval(
            image_3d,
            diameter=diameter_px,
            anisotropy=anisotropy,
            do_3D=True,
            channels=[0, 0],
        )

    labels = np.zeros_like(cell_labels, dtype=np.int32)
    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue
        instance_mask = cell_labels == cell_id
        overlapping_nuclei = nuclei_labels[instance_mask]
        overlapping_nuclei = overlapping_nuclei[overlapping_nuclei > 0]
        if overlapping_nuclei.size == 0:
            continue
        labels[instance_mask] = int(np.argmax(np.bincount(overlapping_nuclei)))

    print(
        f"Cellpose: {int(np.count_nonzero(np.unique(labels)))} cells shaped "
        f"(model={model_type}, diameter_px={diameter_px:.1f}, "
        f"anisotropy={anisotropy:.2f})"
    )
    return labels


def _display_nuclei_roundness_size_table(debug_info):
    """Display a one-row summary table of the roundness/size-based keep-or-discard
    decisions made by `segment_nuclei_watershed`, in place of verbose per-cluster text.
    """
    stats_df = pd.DataFrame([{
        'Roundness threshold': debug_info['nuclei_min_roundness'],
        'Removed (too small/non-round)': debug_info['removed_by_roundness'],
        'Non-round clusters re-split': debug_info['nonround_cluster_splits'],
        'Still-merged clusters (kept)': debug_info['protected_nonround_clusters'],
        'Expected nucleus size (voxels)': int(round(debug_info['expected_nucleus_vox'])),
        'Multi-nucleus size threshold (voxels)': int(round(debug_info['multi_nucleus_vox_thresh'])),
    }])
    try:
        from IPython.display import display as _display
        _display(stats_df)
    except Exception:
        pass
    return stats_df


# Escalating split configurations used by `refine_oversized_nuclei`, from
# gentlest to strongest. Each entry is (name, config overrides, diameter scale).
# The diameter scale shrinks the nucleus diameter the crop re-segmentation
# assumes, so seeds/peaks are allowed closer together at higher levels.
_NUCLEI_SPLIT_ESCALATION_LEVELS = (
    ("aggressive", {
        "nuclei_bridge_shrink_factor": 0.30,
        "nuclei_split_diameter_min_factor": 0.40,
        "nuclei_split_diameter_max_factor": 1.4,
        "nuclei_split_diameter_scales": 5,
        "nuclei_seed_min_fraction": 0.02,
        "nuclei_intensity_weight": 1.25,
        "nuclei_gradient_weight": 1.5,
        "nuclei_gradient_smooth_sigma": 1.0,
    }, 1.0),
    ("very aggressive", {
        "nuclei_bridge_shrink_factor": 0.36,
        "nuclei_split_diameter_min_factor": 0.35,
        "nuclei_split_diameter_max_factor": 1.2,
        "nuclei_split_diameter_scales": 6,
        "nuclei_seed_min_fraction": 0.015,
        "nuclei_intensity_weight": 1.5,
        "nuclei_gradient_weight": 2.0,
        "nuclei_gradient_smooth_sigma": 0.75,
        "z_split_aggressive": True,
    }, 0.9),
    ("extreme", {
        "nuclei_bridge_shrink_factor": 0.44,
        "nuclei_split_diameter_min_factor": 0.30,
        "nuclei_split_diameter_max_factor": 1.0,
        "nuclei_split_diameter_scales": 8,
        "nuclei_seed_min_fraction": 0.01,
        "nuclei_intensity_weight": 2.0,
        "nuclei_gradient_weight": 3.0,
        "nuclei_gradient_smooth_sigma": 0.5,
        "z_split_aggressive": True,
    }, 0.8),
)

# For each escalated key: True = larger value splits more, False = smaller does.
# Used so an escalation level is never gentler than the user's base config.
_NUCLEI_SPLIT_AGGRESSIVE_DIRECTION = {
    "nuclei_bridge_shrink_factor": True,
    "nuclei_split_diameter_min_factor": False,
    "nuclei_split_diameter_max_factor": False,
    "nuclei_split_diameter_scales": True,
    "nuclei_seed_min_fraction": False,
    "nuclei_intensity_weight": True,
    "nuclei_gradient_weight": True,
    "nuclei_gradient_smooth_sigma": False,
    "z_split_aggressive": True,
}


def _escalated_nuclei_split_config(base_config, level):
    """Return (name, config, diameter_scale) for escalation *level* (0-based),
    never gentler than *base_config* on any escalated key."""
    name, overrides, diameter_scale = _NUCLEI_SPLIT_ESCALATION_LEVELS[level]
    cfg = dict(base_config or {})
    for key, value in overrides.items():
        if key in cfg:
            pick = max if _NUCLEI_SPLIT_AGGRESSIVE_DIRECTION[key] else min
            cfg[key] = type(value)(pick(cfg[key], value))
        else:
            cfg[key] = value
    # The stronger levels rely on the signal's own dip between nuclei.
    cfg["split_by_intensity_gradient"] = True
    return name, cfg, diameter_scale


def apply_split_gamma(img, gamma):
    """Gamma on an intensity image used for splitting: values are scaled to
    0-1 by the image maximum, raised to *gamma* and scaled back. The brightest
    voxel keeps its value; with gamma > 1 dim voxels -- e.g. the seams between
    touching nuclei or cells -- get darker faster than the bright signal, so
    the split zones stand out more. ``gamma`` None or 1 returns *img*
    unchanged."""
    if img is None or gamma is None or abs(float(gamma) - 1.0) < 1e-6:
        return img
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, None)
    top = float(arr.max()) if arr.size else 0.0
    if top <= 0:
        return arr
    return ((arr / top) ** float(gamma) * top).astype(np.float32)


def _resegment_nucleus_assembly(
    assembly_mask,
    intensity_crop,
    intensity_raw_crop,
    r_zxyz,
    nuclei_diameter,
    split_config,
    min_piece_vox,
):
    """Re-run the watershed split on one cropped nuclei assembly.

    Returns an int32 label crop (0 outside *assembly_mask*) covering every
    voxel of the assembly, or None if it could not be split into >= 2 pieces
    of at least *min_piece_vox* voxels.
    """
    r_zX, r_zY, r_zZ = r_zxyz
    pad = 4  # > the 3-voxel boundary margin used by segment_nuclei_watershed
    mask_p = np.pad(assembly_mask, pad, mode="constant", constant_values=False)
    inten_p = np.pad(intensity_crop.astype(np.float32), pad, mode="edge")
    inten_raw_p = None
    if intensity_raw_crop is not None:
        inten_raw_p = np.pad(intensity_raw_crop.astype(np.float32), pad, mode="edge")
    # Only the assembly itself should carry signal, not neighbouring nuclei
    # that fall inside its bounding box.
    inten_p = np.where(mask_p, inten_p, 0.0)
    if inten_raw_p is not None:
        inten_raw_p = np.where(mask_p, inten_raw_p, 0.0)

    sub_labels, _ = segment_nuclei_watershed(
        binary_mask=mask_p,
        r_zX=r_zX, r_zY=r_zY, r_zZ=r_zZ,
        nuclei_diameter=nuclei_diameter,
        intensity_img=inten_p,
        intensity_img_raw=inten_raw_p,
        progress=None,
        **split_config,
    )
    sub_labels = sub_labels[pad:-pad, pad:-pad, pad:-pad]
    sub_labels = np.where(assembly_mask, sub_labels, 0).astype(np.int32)

    # Drop fragments too small to be a nucleus; their voxels are refilled below.
    ids, counts = np.unique(sub_labels[sub_labels > 0], return_counts=True)
    keep_ids = ids[counts >= min_piece_vox]
    if keep_ids.size < 2:
        return None
    sub_labels[~np.isin(sub_labels, keep_ids)] = 0

    # Give every assembly voxel back to a piece so no nuclear signal is lost:
    # watershed along the distance map first, then nearest piece for any
    # voxel the watershed could not reach (e.g. disconnected fragments).
    distance = ndi.distance_transform_edt(assembly_mask, sampling=(r_zZ, r_zY, r_zX))
    filled = watershed(-distance, sub_labels, mask=assembly_mask).astype(np.int32)
    holes = assembly_mask & (filled == 0)
    if np.any(holes):
        _, nearest = ndi.distance_transform_edt(filled == 0, return_indices=True)
        filled[holes] = filled[tuple(idx[holes] for idx in nearest)]
    return filled


def refine_oversized_nuclei(
    labels,
    intensity_img,
    r_zxyz,
    nuclei_diameter,
    base_split_config=None,
    intensity_img_raw=None,
    oversize_factor=2.0,
    max_iterations=3,
    reference_vox=None,
    undersize_factor=0.5,
    progress=None,
    verbose=True,
    intensity_gamma_levels=None,
):
    """Iteratively split nuclei labels that are much bigger than one nucleus.

    A label whose volume is more than ``oversize_factor`` times the reference
    single-nucleus volume (``reference_vox``, e.g. from
    `estimate_reference_nucleus_size`; defaults to a sphere of
    ``nuclei_diameter``) is treated as an assembly of several nuclei that the
    first segmentation pass left merged.
    Each assembly is cropped and re-segmented on its own with a split
    configuration that gets stronger each iteration:

    1. "aggressive"       - stronger erosion, more scales, smaller seeds
    2. "very aggressive"  - + stronger intensity/gradient weighting, Z split
    3. "extreme"          - strongest erosion, closest allowed seeds

    Pieces that are still oversized after one iteration are passed on to the
    next, stronger level. Assemblies that cannot be split at any level are
    kept unchanged (never deleted). Every voxel of a split assembly is
    reassigned to one of its pieces, so no nuclear signal is lost.

    Parameters
    ----------
    labels : ndarray (Z, Y, X), int
        Nuclei label volume from the first segmentation pass.
    intensity_img : ndarray (Z, Y, X)
        Nuclei-channel intensity ('Filtered image').
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in micrometers.
    nuclei_diameter : float
        Expected nucleus diameter in micrometers.
    base_split_config : dict, optional
        The user's `nuclei_split_config`; escalation levels are never gentler
        than it.
    intensity_img_raw : ndarray (Z, Y, X), optional
        Same channel before contrast/gamma clipping ('Denoised image').
    oversize_factor : float
        Volume, in multiples of one expected nucleus, above which a label is
        treated as an assembly (default 2.0).
    max_iterations : int
        Number of escalation levels to try (1-3, default 3).
    reference_vox : float, optional
        Reference single-nucleus volume in voxels. None = sphere of
        ``nuclei_diameter``. The crop re-segmentation also assumes the
        matching diameter.
    undersize_factor : float
        A split is only accepted with pieces of at least this many reference
        volumes (scaled with the level's diameter), so splitting never
        creates labels `merge_undersized_nuclei` would treat as fragments.
    verbose : bool
        Print a per-iteration summary and display a per-assembly table.
    intensity_gamma_levels : sequence of float, optional
        Extra gamma applied to the assembly's intensity crop at each level
        (`apply_split_gamma`), e.g. ``[1.33, 1.67, 2.0]``: a growing gamma
        darkens the seams between merged nuclei more at every level, making
        the split more aggressive. None (default) = no gamma.

    Returns
    -------
    labels_out : ndarray (Z, Y, X), int32
        Refined, sequentially relabelled nuclei volume.
    refine_info : dict
        Keys: 'expected_nucleus_vox', 'oversize_vox_thresh', 'iterations'
        (list of per-iteration dicts), 'assemblies' (per-assembly DataFrame),
        'n_before', 'n_after', 'unresolved'.
    """
    from skimage.segmentation import relabel_sequential

    r_zX, r_zY, r_zZ = r_zxyz
    voxel_um3 = r_zZ * r_zY * r_zX
    input_vox = max(1.0, _sphere_volume_um3(nuclei_diameter) / voxel_um3)
    expected_vox = float(reference_vox) if reference_vox else input_vox
    expected_um3 = expected_vox * voxel_um3
    # Diameter matching the reference volume, used by the crop re-segmentation.
    reference_diameter = nuclei_diameter * (expected_vox / input_vox) ** (1.0 / 3.0)
    oversize_vox = float(oversize_factor) * expected_vox
    n_iter = int(np.clip(max_iterations, 1, len(_NUCLEI_SPLIT_ESCALATION_LEVELS)))

    labels_out = np.asarray(labels).astype(np.int32, copy=True)
    n_before = int(np.count_nonzero(np.unique(labels_out)))
    next_id = int(labels_out.max()) + 1
    iterations = []
    records = []
    # Labels that failed to split at the previous level, or pieces produced
    # there that are still oversized, are retried at the next level.
    candidate_ids = None

    if verbose:
        print(
            f"Iterative split: reference nucleus ~ {expected_um3:.0f} um^3 "
            f"({expected_vox:.0f} voxels); labels > {oversize_factor:g}x "
            f"({oversize_vox:.0f} voxels) are treated as nuclei assemblies."
        )

    for level in range(n_iter):
        level_name, cfg, diameter_scale = _escalated_nuclei_split_config(
            base_split_config, level
        )
        volumes = np.bincount(labels_out.ravel())
        oversized = [int(i) for i in np.nonzero(volumes > oversize_vox)[0] if i > 0]
        if candidate_ids is not None:
            oversized = [i for i in oversized if i in candidate_ids]
        if not oversized:
            break
        level_gamma = None
        if intensity_gamma_levels:
            level_gamma = intensity_gamma_levels[min(level, len(intensity_gamma_levels) - 1)]

        if verbose:
            print(
                f"  Iteration {level + 1}/{n_iter} - {len(oversized)} nuclei "
                f"assembl{'y' if len(oversized) == 1 else 'ies'} detected -> "
                f"re-splitting with '{level_name}' configuration "
                f"(bridge shrink={cfg['nuclei_bridge_shrink_factor']:.2f}, "
                f"gradient weight={cfg['nuclei_gradient_weight']:.1f}, "
                f"diameter x{diameter_scale:g}"
                + (f", intensity gamma x{level_gamma:.2f}" if level_gamma else "")
                + ")"
            )

        slices = ndi.find_objects(labels_out)
        min_piece_vox = max(
            8, int(undersize_factor * expected_vox * diameter_scale ** 3)
        )
        split_count = 0
        new_nuclei = 0
        next_candidates = set()

        for lab in _progress_iter(
            oversized, progress,
            desc=f'Step 17 - Split assemblies ({level_name})', leave=False,
        ):
            sl = slices[lab - 1]
            if sl is None:
                continue
            crop_labels = labels_out[sl]
            assembly_mask = crop_labels == lab
            vol = int(np.count_nonzero(assembly_mask))
            pieces = _resegment_nucleus_assembly(
                assembly_mask,
                apply_split_gamma(intensity_img[sl], level_gamma),
                None if intensity_img_raw is None
                else apply_split_gamma(intensity_img_raw[sl], level_gamma),
                r_zxyz,
                reference_diameter * diameter_scale,
                cfg,
                min_piece_vox,
            )
            if pieces is None:
                next_candidates.add(lab)
                records.append({
                    'Iteration': level + 1, 'Configuration': level_name,
                    'Label': lab, 'Volume (voxels)': vol,
                    'Size (x reference)': round(vol / expected_vox, 2),
                    'Result': 'not split', 'Pieces': 1,
                })
                continue

            piece_ids = [int(p) for p in np.unique(pieces) if p > 0]
            # The first piece keeps the assembly's label id; the others get new ids.
            id_map = {piece_ids[0]: lab}
            for p in piece_ids[1:]:
                id_map[p] = next_id
                next_id += 1
            for p, new_lab in id_map.items():
                piece_mask = pieces == p
                crop_labels[piece_mask] = new_lab
                if np.count_nonzero(piece_mask) > oversize_vox:
                    next_candidates.add(new_lab)

            split_count += 1
            new_nuclei += len(piece_ids) - 1
            records.append({
                'Iteration': level + 1, 'Configuration': level_name,
                'Label': lab, 'Volume (voxels)': vol,
                'Size (x reference)': round(vol / expected_vox, 2),
                'Result': f'split into {len(piece_ids)}', 'Pieces': len(piece_ids),
            })

        iterations.append({
            'iteration': level + 1,
            'configuration': level_name,
            'assemblies': len(oversized),
            'split': split_count,
            'new_nuclei': new_nuclei,
        })
        if verbose:
            print(
                f"    -> {split_count}/{len(oversized)} split, "
                f"+{new_nuclei} nuclei recovered"
            )
        candidate_ids = next_candidates

    volumes = np.bincount(labels_out.ravel())
    unresolved = [int(i) for i in np.nonzero(volumes > oversize_vox)[0] if i > 0]
    labels_out, _, _ = relabel_sequential(labels_out)
    n_after = int(np.count_nonzero(np.unique(labels_out)))
    assemblies_df = pd.DataFrame(records)

    if verbose:
        if not iterations:
            print("  No oversized nuclei assemblies found - nothing to split.")
        else:
            print(
                f"Iterative split done: {n_before} -> {n_after} nuclei "
                f"(+{n_after - n_before}). "
                f"{len(unresolved)} oversized label(s) still unresolved "
                f"after all levels (kept as-is)."
            )
            try:
                from IPython.display import display as _display
                _display(assemblies_df)
            except Exception:
                pass

    refine_info = {
        'expected_nucleus_vox': expected_vox,
        'oversize_vox_thresh': oversize_vox,
        'iterations': iterations,
        'assemblies': assemblies_df,
        'n_before': n_before,
        'n_after': n_after,
        'unresolved': len(unresolved),
    }
    return labels_out, refine_info


def _sphere_volume_um3(diameter_um):
    """Volume (um^3) of a sphere of the given diameter."""
    return (4.0 * np.pi * ((float(diameter_um) / 2.0) ** 3)) / 3.0


def _display_if_notebook(df):
    try:
        from IPython.display import display as _display
        _display(df)
    except Exception:
        pass


def estimate_reference_nucleus_size(
    labels,
    r_zxyz,
    nuclei_diameter,
    size_reference="blend",
    plausible_range=(0.25, 4.0),
    min_labels=5,
    verbose=True,
):
    """Estimate the reference single-nucleus volume used to judge label sizes.

    Two estimates are combined:

    - the **input** volume: a sphere of ``nuclei_diameter``;
    - the **detected** volume: the median volume of the labels found by the
      first segmentation pass, using only "plausible" labels (between
      ``plausible_range`` x the input volume) so fragments and assemblies do
      not bias it. The median is used because it is robust to the remaining
      outliers.

    Parameters
    ----------
    size_reference : {"blend", "input", "detected"}
        - "blend"    (default): geometric mean of the input and detected
          volumes -- follows the data when cells are on average bigger or
          smaller than the input diameter, but stays anchored to it.
        - "input"    : input diameter only.
        - "detected" : detected median only.
        If fewer than ``min_labels`` plausible labels exist, the input
        volume is used whatever the setting.

    Returns
    -------
    reference_vox : float
        Reference single-nucleus volume in voxels.
    info : dict
        'input_vox', 'detected_vox' (or None), 'n_plausible',
        'reference_vox', 'reference_diameter_um', 'size_reference'.
    """
    choices = ("blend", "input", "detected")
    if size_reference not in choices:
        raise ValueError(
            f"Unknown size_reference '{size_reference}'. Choose from {choices}."
        )
    r_zX, r_zY, r_zZ = r_zxyz
    voxel_um3 = r_zZ * r_zY * r_zX
    input_vox = max(1.0, _sphere_volume_um3(nuclei_diameter) / voxel_um3)

    volumes = np.bincount(np.asarray(labels).ravel())[1:]
    volumes = volumes[volumes > 0]
    lo, hi = plausible_range
    plausible = volumes[(volumes >= lo * input_vox) & (volumes <= hi * input_vox)]
    detected_vox = float(np.median(plausible)) if plausible.size >= min_labels else None

    used = size_reference
    if detected_vox is None or size_reference == "input":
        reference_vox = input_vox
        used = "input"
    elif size_reference == "detected":
        reference_vox = detected_vox
    else:
        reference_vox = float(np.sqrt(input_vox * detected_vox))

    reference_diameter = nuclei_diameter * (reference_vox / input_vox) ** (1.0 / 3.0)

    if verbose:
        def _diam(vox):
            return nuclei_diameter * (vox / input_vox) ** (1.0 / 3.0)
        print(
            f"Reference nucleus size: input diameter {nuclei_diameter:g} um -> "
            f"{input_vox:.0f} voxels"
        )
        if detected_vox is None:
            print(
                f"  Detected: only {plausible.size} plausible label(s) "
                f"(< {min_labels}) -> using the input size."
            )
        else:
            print(
                f"  Detected: median of {plausible.size} plausible labels = "
                f"{detected_vox:.0f} voxels (~{_diam(detected_vox):.1f} um, "
                f"{detected_vox / input_vox:.2f}x input)"
            )
        print(
            f"  Reference ('{used}'): {reference_vox:.0f} voxels "
            f"(~{reference_diameter:.1f} um diameter)"
        )

    info = {
        'input_vox': input_vox,
        'detected_vox': detected_vox,
        'n_plausible': int(plausible.size),
        'reference_vox': reference_vox,
        'reference_diameter_um': reference_diameter,
        'size_reference': used,
    }
    return reference_vox, info


def merge_undersized_nuclei(
    labels,
    intensity_img,
    reference_vox,
    undersize_factor=0.5,
    max_merged_factor=1.5,
    min_interface_ratio=0.85,
    smooth_sigma=1.0,
    max_rounds=5,
    progress=None,
    verbose=True,
):
    """Merge labels that are too small to be a whole nucleus into a touching
    neighbour when both most likely belong to the same nucleus (the
    conservative counterpart of `refine_oversized_nuclei`).

    A label smaller than ``undersize_factor`` x ``reference_vox`` is a
    candidate fragment. It is merged into the touching neighbour it shares
    the largest contact surface with, provided that:

    - the merged label would still be one plausible nucleus
      (<= ``max_merged_factor`` x ``reference_vox``), and
    - there is no intensity dip at the contact surface: the mean intensity
      there is at least ``min_interface_ratio`` x the dimmer label's mean.
      Two different nuclei touching normally show a darker seam between
      them; two fragments of one nucleus don't.

    If the best-contact neighbour fails, the next one is tried. Fragments
    that touch nothing, or where every neighbour fails, are kept as they are
    (never deleted). This is repeated for up to ``max_rounds`` rounds, since
    a merged fragment may still be small and merge again.

    Returns
    -------
    labels_out : ndarray (Z, Y, X), int32
        Sequentially relabelled nuclei volume.
    merge_info : dict
        'undersize_vox_thresh', 'n_small', 'n_merged', 'n_before', 'n_after',
        'rounds', 'fragments' (per-fragment DataFrame).
    """
    from skimage.segmentation import relabel_sequential

    labels_out = np.asarray(labels).astype(np.int32, copy=True)
    smoothed = ndi.gaussian_filter(
        np.asarray(intensity_img, dtype=np.float32), sigma=max(float(smooth_sigma), 1e-3)
    )
    small_vox = float(undersize_factor) * reference_vox
    max_merged_vox = float(max_merged_factor) * reference_vox
    n_before = int(np.count_nonzero(np.unique(labels_out)))

    n_bins = int(labels_out.max()) + 1
    volumes = np.bincount(labels_out.ravel(), minlength=n_bins).astype(np.int64)
    sums = np.bincount(labels_out.ravel(), weights=smoothed.ravel(), minlength=n_bins)
    n_small = int(np.count_nonzero((volumes[1:] > 0) & (volumes[1:] < small_vox)))
    struct = ndi.generate_binary_structure(3, 1)
    status = {}  # fragment label -> latest record
    rounds = 0

    if verbose:
        print(
            f"Merge of undersized labels: labels < {undersize_factor:g}x reference "
            f"({small_vox:.0f} voxels) are treated as possible fragments of one "
            f"nucleus - {n_small} found."
        )

    for rnd in range(max_rounds):
        order = np.argsort(volumes)
        small_ids = [int(i) for i in order if i > 0 and 0 < volumes[i] < small_vox]
        if not small_ids:
            break
        rounds += 1
        slices = ndi.find_objects(labels_out)
        grown = set()  # labels that absorbed a fragment: bbox is stale this round
        merged_this_round = 0

        for a in _progress_iter(
            small_ids, progress, desc=f'Step 17 - Merge fragments (round {rnd + 1})',
            leave=False,
        ):
            if volumes[a] == 0 or volumes[a] >= small_vox or a in grown:
                continue
            sl = slices[a - 1]
            if sl is None:
                continue
            sl_p = tuple(
                slice(max(0, s.start - 1), min(dim, s.stop + 1))
                for s, dim in zip(sl, labels_out.shape)
            )
            crop = labels_out[sl_p]
            inten = smoothed[sl_p]
            mask_a = crop == a
            ring = ndi.binary_dilation(mask_a, structure=struct) & ~mask_a
            nb_ids, contact = np.unique(crop[ring], return_counts=True)
            keep = nb_ids > 0
            nb_ids, contact = nb_ids[keep], contact[keep]

            record = {
                'Label': a,
                'Volume (voxels)': int(volumes[a]),
                'Size (x reference)': round(volumes[a] / reference_vox, 2),
                'Merged into': None,
                'Contact (voxels)': 0,
                'Interface/intensity ratio': None,
                'Result': 'kept (isolated)',
            }
            if nb_ids.size == 0:
                status[a] = record
                continue

            record['Result'] = 'kept'
            reasons = []
            for idx in np.argsort(-contact):
                b = int(nb_ids[idx])
                if volumes[a] + volumes[b] > max_merged_vox:
                    reasons.append('merged size too big')
                    continue
                mask_b = crop == b
                interface = (ring & mask_b) | (
                    ndi.binary_dilation(mask_b, structure=struct) & mask_a
                )
                interface_mean = float(inten[interface].mean())
                mean_a = sums[a] / volumes[a]
                mean_b = sums[b] / volumes[b]
                ratio = interface_mean / max(min(mean_a, mean_b), 1e-6)
                if ratio < min_interface_ratio:
                    reasons.append('intensity dip at contact')
                    record['Interface/intensity ratio'] = round(ratio, 2)
                    continue

                crop[mask_a] = b
                volumes[b] += volumes[a]
                sums[b] += sums[a]
                volumes[a] = 0
                sums[a] = 0.0
                grown.add(b)
                merged_this_round += 1
                record.update({
                    'Merged into': b,
                    'Contact (voxels)': int(contact[idx]),
                    'Interface/intensity ratio': round(ratio, 2),
                    'Result': 'merged',
                })
                break
            if record['Result'] != 'merged':
                record['Result'] = 'kept (' + ', '.join(sorted(set(reasons))) + ')'
            status[a] = record

        if verbose:
            print(f"  Round {rnd + 1}: {merged_this_round} fragment(s) merged")
        if merged_this_round == 0:
            break

    labels_out, _, _ = relabel_sequential(labels_out)
    n_after = int(np.count_nonzero(np.unique(labels_out)))
    fragments_df = pd.DataFrame(list(status.values()))
    n_merged = int((fragments_df['Result'] == 'merged').sum()) if len(fragments_df) else 0

    if verbose and n_small:
        print(
            f"Merge done: {n_before} -> {n_after} nuclei (-{n_before - n_after}). "
            f"{n_merged} fragment(s) merged into a touching neighbour, "
            f"{len(fragments_df) - n_merged} small label(s) kept as-is."
        )
        _display_if_notebook(fragments_df)

    merge_info = {
        'undersize_vox_thresh': small_vox,
        'n_small': n_small,
        'n_merged': n_merged,
        'n_before': n_before,
        'n_after': n_after,
        'rounds': rounds,
        'fragments': fragments_df,
    }
    return labels_out, merge_info


def _label_overlap_map(labels_a, labels_b):
    """For each label of *labels_a*, the sorted list of *labels_b* labels it
    overlaps (voxel by voxel). Labels with no counterpart map to []."""
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    inside = (a > 0) & (b > 0)
    base = int(b.max()) + 1
    keys = np.unique(a[inside].astype(np.int64) * base + b[inside])
    out = defaultdict(list)
    for key in keys:
        out[int(key // base)].append(int(key % base))
    return out


def _nuclei_refinement_recap(labels_first, labels_merged, labels_final,
                             reference_vox, merge_info, split_info):
    """One row per first-pass nucleus: what the merge and split steps did to
    it (or tried and didn't), and its label number(s) after refinement.

    The merge step works on first-pass labels, the split step on the labels
    renumbered after merging, and both renumber at the end - this links the
    three numberings through voxel overlap (merging and splitting only
    regroup voxels, so the overlap is exact).
    """
    first_to_merged = _label_overlap_map(labels_first, labels_merged)
    merged_to_final = _label_overlap_map(labels_merged, labels_final)
    merged_members = defaultdict(list)  # post-merge label -> first-pass labels
    for first_id, merged_ids in first_to_merged.items():
        for m in merged_ids:
            merged_members[m].append(first_id)

    fragments = {}
    absorbed = defaultdict(list)
    if merge_info is not None and len(merge_info['fragments']):
        for rec in merge_info['fragments'].to_dict('records'):
            fragments[int(rec['Label'])] = rec
            if rec['Result'] == 'merged':
                absorbed[int(rec['Merged into'])].append(int(rec['Label']))

    split_tried = {}
    if split_info is not None and len(split_info['assemblies']):
        # Only the labels present right after the merge step; ids created by
        # a split are its own pieces and are followed through the overlap.
        for rec in split_info['assemblies'].to_dict('records'):
            lab = int(rec['Label'])
            if lab in merged_members:
                split_tried[lab] = max(split_tried.get(lab, 0), int(rec['Iteration']))

    volumes = np.bincount(np.asarray(labels_first).ravel())
    rows = []
    for first_id in sorted(first_to_merged):
        merged_ids = first_to_merged[first_id]
        final_ids = sorted({f for m in merged_ids for f in merged_to_final.get(m, [])})
        members = sorted({x for m in merged_ids for x in merged_members[m]})

        # Merge step
        if first_id in fragments:
            rec = fragments[first_id]
            merge_txt = (f"merged into {int(rec['Merged into'])}"
                         if rec['Result'] == 'merged'
                         else f"fragment, {rec['Result']}")
        elif absorbed.get(first_id):
            merge_txt = f"absorbed fragment(s) {', '.join(map(str, sorted(absorbed[first_id])))}"
        else:
            merge_txt = '-'

        # Split step
        tried = [split_tried[m] for m in merged_ids if m in split_tried]
        if len(final_ids) > 1:
            split_txt = f"split into {len(final_ids)}"
        elif tried:
            split_txt = f"oversized, not split ({max(tried)} level(s) tried)"
        else:
            split_txt = '-'

        merged = len(members) > 1
        split = len(final_ids) > 1
        outcome = ('merged + split' if merged and split else
                   'merged' if merged else
                   'split' if split else 'kept')
        rows.append({
            'First-pass label': first_id,
            'Volume (voxels)': int(volumes[first_id]),
            'Size (x reference)': round(volumes[first_id] / reference_vox, 2),
            'Merge step': merge_txt,
            'Split step': split_txt,
            'Outcome': outcome,
            'Merged with': ', '.join(str(x) for x in members if x != first_id) or '-',
            'Final label(s)': ', '.join(map(str, final_ids)) or '-',
        })
    return pd.DataFrame(rows)


def refine_nuclei_by_size(
    labels,
    intensity_img,
    r_zxyz,
    nuclei_diameter,
    base_split_config=None,
    intensity_img_raw=None,
    size_reference="blend",
    merge_undersized=True,
    undersize_factor=0.5,
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    progress=None,
    verbose=True,
    intensity_gamma_levels=None,
):
    """Size-based refinement of a nuclei label volume, in three steps:

    1. `estimate_reference_nucleus_size` - reference single-nucleus volume
       from the input diameter and/or the detected label sizes.
    2. `merge_undersized_nuclei` (if ``merge_undersized``) - conservative:
       labels < ``undersize_factor`` x reference are merged into a touching
       neighbour when both look like one nucleus.
    3. `refine_oversized_nuclei` (if ``iterative_split``) - aggressive:
       labels > ``oversize_factor`` x reference are re-split with escalating
       configurations.

    Finally a recap table (`_nuclei_refinement_recap`) lists every
    first-pass nucleus with what each step did to it (or tried and didn't)
    and its final label number(s), since both steps renumber the labels.

    ``intensity_gamma_levels`` (optional) is passed to
    `refine_oversized_nuclei`: an extra intensity gamma per re-split level.

    Returns
    -------
    labels_out : ndarray (Z, Y, X), int32
    info : dict with keys 'reference', 'merge', 'split' (None if skipped)
        and 'recap' (DataFrame).
    """
    reference_vox, ref_info = estimate_reference_nucleus_size(
        labels, r_zxyz, nuclei_diameter,
        size_reference=size_reference, verbose=verbose,
    )
    labels_first = np.asarray(labels).astype(np.int32, copy=False)
    labels_out = labels_first
    merge_info = split_info = None

    if merge_undersized:
        labels_out, merge_info = merge_undersized_nuclei(
            labels_out, intensity_img, reference_vox,
            undersize_factor=undersize_factor,
            progress=progress, verbose=verbose,
        )
    labels_merged = labels_out
    if iterative_split:
        labels_out, split_info = refine_oversized_nuclei(
            labels_out,
            intensity_img=intensity_img,
            intensity_img_raw=intensity_img_raw,
            r_zxyz=r_zxyz,
            nuclei_diameter=nuclei_diameter,
            base_split_config=base_split_config,
            oversize_factor=oversize_factor,
            max_iterations=max_split_iterations,
            reference_vox=reference_vox,
            undersize_factor=undersize_factor,
            progress=progress,
            verbose=verbose,
            intensity_gamma_levels=intensity_gamma_levels,
        )

    recap_df = _nuclei_refinement_recap(
        labels_first, labels_merged, labels_out,
        reference_vox, merge_info, split_info,
    )
    if verbose and len(recap_df):
        counts = recap_df['Outcome'].value_counts()
        print(
            "\nRefinement recap (first-pass label -> final label): "
            + ", ".join(f"{counts.get(k, 0)} {k}" for k in
                        ('kept', 'merged', 'split', 'merged + split'))
            + f" -> {int(np.count_nonzero(np.unique(labels_out)))} final nuclei. "
            "Labels are renumbered: use 'Final label(s)' for the later cells."
        )
        with pd.option_context('display.max_rows', None):
            _display_if_notebook(recap_df)
    return labels_out, {'reference': ref_info, 'merge': merge_info,
                        'split': split_info, 'recap': recap_df}


def _segment_nuclei_2d(mask, intensity, r_zxyz, nuclei_diameter, trig_cellpose=False,
                       trig_stardist=False, size_reference="blend", iterative_split=True,
                       oversize_factor=2.0, max_split_iterations=3, merge_undersized=True,
                       undersize_factor=0.5, split_gamma=1.0, split_gamma_step=0.0,
                       min_piece_fraction=0.0, progress=None):
    """Nuclei of a 2D image (single plane), for `segment_nuclei`: the 2D
    segmentation of the LIVE/DEAD notebook (`_ld_segment_objects_2d`) --
    watershed seeded at the bright centres separated by an intensity dip
    (or Cellpose / StarDist in 2D), split gamma, merge of tiny split pieces
    and size refinement on areas. Returns the labels as (1, Y, X)."""
    r_zX, r_zY, _ = r_zxyz
    method = 'cellpose' if trig_cellpose else ('stardist' if trig_stardist else 'watershed')
    print("2D image: nuclei segmented in the image plane.")
    labels = _ld_segment_objects_2d(
        mask,
        intensity,
        pixel_yx=(r_zY, r_zX),
        diameter_um=nuclei_diameter,
        what="nuclei",
        method=method,
        cellpose_model='nuclei',
        size_reference=size_reference,
        iterative_split=iterative_split,
        oversize_factor=oversize_factor,
        max_split_iterations=max_split_iterations,
        merge_undersized=merge_undersized,
        undersize_factor=undersize_factor,
        split_gamma=split_gamma,
        split_gamma_step=split_gamma_step,
        min_piece_fraction=min_piece_fraction,
        progress=progress,
    )
    return np.asarray(labels, dtype=np.int32)[np.newaxis]


def _basin_saddles(basins, surface, n_basins):
    """Saddle height between every pair of touching basins: the highest
    voxel pair across their common boundary, each pair valued at the lower of
    its two intensities (the pass point on the way from one peak to the
    other). Returns ``{(i, j): height}`` with i < j."""
    saddles = {}
    for axis in range(basins.ndim):
        a = [slice(None)] * basins.ndim
        b = [slice(None)] * basins.ndim
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        la, lb = basins[tuple(a)], basins[tuple(b)]
        sel = (la > 0) & (lb > 0) & (la != lb)
        if not sel.any():
            continue
        v = np.minimum(surface[tuple(a)][sel], surface[tuple(b)][sel])
        code = (np.minimum(la[sel], lb[sel]).astype(np.int64) * (n_basins + 1)
                + np.maximum(la[sel], lb[sel]))
        order = np.argsort(code, kind="stable")
        code, v = code[order], v[order]
        uniq, start = np.unique(code, return_index=True)
        for c, s in zip(uniq, np.maximum.reduceat(v, start)):
            key = (int(c // (n_basins + 1)), int(c % (n_basins + 1)))
            saddles[key] = max(saddles.get(key, -np.inf), float(s))
    return saddles


def split_nuclei_by_saddles(labels, intensity, spacing_zyx, nuclei_diameter, max_dip=0.8,
                            smooth_fraction=0.1, seed_separation=0.5, min_piece_fraction=0.25,
                            background=None):
    """v1.6.4: split labels that hold several nuclei, where the image shows an
    intensity valley between them.

    Inside every label, the intensity (smoothed by ``smooth_fraction`` x
    ``nuclei_diameter``) is flooded from its local peaks (at least
    ``seed_separation`` x the diameter apart). Neighbouring basins are then
    merged back, the highest saddle first, unless the saddle between them
    is below ``max_dip`` x the dimmer of the two peaks (intensities above
    ``background``), i.e. unless there is a real dark seam between two
    bright nuclei. Basins smaller than ``min_piece_fraction`` x a nucleus
    of ``nuclei_diameter`` are always merged back. This catches nuclei
    stacked along Z (merged by the axial blur, often with a dim one under a
    bright one), which the shape-based watershed cannot separate because
    they overlap in every XY slice.

    Parameters
    ----------
    labels : ndarray (Z, Y, X) int
    intensity : ndarray (Z, Y, X)
        Linear nuclei intensity (e.g. the 'Denoised image' channel, before
        contrast / gamma).
    spacing_zyx : tuple of float
        Physical voxel size (um).
    nuclei_diameter : float
        um.
    max_dip : float
        0-1; 0 = off. Higher splits more (a shallower valley is enough).
    background : float, optional
        Intensity of the background; default the 25th percentile of
        ``intensity``.

    Returns
    -------
    (labels, n_new) : the split labels (ids of untouched labels kept, new
    pieces numbered after the maximum) and the number of new labels.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    labels = np.asarray(labels)
    if not max_dip or max_dip <= 0 or not labels.any():
        return labels, 0
    spacing = np.asarray(spacing_zyx, dtype=float)
    if labels.ndim != 3 or labels.shape[0] == 1:
        spacing = spacing[-2:]
        labels3 = labels.reshape(labels.shape[-2:])
        intensity3 = np.asarray(intensity).reshape(labels.shape[-2:])
    else:
        labels3, intensity3 = labels, np.asarray(intensity)
    if background is None:
        background = float(np.percentile(intensity3[::2, ::2] if intensity3.ndim == 2
                                          else intensity3[::2, ::2, ::2], 25))
    ndim = labels3.ndim
    sigma = float(smooth_fraction) * float(nuclei_diameter) / spacing
    nucleus_vox = (_sphere_volume_um3(nuclei_diameter) if ndim == 3
                   else np.pi * (nuclei_diameter / 2.0) ** 2) / float(np.prod(spacing))
    min_piece = float(min_piece_fraction) * nucleus_vox
    rad = np.maximum(1, np.round(float(seed_separation) * nuclei_diameter / 2.0 / spacing)).astype(int)
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in rad)]
    footprint = sum((g / r) ** 2 for g, r in zip(grids, rad)) <= 1.0
    pad = np.ceil(3 * sigma).astype(int) + 1

    out = labels3.copy()
    next_id = int(out.max()) + 1
    n_new = 0
    for lab, sl in enumerate(ndi.find_objects(labels3), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - p), min(n, s.stop + p))
                   for s, p, n in zip(sl, pad, labels3.shape))
        m = labels3[sl] == lab
        if m.sum() < 2 * min_piece:
            continue
        surf = ndi.gaussian_filter(intensity3[sl].astype(np.float32), sigma) - background
        peaks = peak_local_max(np.where(m, surf, surf.min()), footprint=footprint,
                               labels=m.astype(np.int32), exclude_border=False)
        if len(peaks) < 2:
            continue
        k = len(peaks)
        markers = np.zeros(m.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, k + 1)
        basins = watershed(-surf, markers, mask=m)
        peak_val = {i + 1: float(surf[tuple(p)]) for i, p in enumerate(peaks)}
        size = dict(enumerate(np.bincount(basins.ravel(), minlength=k + 1)[1:].astype(float), 1))
        saddles = _basin_saddles(basins, surf, k)
        parent = list(range(k + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        changed = True
        while changed:
            changed = False
            between = {}
            for (i, j), s in saddles.items():
                ri, rj = find(i), find(j)
                if ri != rj:
                    key = (min(ri, rj), max(ri, rj))
                    between[key] = max(between.get(key, -np.inf), s)
            for (ri, rj), s in sorted(between.items(), key=lambda t: -t[1]):
                if find(ri) != ri or find(rj) != rj:
                    continue  # merged earlier in this pass
                dip = s / max(min(peak_val[ri], peak_val[rj]), 1e-6)
                if dip > max_dip or min(size[ri], size[rj]) < min_piece:
                    parent[rj] = ri
                    peak_val[ri] = max(peak_val[ri], peak_val[rj])
                    size[ri] += size[rj]
                    changed = True
        roots = sorted({find(i) for i in range(1, k + 1)})
        if len(roots) < 2:
            continue
        keep = max(roots, key=lambda r: size[r])  # the biggest piece keeps the id
        lut = np.zeros(k + 1, np.int64)
        for r in roots:
            if r == keep:
                continue
            for i in range(1, k + 1):
                if find(i) == r:
                    lut[i] = next_id
            next_id += 1
            n_new += 1
        new_ids = lut[basins]
        region = out[sl]
        sel = m & (new_ids > 0)
        region[sel] = new_ids[sel]
    return out.reshape(labels.shape), n_new


def typical_nucleus_shape(labels, spacing_zyx):
    """Median volume (um^3) and median elongation (Z extent / XY diameter of
    the widest slice) of the labels: the image's own typical nucleus, robust
    to the few merged or fragmented labels. Returns ``(volume, elongation,
    per_label)`` with ``per_label`` = {id: (volume, z_extent, xy_diameter)}."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    per_label = {}
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        area = (labels[sl] == lab).sum(axis=(1, 2))
        per_label[lab] = (float(area.sum() * spacing.prod()), float((area > 0).sum() * spacing[0]),
                          float(2.0 * np.sqrt(area.max() * spacing[1] * spacing[2] / np.pi)))
    if not per_label:
        return 0.0, 1.0, per_label
    vols = np.array([v[0] for v in per_label.values()])
    elong = np.array([v[1] / max(v[2], 1e-6) for v in per_label.values()])
    return float(np.median(vols)), float(np.median(elong)), per_label


def split_tall_nuclei(labels, intensity, spacing_zyx, volume_factor=1.3, elongation_factor=1.3,
                      max_pieces=4, min_piece_fraction=0.4):
    """v1.6.4: size-aware split of nuclei stacked along Z with no visible seam.

    The typical nucleus of the image is measured from its own labels
    (`typical_nucleus_shape`: median volume and median Z/XY elongation). A
    label both bigger than ``volume_factor`` x that volume AND more elongated
    than ``elongation_factor`` x that elongation is taken as n nuclei on top
    of each other (n = the smaller of the two ratios, rounded, 2 to
    ``max_pieces``): a big but not taller nucleus - a large single one - is
    left alone, and so are small and normal labels. The label is cut across
    its long axis (Z scaled by the typical elongation, so a typical nucleus
    is round) near each equal-volume position, at the minimum of the
    intensity profile along the axis (the faint seam, if any). The cut is
    dropped if a piece would be smaller than ``min_piece_fraction`` x the
    typical volume.

    On the GT2/GT4 ground truths, the merges left after `split_nuclei_by_saddles`
    were mostly such pairs, 6-8 um apart in Z with no intensity dip.

    Returns ``(labels, n_new, typical_volume, typical_elongation)``.
    """
    labels = np.asarray(labels)
    if not volume_factor or volume_factor <= 0 or labels.ndim != 3 or labels.shape[0] < 3 \
            or not labels.any():
        return labels, 0, 0.0, 1.0
    spacing = np.asarray(spacing_zyx, dtype=float)
    ref_vol, ref_el, per_label = typical_nucleus_shape(labels, spacing)
    objs = ndi.find_objects(labels)
    out = labels.copy()
    next_id = int(out.max()) + 1
    n_new = 0
    for lab, (vol, z_ext, d_xy) in per_label.items():
        el = z_ext / max(d_xy, 1e-6)
        if vol < volume_factor * ref_vol or el < elongation_factor * ref_el:
            continue
        n = int(np.clip(round(min(vol / ref_vol, el / ref_el)), 2, max_pieces))
        sl = objs[lab - 1]
        m = labels[sl] == lab
        coords = np.argwhere(m).astype(float) * spacing
        coords[:, 0] /= ref_el
        centred = coords - coords.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        t = centred @ axis
        weights = np.asarray(intensity)[sl][m].astype(float)
        n_bins = max(8, int(np.ptp(t) / 0.5))
        mass, edges = np.histogram(t, bins=n_bins, weights=weights)
        mass = ndi.gaussian_filter1d(mass, 1.0)
        centres = 0.5 * (edges[1:] + edges[:-1])
        half_window = 0.5 * np.ptp(t) / n
        cuts = []
        for q in np.quantile(t, np.arange(1, n) / n):
            window = np.abs(centres - q) <= half_window
            cuts.append(centres[window][np.argmin(mass[window])] if window.any() else q)
        piece = np.searchsorted(np.sort(cuts), t)
        if np.bincount(piece, minlength=n).min() * spacing.prod() < min_piece_fraction * ref_vol:
            continue
        pieces = np.zeros(m.shape, np.int64)
        pieces[m] = piece
        region = out[sl]
        for k in range(1, n):
            region[m & (pieces == k)] = next_id
            next_id += 1
            n_new += 1
    return out, n_new, ref_vol, ref_el


def refine_nuclei_boundaries(labels, intensity, spacing_zyx, edge_fraction=0.3, margin_um=1.0,
                             background=None, peak_percentile=95, smooth_um=0.3):
    """v1.6.4: redraw every nucleus edge at ``edge_fraction`` of its own peak.

    A single threshold cuts dim nuclei too small and leaves bright ones too
    big (and too long in Z, where the axial blur keeps them above the
    threshold). Here each label keeps the voxels of its territory (the label
    grown by ``margin_um`` into free space, never into a neighbour) brighter
    than ``background + edge_fraction x (peak - background)``, with ``peak``
    its own ``peak_percentile`` intensity: the same relative edge for every
    nucleus, like a contrast adapted to each one. Only the part connected to
    the original label is kept, holes are filled. A label whose new mask
    would lose more than 80% of it is left as it was.

    On the two ground truths, the manual edges sit at ~0.35-0.4 of each
    nucleus' peak (dim and bright alike); ``edge_fraction`` 0.3 matched them
    best (the cut falls just outside the last painted voxel).

    Parameters
    ----------
    labels : ndarray (Z, Y, X) int
    intensity : ndarray (Z, Y, X)
        Linear nuclei intensity ('Denoised image' channel).
    spacing_zyx : tuple of float
        Physical voxel size (um).
    edge_fraction : float
        0-1; 0 = off.
    margin_um : float
        How far (um) an edge may move out of the original label.
    background : float, optional
        Default: the 25th percentile of ``intensity``.

    Returns
    -------
    labels : ndarray, same shape and ids.
    """
    labels = np.asarray(labels)
    if not edge_fraction or edge_fraction <= 0 or not labels.any():
        return labels
    spacing = np.asarray(spacing_zyx, dtype=float)
    intensity = np.asarray(intensity, dtype=np.float32)
    if labels.ndim == 3 and labels.shape[0] == 1:
        return refine_nuclei_boundaries(labels[0], intensity.reshape(labels.shape[1:]), spacing[-2:],
                                        edge_fraction, margin_um, background, peak_percentile,
                                        smooth_um)[np.newaxis]
    spacing = spacing[-labels.ndim:]
    if background is None:
        background = float(np.percentile(intensity[(slice(None, None, 2),) * labels.ndim], 25))
    if smooth_um and smooth_um > 0:
        intensity = ndi.gaussian_filter(intensity, float(smooth_um) / spacing)
    territory = expand_labels_um(labels, float(margin_um), tuple(spacing)) if margin_um > 0 \
        else labels.copy()
    out = np.zeros_like(labels)
    for lab, sl in enumerate(ndi.find_objects(territory), start=1):
        if sl is None:
            continue
        core = labels[sl] == lab
        if not core.any():
            continue
        terr = territory[sl] == lab
        it = intensity[sl]
        peak = float(np.percentile(it[core], peak_percentile))
        new = core
        if peak > background:
            m = terr & (it >= background + float(edge_fraction) * (peak - background))
            comp, _ = ndi.label(m)
            touching = np.unique(comp[core & m])
            m = np.isin(comp, touching[touching > 0])
            m = ndi.binary_fill_holes(m) & terr
            if m.sum() >= 0.2 * core.sum():
                new = m
        out[sl][new] = lab
    return out


def rescue_missed_nuclei(labels, intensity, spacing_zyx, min_contrast=0.25, grow_fraction=0.35,
                         background=None, min_volume_fraction=0.25, max_volume_fraction=2.0):
    """v1.6.4: add back nuclei the threshold mask missed (small, dim ones).

    The typical nucleus of the image is measured from its own labels
    (`typical_nucleus_shape`). Nucleus-sized bright spots are found with a
    difference of Gaussians at that scale (Z stretched by the typical
    elongation); a spot farther than one typical radius from every label and
    brighter than ``background + min_contrast x`` (the labels' median peak
    above background) is grown to ``grow_fraction`` of its own peak - the
    same relative edge as `refine_nuclei_boundaries` - connected to the spot
    and outside every label. It is kept if its volume is ``min_volume_fraction``
    - ``max_volume_fraction`` x the typical nucleus, it is at most twice as
    elongated, and it does not run out of its search box (haze, not a nucleus).

    Returns ``(labels, n_added)``.
    """
    labels = np.asarray(labels)
    if not min_contrast or min_contrast <= 0 or labels.ndim != 3 or labels.shape[0] < 3 \
            or not labels.any():
        return labels, 0
    spacing = np.asarray(spacing_zyx, dtype=float)
    raw = np.asarray(intensity, dtype=np.float32)
    if background is None:
        background = float(np.percentile(raw[::2, ::2, ::2], 25))
    ref_vol, ref_el, _ = typical_nucleus_shape(labels, spacing)
    r_um = (3.0 * ref_vol / (4.0 * np.pi * ref_el)) ** (1.0 / 3.0)
    semi_axes = np.array([r_um * ref_el, r_um, r_um])
    sigma = semi_axes / np.sqrt(3.0) / spacing
    dog = ndi.gaussian_filter(raw, sigma) - ndi.gaussian_filter(raw, 2.0 * sigma)
    smooth = ndi.gaussian_filter(raw, 0.5 / spacing) - background
    peaks = ndi.labeled_comprehension(smooth, labels, np.arange(1, int(labels.max()) + 1),
                                      lambda v: np.percentile(v, 95), float, 0)
    ref_peak = float(np.median(peaks))
    size = np.maximum(1, np.round(semi_axes / spacing)).astype(int) | 1
    maxima = (dog == ndi.maximum_filter(dog, size=size)) & (dog > 0)
    maxima &= expand_labels_um(labels, r_um, tuple(spacing)) == 0
    cand = np.argwhere(maxima)
    contrast = smooth[tuple(cand.T)]
    keep = contrast >= float(min_contrast) * ref_peak
    cand, contrast = cand[keep], contrast[keep]
    out = labels.copy()
    next_id = int(out.max()) + 1
    half = np.ceil(1.5 * semi_axes / spacing).astype(int)
    shape = np.array(raw.shape)
    n_added = 0
    for i in np.argsort(contrast)[::-1]:
        c = cand[i]
        if out[tuple(c)]:
            continue
        lo, hi = np.maximum(c - half, 0), np.minimum(c + half + 1, shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        m = (smooth[sl] > grow_fraction * contrast[i]) & (out[sl] == 0)
        comp, _ = ndi.label(m)
        k = comp[tuple(c - lo)]
        if k == 0:
            continue
        m = comp == k
        vol = m.sum() * spacing.prod()
        if not (min_volume_fraction * ref_vol <= vol <= max_volume_fraction * ref_vol):
            continue
        area = m.sum(axis=(1, 2))
        elong = (area > 0).sum() * spacing[0] / (2.0 * np.sqrt(area.max() * spacing[1] * spacing[2] / np.pi))
        if elong > 2.0 * ref_el:
            continue
        touches = [(m[0].any() and lo[0] > 0) or (m[-1].any() and hi[0] < shape[0]),
                   m[:, 0].any() and lo[1] > 0, m[:, -1].any() and hi[1] < shape[1],
                   m[:, :, 0].any() and lo[2] > 0, m[:, :, -1].any() and hi[2] < shape[2]]
        if any(touches):
            continue
        out[sl][m] = next_id
        next_id += 1
        n_added += 1
    return out, n_added


# v1.6.4: what the settings of validation_tuning/multi_v168/best_settings.json
# were tuned for. On GT1 (0.33 um pixels) and GT2/GT4 (0.21 um), both at
# scale_factor 0.5, a nucleus of 8 um spanned 12 and 19 voxels: the voxel-based
# settings (sigma, gradient smoothing, minimum piece size...) are validated in
# that range. The um settings below were tuned for an 8 um nucleus.
AUTO_REFERENCE = {
    "nuclei_diameter": 8.0,
    "voxels_per_diameter": (12.0, 19.0),
    "um_settings": {"z_bg_radius_um": 10.0, "nuclei_edge_margin_um": 1.0},
}


def _profile_extent(profile, centre, level, rise):
    """Number of steps (half-step precision) around ``centre`` over which
    ``profile`` stays >= ``level``; None if a neighbour (a rise of more than
    ``rise`` above the running minimum) or the image border comes first."""
    total = 0.0
    for step in (-1, 1):
        i, run_min, n = centre, profile[centre], 0.0
        while True:
            if not 0 <= i + step < len(profile):
                return None
            v = profile[i + step]
            if v < level:
                total += n + 0.5
                break
            if v > run_min + rise:
                return None
            run_min = min(run_min, v)
            i += step
            n += 1
    return total


def measure_nuclei_in_image(nuclei_zyx, voxel_size_zyx, contrast=None, edge_level=0.3,
                            work_um=0.5, min_peak=0.15, max_candidates=600, verbose=True):
    """v1.6.4: typical nucleus of an image, measured from the image alone.

    ``nuclei_zyx`` is the NUCLEI channel on its original grid (any range); it
    is normalized to 0-255 like Cell 9 and, with ``contrast`` =
    ``(Cont_min, Cont_max, Gamma)``, given the contrast and gamma of the
    stain table (Cell 8), so the measure sees what the segmentation sees.

    Nucleus centres are the maxima of a scale-normalized Laplacian of
    Gaussian (in XY, 2.8-17 um diameters, on a ~``work_um`` grid). Through
    each centre the intensity profiles along X, Y and Z give the nucleus
    extent at ``edge_level`` x its own peak above background; only clean
    profiles count (the intensity falls below that level before rising
    towards a neighbour or reaching the border), so touching nuclei do not
    shorten it. ``edge_level`` 0.3 calibrated: on the GT1 and GT2/GT4 ground
    truths the equivalent diameter (d_xy^2 d_z)^(1/3) came out 8.1 and 7.7 um,
    where the tuner found ``nuclei_diameter`` 8 best on both.

    Returns a dict: ``n_measured``, ``n_candidates``, ``d_xy_um``, ``d_z_um``,
    ``elongation`` (Z / XY), ``eq_diameter_um`` and its interquartile range,
    ``background``, ``noise``, ``snr`` (median peak / noise),
    ``haze`` (spread of the per-slice background, as a fraction of the
    median peak).
    """
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    img = np.asarray(nuclei_zyx, dtype=np.float32)
    if img.ndim != 3 or img.shape[0] < 3:
        raise ValueError("measure_nuclei_in_image needs a 3D (Z, Y, X) stack.")
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo + 1e-8) * 255.0
    if contrast is not None:
        cmin, cmax, gamma = contrast
        img = napari_contrast_gamma_uint8(img, (float(cmin), float(cmax)), float(gamma)) \
            .astype(np.float32)
    f = max(1, int(round(work_um / spacing[1])))
    if f > 1:
        img = ndi.zoom(img, (1, 1.0 / f, 1.0 / f), order=1)
        spacing = spacing * [1, f, f]
    sm = ndi.gaussian_filter(img, 0.5 / spacing)
    sample = sm[::2, ::2, ::2]
    bg = float(np.percentile(sample, 25))
    noise = 1.4826 * float(np.median(np.abs(img[::2, ::2, ::2] - np.median(img[::2, ::2, ::2]))))
    sig = sm - bg
    best_r = best_s = None
    for s in np.geomspace(1.0, 6.0, 9):  # LoG sigma, um (disk diameter ~2.8 sigma)
        sg = (0, s / spacing[1], s / spacing[2])
        r = -(s ** 2) * (ndi.gaussian_filter(sm, sg, order=(0, 2, 0))
                         + ndi.gaussian_filter(sm, sg, order=(0, 0, 2)))
        if best_r is None:
            best_r, best_s = r, np.full(r.shape, s, np.float32)
        else:
            better = r > best_r
            best_r = np.where(better, r, best_r)
            best_s = np.where(better, s, best_s)
    size = np.maximum(3, np.round(4.0 / spacing).astype(int) | 1)
    maxima = (best_r == ndi.maximum_filter(best_r, size=size)) & (best_r > 0)
    pts = np.argwhere(maxima)
    peak = sig[tuple(pts.T)]
    keep = peak > min_peak * float(np.percentile(sig, 99.5))
    pts, peak = pts[keep], peak[keep]
    order = np.argsort(best_r[tuple(pts.T)])[::-1][:max_candidates]
    rows = []
    clean = np.zeros(len(order), dtype=bool)
    for n, i in enumerate(order):
        z, y, x = pts[i]
        p = float(peak[i])
        level, rise = edge_level * p, 0.1 * p
        ex = _profile_extent(sig[z, y, :], x, level, rise)
        ey = _profile_extent(sig[z, :, x], y, level, rise)
        ez = _profile_extent(sig[:, y, x], z, level, rise)
        if ex is None or ey is None or ez is None:
            continue
        clean[n] = True
        rows.append((0.5 * (ex * spacing[2] + ey * spacing[1]), ez * spacing[0], p))
    slice_bg = np.percentile(sm[:, ::2, ::2].reshape(sm.shape[0], -1), 25, axis=1)
    out = {"n_measured": len(rows), "n_candidates": int(len(order)), "background": bg,
           "noise": noise, "edge_level": edge_level,
           # v1.6.4: for measure_image_features (um, original grid)
           "centres_um": pts[order] * spacing if len(order) else np.zeros((0, 3)),
           "clean_mask": clean}
    if rows:
        a = np.array(rows)
        eq = (a[:, 0] ** 2 * a[:, 1]) ** (1.0 / 3.0)
        med_peak = float(np.median(a[:, 2]))
        out.update(d_xy_um=float(np.median(a[:, 0])), d_z_um=float(np.median(a[:, 1])),
                   elongation=float(np.median(a[:, 1] / a[:, 0])),
                   eq_diameter_um=float(np.median(eq)),
                   eq_diameter_iqr_um=tuple(float(v) for v in np.percentile(eq, [25, 75])),
                   snr=med_peak / max(noise, 1e-6),
                   haze=float(np.ptp(np.percentile(slice_bg, [10, 90])) / max(med_peak, 1e-6)))
    if verbose:
        if rows:
            print(f"Measured {out['n_measured']} nuclei (of {out['n_candidates']} candidates): "
                  f"diameter {out['d_xy_um']:.1f} um in XY, {out['d_z_um']:.1f} um in Z "
                  f"(elongation {out['elongation']:.2f}); equivalent diameter "
                  f"{out['eq_diameter_um']:.1f} um (IQR {out['eq_diameter_iqr_um'][0]:.1f}-"
                  f"{out['eq_diameter_iqr_um'][1]:.1f}); signal/noise {out['snr']:.0f}, "
                  f"haze {out['haze']:.2f}")
        else:
            print(f"No nucleus with a clean profile among {out['n_candidates']} candidates.")
    return out


def _edge_profile_stats(profile, centre, bg, peak, rise, floor=0.05):
    """v1.6.4: steepest-edge level and 80-20% edge width (in steps) of a
    nucleus profile, on both sides of ``centre``.

    Walks outwards until the profile drops below ``bg + floor x (peak - bg)``;
    None for a side that meets a neighbour (a rise of more than ``rise`` above
    the running minimum) or the border first. Returns a list of
    ``(level, width)``, one per clean side: ``level`` = the intensity at the
    steepest descent as a fraction of ``peak`` above ``bg``.
    """
    out = []
    amp = peak - bg
    if amp <= 0:
        return out
    stop = bg + floor * amp
    for step in (-1, 1):
        vals = [float(profile[centre])]
        i, run_min, ok = centre, float(profile[centre]), False
        while 0 <= i + step < len(profile):
            v = float(profile[i + step])
            if v > run_min + rise:
                break
            vals.append(v)
            run_min = min(run_min, v)
            i += step
            if v < stop:
                ok = True
                break
        if not ok or len(vals) < 3:
            continue
        a = (np.asarray(vals) - bg) / amp
        drop = a[:-1] - a[1:]                  # descent between consecutive samples
        k = int(np.argmax(drop))
        level = 0.5 * (a[k] + a[k + 1])
        # 80% and 20% crossings (linear interpolation), first time below
        def cross(t):
            j = int(np.argmax(a < t))
            if j == 0:
                return None
            return j - 1 + (a[j - 1] - t) / max(a[j - 1] - a[j], 1e-9)
        c80, c20 = cross(0.8), cross(0.2)
        width = (c20 - c80) if c80 is not None and c20 is not None else None
        out.append((float(level), width))
    return out


def measure_image_features(nuclei_zyx, voxel_size_zyx, contrast=None, measurement=None,
                           smooth_um=0.3, work_um=0.5, min_peak=0.15, max_candidates=600,
                           verbose=True):
    """v1.6.4: image properties that the threshold / edge / split settings may
    depend on, measured from the image alone (no ground truth).

    The nucleus centres are those of `measure_nuclei_in_image` (found on the
    image with its contrast / gamma); everything else is measured on the
    LINEAR intensity, normalized 0-255 and smoothed by ``smooth_um``, with the
    25th-percentile background: the same intensity and background
    `refine_nuclei_boundaries` (``nuclei_edge_fraction``) works on.

    Returns a dict:

    - ``edge_level_xy`` / ``edge_level_z``: median intensity at the steepest
      descent of the nucleus profiles, as a fraction of the nucleus peak above
      background (sharp, well separated edges ~0.5; blurred / hazy edges, or
      nuclei smaller than the blur, lower).
    - ``edge_width_xy_um`` / ``edge_width_z_um``: median 80%-20% edge width.
    - ``nn_ratio``: median distance between neighbouring nucleus centres /
      the equivalent diameter (crowding; < 1.2 = mostly touching).
    - ``touching_fraction``: candidates with a neighbour closer than one
      diameter; ``clean_fraction``: candidates with clean profiles.
    - ``peak_spread``: 90th / 10th percentile of the nucleus peaks (dim nuclei
      among bright ones).
    - ``saturated_fraction``: voxels at the maximum value of the raw stack.
    - ``bg_unevenness``: spread (90th - 10th percentile) of the local
      background over XY tiles of ~4 diameters, / the median peak.
    - ``mask_fraction``: voxels above background + 0.25 x median peak
      (how much of the volume is nuclei-like).
    """
    from scipy.spatial import cKDTree
    raw = np.asarray(nuclei_zyx)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    if measurement is None:
        measurement = measure_nuclei_in_image(raw, spacing, contrast=contrast, work_um=work_um,
                                              min_peak=min_peak, max_candidates=max_candidates,
                                              verbose=False)
    centres = measurement.get("centres_um")
    d_eq = measurement.get("eq_diameter_um")
    hi = float(raw.max())
    out = {"saturated_fraction": float(np.mean(raw[::2, ::2, ::2] >= hi)) if hi > 0 else 0.0}
    if centres is None or len(centres) < 5 or d_eq is None:
        if verbose:
            print("Too few nuclei found to measure the image features.")
        return out
    lin = raw.astype(np.float32)
    lo = float(lin.min())
    lin = (lin - lo) / (hi - lo + 1e-8) * 255.0
    lin = ndi.gaussian_filter(lin, float(smooth_um) / spacing)
    bg = float(np.percentile(lin[::2, ::2, ::2], 25))

    pts = np.asarray(centres, dtype=float)
    idx = np.clip(np.round(pts / spacing).astype(int), 0, np.array(lin.shape) - 1)
    # snap to the local maximum (the centres come from a coarser grid)
    r = np.maximum(1, np.round(np.array([0.5, 0.6, 0.6]) / spacing).astype(int))
    peaks = []
    for n, (z, y, x) in enumerate(idx):
        sl = tuple(slice(max(0, c - k), c + k + 1) for c, k in zip((z, y, x), r))
        sub = lin[sl]
        off = np.unravel_index(int(np.argmax(sub)), sub.shape)
        idx[n] = [s.start + o for s, o in zip(sl, off)]
        peaks.append(float(sub.max()))
    peaks = np.asarray(peaks)

    # crowding, on all candidates
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nn = dist[:, 1]
    out.update(nn_ratio=float(np.median(nn) / d_eq),
               touching_fraction=float(np.mean(nn < d_eq)),
               clean_fraction=measurement["n_measured"] / max(1, measurement["n_candidates"]),
               peak_spread=float(np.percentile(peaks - bg, 90) /
                                 max(np.percentile(peaks - bg, 10), 1e-6)))

    # edges, on the clean nuclei only (measured profiles)
    clean = measurement.get("clean_mask")
    levels = {"xy": [], "z": []}
    widths = {"xy": [], "z": []}
    for n in (np.flatnonzero(clean) if clean is not None else range(len(idx))):
        z, y, x = idx[n]
        p = peaks[n]
        rise = 0.1 * (p - bg)
        for axis, prof, c, step in (("xy", lin[z, y, :], x, spacing[2]),
                                    ("xy", lin[z, :, x], y, spacing[1]),
                                    ("z", lin[:, y, x], z, spacing[0])):
            for level, width in _edge_profile_stats(prof, c, bg, p, rise):
                levels[axis].append(level)
                if width is not None:
                    widths[axis].append(width * step)
    for axis in ("xy", "z"):
        if levels[axis]:
            out[f"edge_level_{axis}"] = float(np.median(levels[axis]))
        if widths[axis]:
            out[f"edge_width_{axis}_um"] = float(np.median(widths[axis]))
    med_peak = float(np.median(peaks)) - bg

    # background over XY tiles (~4 diameters), 10th percentile through Z
    tile = np.maximum(8, np.round(4 * d_eq / spacing[1:]).astype(int))
    tiles = []
    for y0 in range(0, lin.shape[1] - tile[0] // 2, tile[0]):
        for x0 in range(0, lin.shape[2] - tile[1] // 2, tile[1]):
            tiles.append(float(np.percentile(lin[:, y0:y0 + tile[0]:2, x0:x0 + tile[1]:2], 10)))
    if len(tiles) >= 2:
        out["bg_unevenness"] = float(np.ptp(np.percentile(tiles, [10, 90])) / max(med_peak, 1e-6))
    out["mask_fraction"] = float(np.mean(lin[::2, ::2, ::2] > bg + 0.25 * med_peak))
    out["n_edges"] = len(levels["xy"])
    if verbose:
        def f(k, fmt=".2f"):
            return format(out[k], fmt) if k in out else "n/a"
        print(f"Image features: edge level {f('edge_level_xy')} XY / {f('edge_level_z')} Z of the "
              f"peak, edge width {f('edge_width_xy_um', '.1f')} / {f('edge_width_z_um', '.1f')} um; "
              f"neighbours at {f('nn_ratio')} diameters ({f('touching_fraction')} touching); "
              f"peak spread {f('peak_spread', '.1f')}x; saturated {f('saturated_fraction', '.4f')}; "
              f"background unevenness {f('bg_unevenness')}; nuclei-like volume "
              f"{f('mask_fraction')}")
    return out


def _scale_factor_change(nuclei_diameter, scale_factor, pixel_um, reference=None):
    """The ``scale_factor`` rule of `adapt_nuclei_settings`: changed only if a
    nucleus of ``nuclei_diameter`` would span fewer / more pixels than on the
    tuned images (``reference['voxels_per_diameter']``), to the nearest end of
    that range, as a multiple of 0.05 (at most 1.0: never upsampled).
    Returns ``(name, old, new, reason)`` or None."""
    ref = reference or AUTO_REFERENCE
    d, s, r_xy = float(nuclei_diameter), float(scale_factor), float(pixel_um)
    lo_vox, hi_vox = ref["voxels_per_diameter"]
    vox = d * s / r_xy
    if lo_vox <= vox <= hi_vox:
        return None
    steps = (lo_vox if vox < lo_vox else hi_vox) * r_xy / d / 0.05
    steps = np.ceil(steps - 1e-9) if vox < lo_vox else np.floor(steps + 1e-9)
    new_s = float(round(min(1.0, max(0.05, float(steps) * 0.05)), 2))
    if abs(new_s - s) <= 1e-9:
        return None
    return ("scale_factor", s, new_s,
            f"a nucleus spans {vox:.1f} voxels at {s:g}, {d * new_s / r_xy:.1f} at {new_s:g} "
            f"(tuned range {lo_vox:g}-{hi_vox:g})")


def adapt_scale_2d(nuclei_diameter, scale_factor, pixel_um, verbose=True):
    """2D images: the nucleus can't be measured (that needs a 3D stack), so
    only the resolution is adapted, with the ``scale_factor`` rule of the 3D
    adaptation applied to the given ``nuclei_diameter``. Returns
    ``(scale_factor, changes)``."""
    change = _scale_factor_change(nuclei_diameter, scale_factor, pixel_um)
    if verbose:
        if change:
            print(f"2D image: scale_factor {change[1]!r} -> {change[2]!r} ({change[3]}; "
                  f"nucleus size not measured in 2D: nuclei_diameter {nuclei_diameter:g} um "
                  "is used as given).")
        else:
            print(f"2D image: scale_factor {scale_factor:g} kept (a {nuclei_diameter:g} um "
                  "nucleus already spans the tuned number of pixels; nucleus size not "
                  "measured in 2D).")
    return (change[2], [change]) if change else (float(scale_factor), [])


def adapt_nuclei_settings(measurement, settings, voxel_size_zyx, reference=None,
                          min_measured=15, adapt_scale=True, diameter_tolerance=0.1,
                          verbose=True):
    """v1.6.4: adapt the NUCLEI settings to the nuclei measured in the image.

    ``settings``: dict with at least ``nuclei_diameter`` and ``scale_factor``
    (and optionally the um settings of ``reference['um_settings']``), e.g.
    the tuned settings. ``measurement``: from `measure_nuclei_in_image`.
    ``reference`` (default ``AUTO_REFERENCE``): what those settings were
    tuned for.

    - ``nuclei_diameter`` = the measured equivalent diameter, if it is more
      than ``diameter_tolerance`` (10%) off the current one. Smaller changes
      are within the precision of the measure: on the three ground truths
      (GT1, GT2/GT4, GT3) moving 8.0 um by 0.2-0.3 um only added noise.
    - The um settings tuned for the reference nucleus (haze radius, edge
      margin) are scaled with the new diameter (only when it changed).
    - ``scale_factor`` (``adapt_scale``): changed only if a nucleus would
      span fewer / more voxels than on the tuned images, to the nearest end
      of that range (at most 1.0: the image is never upsampled).
    - ``z_elongation`` is not changed: the tuned value did not follow the
      measured elongation (GT2/GT4 nuclei measure ~1.5, 2.5 scored best).

    Nothing changes if fewer than ``min_measured`` nuclei were measured.
    Returns ``(adapted settings, changes)``, ``changes`` = list of
    ``(name, old, new, reason)``.
    """
    ref = reference or AUTO_REFERENCE
    out = dict(settings)
    changes = []
    n = measurement.get("n_measured", 0)
    if n < min_measured or "eq_diameter_um" not in measurement:
        if verbose:
            print(f"Only {n} nuclei measured (< {min_measured}): settings left as they are.")
        return out, changes
    measured = round(float(measurement["eq_diameter_um"]), 1)
    current = float(settings["nuclei_diameter"])
    if abs(measured - current) > diameter_tolerance * current:
        changes.append(("nuclei_diameter", settings["nuclei_diameter"], measured,
                        f"measured equivalent diameter (> {diameter_tolerance:.0%} off)"))
        out["nuclei_diameter"] = d = measured
    else:
        d = current
    k = d / float(ref["nuclei_diameter"])
    for name, value in ref["um_settings"].items():
        if d != current and name in settings and settings[name] and settings[name] > 0:
            new = round(value * k, 1)
            if abs(new - float(settings[name])) >= 0.05:
                changes.append((name, settings[name], new,
                                f"{value:g} um for a {ref['nuclei_diameter']:g} um nucleus"))
                out[name] = new
    if adapt_scale:
        change = _scale_factor_change(d, settings["scale_factor"], voxel_size_zyx[-1], ref)
        if change:
            changes.append(change)
            out["scale_factor"] = change[2]
    if verbose:
        if changes:
            print("Settings adapted to the image:")
            for name, old, new, why in changes:
                print(f"  {name}: {old!r} -> {new!r}  ({why})")
        else:
            print("Settings already match the image: nothing changed.")
        el = measurement.get("elongation")
        if el is not None and "z_elongation" in settings:
            print(f"  (z_elongation kept at {settings['z_elongation']:g}; measured nucleus "
                  f"elongation {el:.2f})")
    return out, changes


def adapt_settings_to_image(original_image, stain_complete_df, voxel_size_zyx, settings,
                            condition="NUCLEI", adapt_scale=True, verbose=True,
                            with_features=False):
    """v1.6.4: `measure_nuclei_in_image` + `adapt_nuclei_settings` for the
    notebook (Cell 8b) and the batch mode.

    ``original_image``: (Z, Y, X, C) 'Original image' of the stack (Cell 7),
    channels in the order of ``stain_complete_df``; the ``condition``
    channel is measured with its ``Cont_min`` / ``Cont_max`` / ``Gamma``
    (Cell 8, chosen by the user or automatic). ``settings``: dict with
    ``nuclei_diameter``, ``scale_factor`` and the um settings to scale.
    Returns ``(adapted settings, changes)``; v1.6.4, ``with_features``:
    ``(adapted settings, changes, features)`` with the image features of
    `measure_image_features` and the nucleus measurement (for the record;
    they do not change any setting).
    """
    c = stain_complete_df.index.get_loc(condition)
    row = stain_complete_df.loc[condition]
    contrast = (float(row["Cont_min"]), float(row["Cont_max"]), float(row["Gamma"]))
    channel = np.asarray(original_image)[..., c]
    measurement = measure_nuclei_in_image(channel, voxel_size_zyx, contrast=contrast,
                                          verbose=verbose)
    out, changes = adapt_nuclei_settings(measurement, settings, voxel_size_zyx,
                                         adapt_scale=adapt_scale, verbose=verbose)
    if not with_features:
        return out, changes
    features = measure_image_features(channel, voxel_size_zyx, measurement=measurement,
                                      verbose=verbose)
    features.update({k: v for k, v in measurement.items()
                     if k not in ("centres_um", "clean_mask")})
    return out, changes, features


def postprocess_nuclei_labels(labels, intensity, spacing_zyx, nuclei_diameter,
                              flare_trim_factor=0.0, saddle_dip=0.0, tall_volume_factor=1.15,
                              tall_elongation_factor=0.0, rescue_contrast=0.0,
                              rescue_grow_fraction=0.35, edge_fraction=0.0, edge_margin_um=1.0,
                              expand_um=0.0, verbose=True):
    """v1.6.4: the steps after the watershed and size refinement, in order.

    flare trim -> saddle split -> tall split -> rescue of missed nuclei ->
    per-nucleus edges -> label growth. Each step is off at 0 (see
    `segment_nuclei` for the parameters). ``intensity`` is the linear NUCLEI
    channel ('Denoised image', before contrast / gamma), ``spacing_zyx`` the
    physical voxel size. One function so the tuner can re-run only these
    steps on cached watershed labels.
    """
    from skimage.segmentation import relabel_sequential
    say = print if verbose else (lambda *a, **k: None)
    im_out = labels
    if flare_trim_factor and flare_trim_factor > 0:
        im_out = trim_flared_slices(im_out, float(flare_trim_factor))
    tall_on = bool(tall_volume_factor and tall_volume_factor > 0 and
                   tall_elongation_factor and tall_elongation_factor > 0)
    if (saddle_dip and saddle_dip > 0) or (edge_fraction and edge_fraction > 0) or tall_on or \
            (rescue_contrast and rescue_contrast > 0):
        linear = np.asarray(intensity)
        background = float(np.percentile(linear[::2, ::2, ::2], 25))
        if saddle_dip and saddle_dip > 0:
            im_out, n_new = split_nuclei_by_saddles(
                im_out, linear, spacing_zyx, nuclei_diameter, max_dip=float(saddle_dip),
                background=background)
            if n_new:
                im_out, _, _ = relabel_sequential(im_out)
            say(f"Split at intensity valleys (saddle < {saddle_dip:g} x the dimmer "
                f"peak): +{n_new} -> {int(im_out.max())} nuclei")
        if tall_on:
            im_out, n_new, ref_vol, ref_el = split_tall_nuclei(
                im_out, ndi.gaussian_filter(linear.astype(np.float32), 0.5 / np.asarray(spacing_zyx)),
                spacing_zyx, volume_factor=float(tall_volume_factor),
                elongation_factor=float(tall_elongation_factor))
            if n_new:
                im_out, _, _ = relabel_sequential(im_out)
            say(f"Typical nucleus of this image: {ref_vol:.0f} um^3, Z/XY elongation "
                f"{ref_el:.2f}. Labels > {tall_elongation_factor:g}x that elongation (and "
                f"{tall_volume_factor:g}x that volume) split along Z: +{n_new} -> "
                f"{int(im_out.max())} nuclei")
        if rescue_contrast and rescue_contrast > 0:
            im_out, n_new = rescue_missed_nuclei(
                im_out, linear, spacing_zyx, min_contrast=float(rescue_contrast),
                grow_fraction=float(rescue_grow_fraction), background=background)
            say(f"Nuclei missed by the threshold added back (spots > {rescue_contrast:g} x the "
                f"median nucleus contrast): +{n_new} -> {int(im_out.max())} nuclei")
        if edge_fraction and edge_fraction > 0:
            im_out = refine_nuclei_boundaries(
                im_out, linear, spacing_zyx, edge_fraction=float(edge_fraction),
                margin_um=float(edge_margin_um), background=background)
            say(f"Nuclei edges redrawn at {edge_fraction:g} x each nucleus' own peak "
                f"(within {edge_margin_um:g} um of the label).")
    if expand_um and expand_um > 0:
        im_out = expand_labels_um(im_out, float(expand_um), tuple(spacing_zyx))
        say(f"Nuclei grown by {expand_um:g} um into free space.")
    return im_out


def segment_nuclei(
    im_final_stack,
    stain_df,
    stain_complete_df,
    r_zxyz,
    nuclei_diameter,
    nuclei_split_config=None,
    trig_stardist=False,
    trig_cellpose=False,
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    size_reference="blend",
    progress=None,
    refine_gamma_levels=None,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.0,
    z_elongation=1.0,
    nuclei_expand_um=0.0,
    nuclei_flare_trim_factor=0.0,
    nuclei_saddle_dip=0.0,
    nuclei_edge_fraction=0.0,
    nuclei_edge_margin_um=1.0,
    nuclei_tall_volume_factor=1.15,
    nuclei_tall_elongation_factor=0.0,
    nuclei_rescue_contrast=0.0,
    nuclei_rescue_grow_fraction=0.35,
):
    """
    Segment nuclei from the image using watershed, StarDist, or Cellpose.

    The first-pass labels are then refined by size with
    `refine_nuclei_by_size` (for every method):

    - a reference single-nucleus volume is estimated from the input
      ``nuclei_diameter`` and/or the median detected label size
      (``size_reference``);
    - with ``merge_undersized``, labels smaller than ``undersize_factor`` x
      reference are merged into a touching neighbour when there is no
      intensity dip between them (conservative: likely one nucleus);
    - with ``iterative_split``, labels larger than ``oversize_factor`` x
      reference are treated as merged nuclei assemblies and re-split with
      increasingly aggressive configurations, for up to
      ``max_split_iterations`` rounds.

    The watershed path passes the 'Filtered image' NUCLEI-channel intensity
    (or the max-projection across channels in LD mode) into
    `segment_nuclei_watershed` as `intensity_img`, so touching-but-round
    nuclei that a pure shape/distance watershed would leave as one
    non-round (and therefore roundness-filtered) blob get split using the
    actual brightness dip/gradient between them — see
    `split_by_intensity_gradient` in `get_nuclei_split_config()`. It also
    passes the same channel from 'Denoised image' (pre-contrast/gamma-clip)
    as `intensity_img_raw`, used only by the post-watershed cluster
    re-split strategies — dense/bright clusters are the regions most likely
    to have been clipped to the display ceiling by Cell 12, which erases
    exactly the internal brightness variation those strategies need.

    Parameters
    ----------
    im_final_stack : dict
        Must contain 'Threshold image', 'Filtered image', and 'Denoised
        image' arrays (Z, Y, X, C).
    stain_df, stain_complete_df : DataFrame
        Staining metadata.
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in micrometers.
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    nuclei_split_config : dict, optional
        Configuration dict returned by get_nuclei_split_config(). Default
        (None): the built-in 'balanced' first pass, with
        ``z_split_aggressive`` on; oversized labels are then re-split
        automatically by the size refinement below.
    trig_stardist : bool
        If True, use StarDist.
    trig_cellpose : bool
        If True, use Cellpose 3D (takes priority over StarDist and watershed).
    iterative_split : bool
        If True, iteratively re-split oversized labels (nuclei assemblies).
    oversize_factor : float
        Label volume, in multiples of the reference nucleus, above which a
        label is treated as an assembly to split (default 2.0).
    max_split_iterations : int
        Number of escalating split levels to try (1-3, default 3).
    merge_undersized : bool
        If True, merge undersized fragments into a touching neighbour when
        they most likely belong to the same nucleus.
    undersize_factor : float
        Label volume, in multiples of the reference nucleus, below which a
        label is treated as a possible fragment (default 0.5).
    size_reference : {"blend", "input", "detected"}
        How the reference nucleus volume is set - see
        `estimate_reference_nucleus_size` (default "blend").
    progress : callable, optional
        Progress wrapper (e.g. tqdm).
    refine_gamma_levels : sequence of float, optional
        Extra intensity gamma per re-split level of the size refinement (see
        `refine_oversized_nuclei`). None (default) = no gamma. Overrides the
        levels derived from ``split_gamma``/``split_gamma_step``.
    split_gamma : float
        NUCLEI-channel path (nuclei notebook v1.6.3): gamma applied to the
        intensity used to split touching nuclei (`apply_split_gamma`) -- the
        watershed and the fragment-merge seam test, not the Cellpose/StarDist
        input. > 1 makes the dark seams between nuclei darker faster than the
        nuclei. Default 1.0 = off.
    split_gamma_step : float
        Added to the gamma at each re-split level of the size refinement
        (level n uses ``split_gamma + n * split_gamma_step``). Default 0 = off.
    min_piece_fraction : float
        NUCLEI-channel watershed: after the first pass, pieces of each split
        island smaller than this x a nucleus of ``nuclei_diameter`` are merged
        into the closest big piece of the same island
        (`absorb_small_split_pieces`). Default 0 = off.
    z_elongation : float
        v1.6.4, 3D images. How many times taller (Z) than wide (XY) a nucleus
        looks in the image - e.g. ~2.5-3 at 20x, where the axial blur
        stretches every nucleus along Z. The nuclei segmentation then works
        with a Z voxel size ``z_elongation`` times smaller, so a stretched
        nucleus counts as one round nucleus of ``nuclei_diameter``: the
        expected nucleus volume, the erosion radii and the split distances
        match the real objects, and a single tall nucleus is no longer taken
        for an assembly and split along Z. The labels stay on the same grid;
        only the segmentation's notion of distance in Z changes. Default 1.0
        = off (v1.6.3 behaviour).
    nuclei_expand_um : float
        v1.6.4. Grow every final nucleus label by this distance (um, physical
        voxel size) into free space, without overlapping its neighbours - for
        nuclei that come out narrower than they are (threshold too tight at
        the edge). Default 0 = off.
    nuclei_flare_trim_factor : float
        v1.6.4, 3D images. Cut back the Z slices where a nucleus is wider
        than this x its median cross-section (blur flaring, typically in the
        first slices); see `trim_flared_slices`. Applied before
        ``nuclei_expand_um``. Default 0 = off.
    nuclei_saddle_dip : float
        v1.6.4, 3D NUCLEI channel. Split a label where the intensity between
        two of its peaks drops below this x the dimmer peak (e.g. two nuclei
        stacked in Z); see `split_nuclei_by_saddles`. 0.8 matched the ground
        truths best. Default 0 = off.
    nuclei_edge_fraction : float
        v1.6.4, 3D NUCLEI channel. Redraw every nucleus edge at this fraction
        of its own peak intensity (within ``nuclei_edge_margin_um`` of the
        label); see `refine_nuclei_boundaries`. 0.3 matched the ground truths
        best. Default 0 = off.
    nuclei_edge_margin_um : float
        How far (um) ``nuclei_edge_fraction`` may move an edge outwards.
    nuclei_tall_volume_factor, nuclei_tall_elongation_factor : float
        v1.6.4, 3D. Size-aware split (`split_tall_nuclei`): a label more
        elongated in Z than ``nuclei_tall_elongation_factor`` x the image's
        typical elongation (and bigger than ``nuclei_tall_volume_factor`` x
        its typical volume) is cut into stacked nuclei; normal labels, and big
        ones that are not taller, are left alone. Runs after the saddle split,
        before the edge refinement. v1.6.4: the elongation is the switch
        (default 0 = off; 1.2 matched the ground truths); the volume factor
        hardly matters (1.15-1.6 gave the same labels) and defaults to 1.15.
        Either at 0 = off (a v1.6.4 setting with the volume factor 0 stays off).
    nuclei_rescue_contrast, nuclei_rescue_grow_fraction : float
        v1.6.4, 3D. Add back nuclei the threshold missed
        (`rescue_missed_nuclei`): nucleus-sized bright spots away from every
        label, brighter than ``nuclei_rescue_contrast`` x the median nucleus
        peak (above background), each grown to ``nuclei_rescue_grow_fraction``
        of its own peak. Runs before the edge refinement. Default 0 = off.

    Returns
    -------
    im_segmentation_stack : dict
        Dict with key 'Nuclei' mapping to the integer label array (Z, Y, X).
    """
    from skimage.measure import label as skimage_label

    # Physical spacing (for the label expansion) vs the spacing seen by the
    # segmentation (Z shrunk by z_elongation, v1.6.4).
    physical_zyx = (r_zxyz[2], r_zxyz[1], r_zxyz[0])
    z_elongation = float(z_elongation or 1.0)
    if abs(z_elongation - 1.0) > 1e-6 and im_final_stack['Threshold image'].shape[0] > 1:
        r_zxyz = (r_zxyz[0], r_zxyz[1], r_zxyz[2] / z_elongation)
        print(f"Z elongation {z_elongation:g}: nuclei segmented with a Z voxel size of "
              f"{r_zxyz[2]:.3f} um instead of {physical_zyx[0]:.3f} um")
    from skimage import morphology
    from skimage.segmentation import relabel_sequential

    from helpers.notebook_setup_helpers import resolve_stardist_flag

    # Falls back to Cellpose/watershed (with a warning) if StarDist isn't installed.
    trig_stardist = resolve_stardist_flag(trig_stardist)

    r_zX, r_zY, r_zZ = r_zxyz
    im_segmentation_stack = {}
    if nuclei_split_config is None:
        nuclei_split_config = get_nuclei_split_config(profile="balanced")

    def _refine(labels, intensity_img, intensity_img_raw, gamma_levels=refine_gamma_levels):
        if not (iterative_split or merge_undersized):
            return labels
        labels, _ = refine_nuclei_by_size(
            labels,
            intensity_img=intensity_img,
            intensity_img_raw=intensity_img_raw,
            r_zxyz=r_zxyz,
            nuclei_diameter=nuclei_diameter,
            base_split_config=nuclei_split_config,
            size_reference=size_reference,
            merge_undersized=merge_undersized,
            undersize_factor=undersize_factor,
            iterative_split=iterative_split,
            oversize_factor=oversize_factor,
            max_split_iterations=max_split_iterations,
            progress=progress,
            intensity_gamma_levels=gamma_levels,
        )
        print(f"Total nuclei after size refinement: {int(labels.max())}")
        return labels

    # Split gamma of the NUCLEI-channel path (v1.6.3).
    base_gamma = float(split_gamma or 1.0)
    gamma_on = abs(base_gamma - 1.0) > 1e-6 or bool(split_gamma_step)
    level_gammas_abs = [base_gamma + float(split_gamma_step or 0.0) * (lvl + 1) for lvl in range(3)]

    if 'NUCLEI' not in stain_df.index and im_final_stack['Threshold image'].shape[0] == 1:
        # LD-style on a 2D image: union of all channels, segmented in 2D.
        thr = im_final_stack['Threshold image'][0]
        im_out = _segment_nuclei_2d(
            np.any(thr > 0, axis=-1),
            np.max(im_final_stack['Filtered image'][0], axis=-1),
            r_zxyz=r_zxyz,
            nuclei_diameter=nuclei_diameter,
            trig_cellpose=trig_cellpose,
            trig_stardist=trig_stardist,
            size_reference=size_reference,
            iterative_split=iterative_split,
            oversize_factor=oversize_factor,
            max_split_iterations=max_split_iterations,
            merge_undersized=merge_undersized,
            undersize_factor=undersize_factor,
            split_gamma=split_gamma,
            split_gamma_step=split_gamma_step,
            min_piece_fraction=min_piece_fraction,
            progress=progress,
        )
        im_segmentation_stack['Nuclei'] = im_out
        im_segmentation_stack['Cytoplasm'] = np.zeros_like(im_out)
        im_segmentation_stack['PCM'] = np.zeros_like(im_out)
        return im_segmentation_stack

    if 'NUCLEI' not in stain_df.index:
        # LD-style: union all channels.
        if trig_cellpose:
            im_filt = im_final_stack['Filtered image'].copy()
            combined = np.max(im_filt, axis=-1)
            im_out = segment_nuclei_cellpose(
                combined,
                nuclei_diameter=nuclei_diameter,
                voxel_size=(r_zZ, r_zY, r_zX),
            )
            im_out = _refine(
                im_out, combined, np.max(im_final_stack['Denoised image'], axis=-1)
            )
            im_segmentation_stack['Nuclei'] = im_out
            im_segmentation_stack['Cytoplasm'] = np.zeros_like(im_out)
            im_segmentation_stack['PCM'] = np.zeros_like(im_out)
            return im_segmentation_stack

        im_in = im_final_stack['Threshold image'].copy()
        im_thresh = np.zeros(im_in.shape[:3], dtype=bool)
        for c in range(im_in.shape[3]):
            im_thresh = im_thresh | (im_in[:, :, :, c] > 0)

        split_cfg = dict(nuclei_split_config)
        intensity_img = np.max(im_final_stack['Filtered image'], axis=-1)
        intensity_img_raw = np.max(im_final_stack['Denoised image'], axis=-1)
        im_out, debug_info = segment_nuclei_watershed(
            binary_mask=im_thresh,
            r_zX=r_zX,
            r_zY=r_zY,
            r_zZ=r_zZ,
            nuclei_diameter=nuclei_diameter,
            intensity_img=intensity_img,
            intensity_img_raw=intensity_img_raw,
            progress=progress,
            **split_cfg,
        )

        print(f"Total nuclei found: {int(im_out.max())}")
        _display_nuclei_roundness_size_table(debug_info)
        im_out = _refine(im_out, intensity_img, intensity_img_raw)

        im_segmentation_stack['Nuclei'] = im_out
        im_segmentation_stack['Cytoplasm'] = np.zeros_like(im_out)
        im_segmentation_stack['PCM'] = np.zeros_like(im_out)
        return im_segmentation_stack

    im_in = im_final_stack['Threshold image'].copy()
    split_cfg = dict(nuclei_split_config)

    for c in _progress_iter(range(im_in.shape[3]), progress, desc='Step 17 - Segment Nuclei'):
        if stain_complete_df.index[c] != 'NUCLEI':
            continue

        if im_in.shape[0] == 1:
            # 2D image: watershed / Cellpose / StarDist and size refinement in
            # the image plane (the 3D erosion-based split would erase it).
            im_segmentation_stack['Nuclei'] = _segment_nuclei_2d(
                im_in[0, :, :, c] > 0,
                im_final_stack['Filtered image'][0, :, :, c],
                r_zxyz=r_zxyz,
                nuclei_diameter=nuclei_diameter,
                trig_cellpose=trig_cellpose,
                trig_stardist=trig_stardist,
                size_reference=size_reference,
                iterative_split=iterative_split,
                oversize_factor=oversize_factor,
                max_split_iterations=max_split_iterations,
                merge_undersized=merge_undersized,
                undersize_factor=undersize_factor,
                split_gamma=split_gamma,
                split_gamma_step=split_gamma_step,
                min_piece_fraction=min_piece_fraction,
                progress=progress,
            )
            continue

        # Intensity used to split touching nuclei: gamma-adjusted for the
        # watershed; Cellpose/StarDist get the original image, and their
        # refinement uses the absolute level gammas instead.
        deep_learning = trig_cellpose or trig_stardist
        applied_gamma = 1.0 if deep_learning else base_gamma
        split_intensity = apply_split_gamma(im_final_stack['Filtered image'][:, :, :, c], applied_gamma)
        split_intensity_raw = apply_split_gamma(im_final_stack['Denoised image'][:, :, :, c], applied_gamma)
        gamma_levels = refine_gamma_levels
        if gamma_levels is None and gamma_on:
            gamma_levels = [g / applied_gamma for g in level_gammas_abs]
            print(("Split gamma: " + (f"{base_gamma:g} on the nuclei intensity used for the split"
                                      if not deep_learning else "not applied to the deep-learning input"))
                  + "; re-split levels: " + ", ".join(f"{g:g}" for g in level_gammas_abs))

        if trig_cellpose:
            im_filt = im_final_stack['Filtered image'].copy()
            im_out = segment_nuclei_cellpose(
                im_filt[:, :, :, c],
                nuclei_diameter=nuclei_diameter,
                voxel_size=(r_zZ, r_zY, r_zX),
            )

        elif trig_stardist:
            im_filt = im_final_stack['Filtered image'].copy()
            transl = stardist3d_from_2d(
                img_3d=im_filt[:, :, :, c],
                nucleus_radius=nuclei_diameter / 2.0,
                voxel_size=(r_zZ, r_zY, r_zX),
                progress=progress,
            )
            im_mask = transl > 0
            im_mask = morphology.erosion(
                im_mask, footprint=np.ones((2, 2, 2))
            ).astype(im_mask.dtype)
            im_out, _ = skimage_label((transl * im_mask) > 0, return_num=True)
            print(f"Total nuclei found: {int(im_out.max())}")

        else:
            binary_mask = im_in[:, :, :, c].astype(bool)
            im_out, debug_info = segment_nuclei_watershed(
                binary_mask=binary_mask,
                r_zX=r_zX,
                r_zY=r_zY,
                r_zZ=r_zZ,
                nuclei_diameter=nuclei_diameter,
                intensity_img=split_intensity,
                intensity_img_raw=split_intensity_raw,
                progress=progress,
                **split_cfg,
            )

            print(f"Total nuclei found: {int(im_out.max())}")
            _display_nuclei_roundness_size_table(debug_info)

            # Second merge round, limited to the pieces of each split island.
            if min_piece_fraction:
                min_vox = max(1.0, float(min_piece_fraction) * _sphere_volume_um3(nuclei_diameter)
                              / (r_zX * r_zY * r_zZ))
                islands, _ = ndi.label(binary_mask)
                im_out, n_absorbed = absorb_small_split_pieces(im_out, islands, min_vox)
                del islands
                if n_absorbed:
                    im_out, _, _ = relabel_sequential(im_out)
                    print(f"Split pieces smaller than {min_piece_fraction:g}x a nucleus "
                          f"({min_vox:.0f} voxels) merged back into a neighbouring piece: "
                          f"{n_absorbed} -> {int(im_out.max())} nuclei")

        im_out = _refine(im_out, split_intensity, split_intensity_raw, gamma_levels)
        del split_intensity, split_intensity_raw
        im_out = postprocess_nuclei_labels(
            im_out, im_final_stack['Denoised image'][:, :, :, c], physical_zyx, nuclei_diameter,
            flare_trim_factor=nuclei_flare_trim_factor, saddle_dip=nuclei_saddle_dip,
            tall_volume_factor=nuclei_tall_volume_factor,
            tall_elongation_factor=nuclei_tall_elongation_factor,
            rescue_contrast=nuclei_rescue_contrast, rescue_grow_fraction=nuclei_rescue_grow_fraction,
            edge_fraction=nuclei_edge_fraction, edge_margin_um=nuclei_edge_margin_um,
            expand_um=nuclei_expand_um)
        im_segmentation_stack['Nuclei'] = im_out

    return im_segmentation_stack


def detect_aggregates(im_segmentation_stack, stain_df, cell_diameter, r_zxyz,
                      aggregate_distance_factor=0.05):
    """Detect cell aggregates: cells closer than ``aggregate_distance_factor``
    x ``cell_diameter`` um to each other belong to the same aggregate.

    Each cell is grown by half that distance (in um, with the physical voxel
    size, so it does not change with ``scale_factor``); cells whose grown
    regions touch form one aggregate. Only runs when a NUCLEI or CYTOPLASM
    channel is present (i.e. not LD mode).

    Returns
    -------
    im_segmentation_stack : dict
        Updated with ``'Aggregates'`` label array (or unchanged if not applicable).
    """
    from scipy.ndimage import label as _label

    if ('NUCLEI' in stain_df.index) | ('CYTOPLASM' in stain_df.index):
        r_zX, r_zY, r_zZ = r_zxyz
        gap_um = float(aggregate_distance_factor) * float(cell_diameter)
        grown = expand_labels_um(im_segmentation_stack['Cytoplasm'], gap_um / 2.0,
                                 (r_zZ, r_zY, r_zX))
        im_out, _ = _label(grown > 0)
        print(f"Aggregates: cells closer than {gap_um:.2f} um "
              f"({aggregate_distance_factor:g} x cell diameter {cell_diameter:g} um)")
        im_segmentation_stack = dict(im_segmentation_stack)
        im_segmentation_stack['Aggregates'] = im_out * (im_segmentation_stack['Cytoplasm'] > 0)
    return im_segmentation_stack




# ── Cell-body completion (v1.6.41, tuned on Cellpose cell bodies) ─────────────
# The stain mask alone leaves the dim parts of a cell body out (~22-42% of
# the Cellpose cell volume on the validation images): these optional steps of
# `segment_cytoplasm` give them back. All off by default (results of v1.6.4).

def close_mask_xy(mask, radius_px):
    """Binary closing in each XY slice (disk of ``radius_px``): bridges the
    small dim gaps of the stain inside a cell."""
    from skimage.morphology import disk
    r = int(round(radius_px))
    if r < 1:
        return mask
    se = disk(r)
    out = np.empty_like(mask)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_closing(mask[z], structure=se, border_value=0) | mask[z]
    return out


def fill_cell_holes(cells):
    """Give every cell the empty holes it encloses in each XY slice."""
    cells = cells.copy()
    for k, sl in enumerate(ndi.find_objects(cells), start=1):
        if sl is None:
            continue
        sub = cells[sl]
        for z in range(sub.shape[0]):
            own = sub[z] == k
            if own.any():
                holes = ndi.binary_fill_holes(own) & (sub[z] == 0)
                sub[z][holes] = k
    return cells


def grow_cells_xy(cells, distance_px):
    """Grow every cell by ``distance_px`` into the empty voxels, in each XY
    slice, without overlapping its neighbours (``expand_labels``).

    A growth below one pixel (but above zero) grows by one pixel: with a coarse
    XY voxel the requested growth in um would otherwise silently do nothing
    (e.g. 0.62 um on 0.66 um pixels: GT1 mean cell IoU 0.330 instead of 0.473)."""
    from skimage.segmentation import expand_labels
    if distance_px <= 0:
        return cells
    distance_px = max(float(distance_px), 1.0)
    return np.stack([expand_labels(cells[z], distance=distance_px)
                     for z in range(cells.shape[0])])


def _gradient_watershed_elevation(
    intensity_imgs,
    combined_mask,
    spacing,
    smooth_sigma=1.0,
    w_distance=1.0,
    w_intensity=1.0,
    w_gradient=1.0,
    ridge_imgs=None,
):
    """Build a watershed elevation surface that splits touching cells where the
    marker signal itself dips or drops steeply, not just where two nuclei are
    geometrically equidistant.

    Combines three normalized (0-1 within ``combined_mask``) terms, each
    raising elevation (i.e. acting as a watershed ridge/boundary) where it
    signals a cell edge:

    - ``1 - distance``: distance-from-background (kept from the original
      approach so flooding still follows the medial axis and stops at the
      true mask edge).
    - ``1 - intensity envelope``: voxels dim in *every* marker become ridges
      even away from the mask edge (e.g. a faint saddle between two bright
      cells). The envelope is the per-voxel max across markers — a voxel
      counts as cytoplasm if any one marker says so.
    - ``combined gradient magnitude``: voxels where the signal is changing
      fast in any marker become ridges, regardless of absolute brightness.
      When several markers are supplied this is a per-voxel quadrature sum
      (``sqrt(sum(grad_c**2))``) of each marker's own gradient, computed
      *before* the markers are merged. This matters: if marker A fades out
      while marker B rises across the same cell-cell boundary (common when
      different markers dominate different neighbor cells), their envelope
      (max) stays flat through the handoff and hides the transition, but
      each channel's own gradient is still large there, so summing gradients
      in quadrature keeps the boundary visible.

    Parameters
    ----------
    intensity_imgs : ndarray (Z, Y, X) or sequence of ndarray (Z, Y, X)
        One or more per-channel marker intensities (e.g. 'Filtered image'
        slices), NOT the binary threshold mask — the whole point is to use
        the graded signal. Pass every cytoplasm-associated marker channel
        that contributed to ``combined_mask`` so the gradient term reflects
        all of them, not just one.
    combined_mask : ndarray (Z, Y, X), bool
        Region watershed is allowed to flood (thresholded signal ∪ nuclei).
    spacing : tuple of float (Z, Y, X)
        Physical voxel size, used for the distance transform.
    smooth_sigma : float
        Gaussian smoothing applied before measuring intensity/gradient, so the
        split follows the marker's underlying trend rather than pixel noise.
    w_distance, w_intensity, w_gradient : float
        Relative weight of each term above. Set a weight to 0 to disable it.
    ridge_imgs : sequence of ndarray (Z, Y, X), optional
        Membrane marker intensities. A membrane is bright exactly *at* the
        border between two cells, so these add a ``w_intensity``-weighted
        ridge where they are bright (the opposite of the intracellular
        markers in *intensity_imgs*, which are dim at the border).

    Returns
    -------
    elevation : ndarray (Z, Y, X), float32
    """
    if isinstance(intensity_imgs, np.ndarray):
        intensity_imgs = [intensity_imgs]
    intensity_imgs = list(intensity_imgs)
    ridge_imgs = list(ridge_imgs or [])

    sigma = max(float(smooth_sigma), 1e-3)
    distance = ndi.distance_transform_edt(combined_mask, sampling=spacing)

    def _norm_in_mask(arr):
        values = arr[combined_mask]
        if values.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    intensity_envelope = np.zeros_like(distance, dtype=np.float32)
    gradient_sq_sum = np.zeros_like(distance, dtype=np.float32)
    for img in intensity_imgs:
        img = img.astype(np.float32)
        smoothed = ndi.gaussian_filter(img, sigma=sigma)
        grad_mag = ndi.gaussian_gradient_magnitude(img, sigma=sigma)
        intensity_envelope = np.maximum(intensity_envelope, _norm_in_mask(smoothed))
        gradient_sq_sum += grad_mag ** 2

    ridge_envelope = np.zeros_like(distance, dtype=np.float32)
    for img in ridge_imgs:
        smoothed = ndi.gaussian_filter(img.astype(np.float32), sigma=sigma)
        ridge_envelope = np.maximum(ridge_envelope, _norm_in_mask(smoothed))

    combined_gradient = np.sqrt(gradient_sq_sum)
    gradient_norm = _norm_in_mask(combined_gradient)

    # Without intracellular markers (membrane only) the envelope is empty and
    # the dimness term would be a constant: leave it out.
    intensity_term = (1.0 - intensity_envelope) if intensity_imgs else 0.0
    elevation = (
        w_distance * (1.0 - _norm_in_mask(distance))
        + w_intensity * intensity_term
        + w_intensity * ridge_envelope
        + w_gradient * gradient_norm
    )
    return elevation.astype(np.float32)


def segment_cytoplasm(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    cyto_markers,
    cyto_factor,
    nuclei_diameter,
    r_zxyz,
    cell_diameter=None,
    trig_cellpose_cyto=False,
    split_by_intensity_gradient=True,
    gradient_smooth_sigma=1.0,
    distance_weight=1.0,
    intensity_weight=1.0,
    gradient_weight=1.0,
    membrane_markers=None,
    drop_bare_cells=True,
    progress=None,
    split_gamma=1.0,
    cyto_threshold=None,
    membrane_close_fraction=0.03,
    membrane_fill_area_factor=1.5,
    mask_close_um=0.0,
    fill_holes_in_cells=False,
    grow_um=0.0,
):
    """
    Segment the cytoplasm, one cell per nucleus.

    The cytoplasm is the union of the stains and nothing more
    (`build_cytoplasm_mask`):

    - the intracellular stains: the CYTOPLASM channel (if any) and the
      ``cyto_markers`` not ticked as membrane, as thresholded in Cell 15
      ('Threshold image');
    - the membrane stains (``membrane_markers``), thresholded the same way,
      with their rings closed and filled (`_fill_membrane_rings`).

    These voxels (plus the nuclei) are assigned to the nuclei by watershed;
    nothing is grown beyond them. A cell with no stained cytoplasm (its
    cytoplasm is only its nucleus) is excluded (``drop_bare_cells``): the
    cells with cytoplasm are relabelled 1..K in 'Nuclei' and 'Cytoplasm',
    the excluded nuclei stay in 'Nuclei' as K+1..N with no cytoplasm, and
    the marker statistics only cover cells 1..K. With ``split_by_intensity_gradient=True`` (default)
    touching cells are split where the intracellular signal dips or drops
    steeply and where a membrane is bright, not just at the geometric
    midpoint between their nuclei (see ``_gradient_watershed_elevation``).

    With ``trig_cellpose_cyto`` (needs a CYTOPLASM channel) the cells are
    instead shaped by Cellpose 3D from the CYTOPLASM channel. With no
    CYTOPLASM channel and no ticked marker there is no stain to follow, and
    the cells are grown from the nuclei (``cyto_factor``).

    Parameters
    ----------
    im_final_stack : dict
        Must contain 'Threshold image' and (when
        ``split_by_intensity_gradient`` or ``trig_cellpose_cyto`` is True)
        'Filtered image'.
    im_segmentation_stack : dict
        Must contain 'Nuclei'.
    stain_df, stain_complete_df : DataFrame
        Staining metadata.
    cyto_markers : list of str
        Markers ticked as intracellular or membrane.
    cyto_factor : float
        Growth factor, only used when there is no stain at all to follow
        (and for the Cellpose gap-fill).
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes.
    cell_diameter : float, optional
        Approximate whole-cell diameter in micrometers: sets the largest
        area a membrane ring is filled over (default ``cyto_factor`` x
        ``nuclei_diameter``). Required when ``trig_cellpose_cyto`` is True.
    trig_cellpose_cyto : bool
        If True and a CYTOPLASM channel is present, shape cells with Cellpose
        3D instead.
    split_by_intensity_gradient : bool
        If True (default), split touching cells using marker
        intensity/gradient, not just geometric distance (see above).
    gradient_smooth_sigma : float
        Smoothing applied before measuring intensity/gradient for the split.
    distance_weight, intensity_weight, gradient_weight : float
        Relative weights of the three elevation terms (geometric distance,
        absolute dimness, and steepness of intensity drop).
    membrane_markers : list of str, optional
        Markers ticked as membrane: filled into the cytoplasm, and their
        bright signal marks the border between cells (a ridge) in the
        gradient-aware watershed.
    drop_bare_cells : bool
        If True (default), exclude cells with no stained cytoplasm beyond the
        nucleus from the marker analysis (nuclei kept, numbered last). If
        False, keep them as nucleus-only cells.
    progress : callable, optional
        Progress wrapper.
    split_gamma : float
        Gamma applied to the intracellular and membrane intensities that
        decide the border between touching cells (`apply_split_gamma`): > 1
        makes the dim zone between two cells darker faster than the
        cytoplasm, and keeps a bright membrane bright against its dimmer
        surroundings, so the border is more distinct. Only the split uses
        it, not the cytoplasm mask. Default 1.0 = off (nuclei notebook
        v1.6.3 setting).
    cyto_threshold : dict, optional
        Threshold the cytoplasm / membrane channels again, with their own
        settings, from the 'Equalized image' (keys ``threshold_method``,
        ``global_threshold_weight``, ``sauvola_weight``, ``window_um`` = the
        Sauvola window, the cell diameter in Cell 15). Default None: the
        'Threshold image' of Cell 15, shared with the nuclei.
    membrane_close_fraction, membrane_fill_area_factor : float
        Membrane rings: gaps closed over this fraction of a cell diameter
        (0.03) and holes filled up to this many cell areas (1.5).
    mask_close_um : float
        Close the cytoplasm mask in each XY slice over this radius (um), so
        dim gaps inside a cell are kept. Default 0 = off.
    fill_holes_in_cells : bool
        Give every cell the holes it encloses in each XY slice. Default off.
    grow_um : float
        Grow every cell by this distance (um) in each XY slice, without
        overlapping its neighbours: the dim rim of the cell body below the
        stain threshold. Default 0 = off. (v1.6.41 settings: tuned on Cellpose
        cell bodies of the manual nuclei, ``validation/tune_cyto.py``.)

    Returns
    -------
    im_segmentation_stack : dict
        Updated dict with 'Cytoplasm' key added.
    stain_complete_df : DataFrame
        Updated dataframe (CYTOPLASM row added if it was absent).
    """
    r_zX, r_zY, r_zZ = r_zxyz

    if 'Nuclei' not in im_segmentation_stack:
        raise RuntimeError(
            "Run nuclei segmentation first so im_segmentation_stack contains 'Nuclei'."
        )

    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)
            or len(cyto_markers) > 0 or cyto_factor > 1):
        return im_segmentation_stack, stain_complete_df

    im_in = im_final_stack['Threshold image']
    nuclei_labels = im_segmentation_stack['Nuclei']
    n_channels = im_in.shape[3]
    channel_index = list(stain_complete_df.index[:n_channels])
    has_cyto_channel = 'CYTOPLASM' in stain_df.index
    intra_channels, membrane_channels = _cytoplasm_stain_channels(
        stain_complete_df, n_channels, cyto_markers, membrane_markers,
    )
    use_cellpose = has_cyto_channel and trig_cellpose_cyto
    from_stains = not use_cellpose and bool(intra_channels or membrane_channels)

    if use_cellpose:
        if cell_diameter is None:
            raise ValueError("cell_diameter is required when trig_cellpose_cyto=True.")
        c = channel_index.index('CYTOPLASM')
        im_out = segment_cytoplasm_cellpose(
            im_final_stack['Filtered image'][:, :, :, c],
            cell_diameter=cell_diameter,
            voxel_size=(r_zZ, r_zY, r_zX),
            nuclei_labels=nuclei_labels,
        )
        print("Cytoplasm shaped from CYTOPLASM channel via Cellpose 3D")

    elif from_stains:
        diameter = cell_diameter if cell_diameter is not None else cyto_factor * nuclei_diameter
        cell_px = diameter / float(np.mean([r_zX, r_zY]))
        if cyto_threshold:
            im_in = _rethreshold_cytoplasm_channels(
                im_final_stack['Equalized image'], stain_complete_df,
                sorted(set(intra_channels) | set(membrane_channels)), nuclei_diameter,
                diameter, (r_zX, r_zY, r_zZ), cyto_threshold)
        cyto_mask = build_cytoplasm_mask(
            im_in, intra_channels, membrane_channels,
            max_cell_area=membrane_fill_area_factor * np.pi * (cell_px / 2) ** 2,
            # Membrane gaps closed over 3% of a cell diameter, in voxels.
            close_radius=max(1, int(round(membrane_close_fraction * cell_px))),
        )
        if mask_close_um > 0:
            cyto_mask = close_mask_xy(cyto_mask, mask_close_um / float(np.mean([r_zX, r_zY])))
        combined_mask = cyto_mask | (nuclei_labels > 0)
        if split_by_intensity_gradient:
            filt = im_final_stack['Filtered image']
            if abs(float(split_gamma or 1.0) - 1.0) > 1e-6:
                print(f"Split gamma {split_gamma:g} on the cytoplasm intensities used "
                      "to split touching cells")
            elevation = _gradient_watershed_elevation(
                [apply_split_gamma(filt[:, :, :, c], split_gamma) for c in intra_channels],
                combined_mask,
                spacing=(r_zZ, r_zY, r_zX),
                smooth_sigma=gradient_smooth_sigma,
                w_distance=distance_weight,
                w_intensity=intensity_weight,
                w_gradient=gradient_weight,
                ridge_imgs=[apply_split_gamma(filt[:, :, :, c], split_gamma)
                            for c in membrane_channels],
            )
        else:
            elevation = -ndi.distance_transform_edt(
                combined_mask, sampling=[r_zZ, r_zY, r_zX]
            )
        im_out = watershed(elevation, nuclei_labels, mask=combined_mask)
        if fill_holes_in_cells:
            im_out = fill_cell_holes(np.asarray(im_out, dtype=np.int32))
        if grow_um > 0:
            im_out = grow_cells_xy(np.asarray(im_out, dtype=np.int32),
                                   grow_um / float(np.mean([r_zX, r_zY])))
        completion = [f"{k} {v:g}" for k, v in (("mask closed over um", mask_close_um),
                                                 ("cells grown by um", grow_um)) if v > 0]
        if fill_holes_in_cells:
            completion.append("holes in cells filled")
        if completion:
            print("Cell bodies completed: " + ", ".join(completion))

        def _names(channels):
            return ', '.join(str(stain_complete_df.loc[channel_index[c], 'Marker']) for c in channels)

        parts = []
        if intra_channels:
            parts.append(f"intracellular: {_names(intra_channels)}")
        if membrane_channels:
            parts.append(f"membrane, rings filled: {_names(membrane_channels)}")
        print(f"Cytoplasm = union of the thresholded stains ({'; '.join(parts)}) - "
              f"{100 * cyto_mask.mean():.1f}% of the volume - assigned to the nuclei "
              f"({'gradient-aware' if split_by_intensity_gradient else 'nearest-nucleus'} "
              f"watershed), {f'cells grown by {grow_um:g} um' if grow_um > 0 else 'no growth'}")
        del cyto_mask, combined_mask, elevation

    else:
        im_out = grow_labels(nuclei_labels, cyto_factor)
        print("No CYTOPLASM channel or intracellular/membrane marker: cytoplasm "
              f"grown from nuclei labels (factor={cyto_factor})")

    if not has_cyto_channel:
        stain_complete_df = stain_complete_df.copy()
        stain_complete_df.loc['CYTOPLASM'] = ['', '', '', 0, 255, 1.0]

    n_nuclei = np.count_nonzero(np.unique(nuclei_labels))
    if from_stains:
        # A cell whose cytoplasm is only its nucleus has no stained cytoplasm.
        with_cyto = np.unique(im_out[(im_out > 0) & (nuclei_labels == 0)])
        bare = n_nuclei - with_cyto.size
        if bare and drop_bare_cells:
            # Cells with cytoplasm become 1..K, in both Nuclei and Cytoplasm.
            # The bare nuclei are kept in Nuclei as K+1..N but get no
            # cytoplasm, so they are left out of PCM and the marker statistics
            # (which only run over cells 1..K, see `_n_cells`).
            present = np.unique(nuclei_labels)
            bare_ids = np.setdiff1d(present[present > 0], with_cyto)
            n_cells = with_cyto.size
            lut = np.zeros(int(nuclei_labels.max()) + 1, dtype=np.int32)
            lut[with_cyto] = np.arange(1, n_cells + 1, dtype=np.int32)
            lut[bare_ids] = np.arange(n_cells + 1, n_cells + bare_ids.size + 1, dtype=np.int32)
            nuclei_labels = lut[nuclei_labels]
            lut[bare_ids] = 0
            im_out = lut[im_out]
            print(f"{bare} cell(s) have no stained cytoplasm beyond the nucleus: "
                  f"excluded from the marker analysis. Cells with cytoplasm are "
                  f"labels 1-{n_cells}, excluded nuclei {n_cells + 1}-{n_cells + bare}")
        elif bare:
            print(f"{bare} cell(s) have no stained cytoplasm beyond the nucleus "
                  f"(left as nucleus only, no growth)")
    elif use_cellpose:
        # Cellpose can miss a nucleus: give it a grown cytoplasm instead.
        filled = 0
        for label_id in _progress_iter(
            range(1, int(nuclei_labels.max()) + 1), progress,
            desc='Step 18C - Gap-fill Cytoplasm', leave=False,
        ):
            nuc_mask = nuclei_labels == label_id
            if not np.any(nuc_mask):
                continue
            cyto_mask = im_out == label_id
            if np.any(cyto_mask & ~nuc_mask):
                continue
            im_out[cyto_mask] = 0
            single = np.zeros_like(nuclei_labels, dtype=np.int32)
            single[nuc_mask] = label_id
            grown = grow_labels(single, cyto_factor)
            im_out[(grown == label_id) & (im_out == 0)] = label_id
            filled += 1
        if filled > 0:
            print(f"Cytoplasm gap-filled for {filled} nuclei by label-grow (factor={cyto_factor})")

    im_segmentation_stack = dict(im_segmentation_stack)
    im_segmentation_stack['Nuclei'] = nuclei_labels
    im_segmentation_stack['Cytoplasm'] = np.asarray(im_out, dtype=np.int32).copy()
    return im_segmentation_stack, stain_complete_df


def segment_pcm(
    im_segmentation_stack,
    stain_df,
    cell_diameter,
    r_zxyz,
    pcm_thickness_factor=0.05,
):
    """
    Build the pericellular matrix (PCM) label volume: a shell around each
    cell's cytoplasm, ``pcm_thickness_factor`` x ``cell_diameter`` um thick.

    The thickness is set in um and grown with the physical voxel size, so it
    does not change with ``scale_factor``.

    Parameters
    ----------
    im_segmentation_stack : dict
        Must contain 'Nuclei' and 'Cytoplasm'.
    stain_df : DataFrame
        Staining metadata.
    cell_diameter : float
        Approximate whole-cell diameter in um (the thickness reference).
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in um.
    pcm_thickness_factor : float
        PCM thickness as a fraction of ``cell_diameter`` (0.05 = 5%).

    Returns
    -------
    im_segmentation_stack : dict
        Updated dict with 'PCM' key added.
    """
    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)):
        return im_segmentation_stack

    r_zX, r_zY, r_zZ = r_zxyz
    thickness_um = float(pcm_thickness_factor) * float(cell_diameter)
    cytoplasm = im_segmentation_stack['Cytoplasm']
    grown = expand_labels_um(cytoplasm, thickness_um, (r_zZ, r_zY, r_zX))
    im_out = np.where(cytoplasm > 0, 0, grown).astype(cytoplasm.dtype)
    print(f"PCM: {thickness_um:.2f} um shell around each cell "
          f"({pcm_thickness_factor:g} x cell diameter {cell_diameter:g} um)")

    im_segmentation_stack = dict(im_segmentation_stack)
    im_segmentation_stack['PCM'] = im_out
    return im_segmentation_stack


def assign_channel_labels(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    progress=None,
):
    """
    Multiply each non-nuclear/non-cytoplasm threshold channel by the combined
    cytoplasm+PCM mask to assign cell labels to marker channels.

    Parameters
    ----------
    im_final_stack : dict
        Must contain 'Threshold image'.
    im_segmentation_stack : dict
        Must contain 'Cytoplasm' and 'PCM'.
    stain_df : DataFrame
        Staining metadata (original, without CYTOPLASM row).
    progress : callable, optional
        Progress wrapper.

    Returns
    -------
    im_segmentation_stack : dict
        Updated dict with per-marker keys added.
    """
    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)):
        # LD mode: label each channel by masking the Nuclei label array with
        # the per-channel threshold so LIVE and DEAD cells are separately identified.
        im_in = im_final_stack['Threshold image'].copy()
        im_segmentation_stack = dict(im_segmentation_stack)
        for c in _progress_iter(range(im_in.shape[3]), progress, desc='Step 20 - Assign Labels To Channels'):
            cond = stain_df.index[c]
            im_segmentation_stack[cond] = (im_in[:, :, :, c] > 0) * im_segmentation_stack['Nuclei']
        return im_segmentation_stack

    im_in = im_final_stack['Threshold image'].copy()
    im_segmentation_stack = dict(im_segmentation_stack)

    for c in _progress_iter(range(im_in.shape[3]), progress, desc='Step 20 - Assign Labels To Channels'):
        cond = stain_df.index[c]
        if cond in ('NUCLEI', 'CYTOPLASM', 'PCM'):
            continue
        cyto_pcm = im_segmentation_stack['Cytoplasm'] + im_segmentation_stack['PCM']
        im_segmentation_stack[cond] = im_in[:, :, :, c] * cyto_pcm
        im_segmentation_stack[cond + '_cyto'] = im_in[:, :, :, c] * im_segmentation_stack['Cytoplasm']
        im_segmentation_stack[cond + '_PCM'] = im_in[:, :, :, c] * im_segmentation_stack['PCM']

    return im_segmentation_stack


def view_processing_results(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    r_xyz,
    r_zxyz,
    napari_module,
    progress=None,
):
    """
    Open two napari viewers showing processing pipeline layers and segmentation results.

    Viewer 0 shows per-channel images at each pipeline stage (original, zoomed,
    denoised, corrected, filtered, equalized), plus a per-channel "Thresh
    islands" labels layer (connected components of the threshold mask) so
    nuclei/marker regions missed or fragmented by thresholding are visible.
    Only the original image and the processing steps that were run (Cells
    9-14 can be skipped) are shown; the last of them (the threshold's input)
    and the threshold-island layers start visible.
    Viewer 1 (only when NUCLEI or CYTOPLASM is present) shows label overlays for
    nuclei, cytoplasm, PCM, and marker channels; the marker labels start hidden.

    Parameters
    ----------
    im_final_stack : dict
        Processing stage arrays (Z, Y, X, C).
    im_segmentation_stack : dict
        Segmentation label arrays.
    stain_df : DataFrame
        Original staining metadata.
    stain_complete_df : DataFrame
        Extended staining metadata (may include CYTOPLASM row).
    r_xyz : tuple of float (r_X, r_Y, r_Z)
        Original voxel sizes in micrometers.
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in micrometers.
    napari_module : module
        The napari module (pass `napari`).
    progress : callable, optional
        Progress wrapper.

    Returns
    -------
    viewer_0 : napari.Viewer
        Channel pipeline viewer.
    viewer_1 : napari.Viewer or None
        Segmentation viewer (None if no NUCLEI/CYTOPLASM channel).
    """
    r_X, r_Y, r_Z = r_xyz
    r_zX, r_zY, r_zZ = r_zxyz
    scale_zoom = (r_zZ, r_zY, r_zX)
    im_thr = im_final_stack['Threshold image']

    _close_all_napari_viewers(napari_module)
    viewer_0 = napari_module.Viewer(title="Post-processing Channels")
    stages = [
        ('Original image',  'ORIGINAL',  [r_Z, r_Y, r_X]),
        ('Zoomed image',    'ZOOMED',    scale_zoom),
        ('Denoised image',  'DENOISED',  scale_zoom),
        ('Adjusted image',  'CORRECTED', scale_zoom),
        ('Filtered image',  'FILTERED',  scale_zoom),
        ('Equalized image', 'EQ',        scale_zoom),
    ]
    # Stages not read by any cell after this one (Original/Zoomed/Denoised/
    # Adjusted) are dropped from im_final_stack right after their layers are
    # added — napari keeps its own reference, so nothing disappears from the
    # viewer, but the dict stops double-holding these large arrays.
    _stages_freed_after_view = ('Original image', 'Zoomed image', 'Denoised image', 'Adjusted image')
    # Only the original image and the processing steps actually run (Cells
    # 9-14 can be skipped; a skipped stage only holds its input again).
    steps_run = im_final_stack.get(_STEPS_RUN_KEY)
    if steps_run is not None:
        shown = [s for s in stages if s[0] == 'Original image' or s[0] in steps_run]
        if 'Normalized image' in steps_run and 'Zoomed image' not in steps_run:
            # Normalized but not resampled: the 'Zoomed image' holds the normalized image
            shown.insert(1, ('Zoomed image', 'NORMALIZED', scale_zoom))
    else:
        shown = stages
    # The last stage shown (the threshold's input) starts visible, with the
    # threshold islands below; the earlier stages are hidden but can be toggled on.
    visible_key = shown[-1][0]
    for stage_key, stage_prefix, scale in _progress_iter(shown, progress, desc='Step 22A - Add Layers To Viewer 0'):
        for c in range(im_thr.shape[3]):
            idx = stain_complete_df.index[c]
            marker = stain_complete_df.loc[idx, 'Marker']
            color = _napari_colormap(stain_complete_df['Color'].iloc[c])
            viewer_0.add_image(
                im_final_stack[stage_key][:, :, :, c],
                name=f'{stage_prefix} {idx} ({marker})', colormap=color,
                blending='additive', scale=scale,
                visible=(stage_key == visible_key),
            )
        if stage_key in _stages_freed_after_view:
            del im_final_stack[stage_key]
            gc.collect()
    for stage_key in _stages_freed_after_view:  # stages of skipped steps (not shown)
        im_final_stack.pop(stage_key, None)

    # Connected-component "islands" of the per-channel threshold mask, so
    # under-segmented/missed nuclei (broken-up or absent islands) are easy to
    # spot against the intensity layers above.
    for c in _progress_iter(range(im_thr.shape[3]), progress, desc='Step 22A2 - Add Threshold Islands To Viewer 0'):
        idx = stain_complete_df.index[c]
        marker = stain_complete_df.loc[idx, 'Marker']
        islands, _ = ndi.label(im_thr[:, :, :, c] > 0)
        viewer_0.add_labels(
            islands.astype(np.int32),
            name=f'Thresh islands {idx} ({marker})', blending='additive', scale=scale_zoom,
        )
    # 'Threshold image' is also not read by any later cell; free it (and the
    # local `im_thr` reference used above) the same way.
    del im_final_stack['Threshold image']
    del im_thr
    gc.collect()
    viewer_0.scale_bar.visible = True
    for layer in viewer_0.layers:
        layer.units = ('um', 'um', 'um')

    viewer_1 = None
    if ('NUCLEI' in stain_complete_df.index) or ('CYTOPLASM' in stain_complete_df.index):
        viewer_1 = napari_module.Viewer(title="Segmentation and labeling stacks")  # viewer_0 already opened above — no extra close needed
        eq_img = im_final_stack['Equalized image']
        for c in range(eq_img.shape[3]):
            idx = stain_df.index[c]
            marker = stain_df.loc[idx, 'Marker']
            color = _napari_colormap(stain_df['Color'].iloc[c])
            viewer_1.add_image(
                eq_img[:, :, :, c],
                name=f'EQ {idx} ({marker})', colormap=color,
                blending='additive', scale=scale_zoom,
            )
        for c in _progress_iter(range(len(stain_complete_df.index)), progress, desc='Step 22B - Add Labels To Viewer 1'):
            idx = stain_complete_df.index[c]
            marker = stain_complete_df.loc[idx, 'Marker']
            if idx == 'NUCLEI' and 'NUCLEI' not in stain_df.index:
                # Virtual NUCLEI entry added for LD mode — skip it in the viewer.
                continue
            elif idx == 'NUCLEI':
                viewer_1.add_labels(
                    im_segmentation_stack['Nuclei'].astype(np.int32),
                    name=f'{idx} ({marker})', blending='additive', scale=scale_zoom,
                )
            elif idx == 'CYTOPLASM':
                viewer_1.add_labels(
                    im_segmentation_stack['Cytoplasm'].astype(np.int32),
                    name=f'{idx} ({marker})', blending='additive', scale=scale_zoom,
                )
                viewer_1.add_labels(
                    im_segmentation_stack['PCM'].astype(np.int32),
                    name='PCM', blending='additive', scale=scale_zoom,
                )
            elif idx not in ('PCM',):
                # Marker labels start hidden; EQ, nuclei, cytoplasm, PCM and
                # aggregates stay visible.
                viewer_1.add_labels(
                    im_segmentation_stack[idx].astype(np.int32),
                    name=f'{idx} ({marker})', blending='additive', scale=scale_zoom,
                    visible=False,
                )
        if 'Cytoplasm' in im_segmentation_stack and 'CYTOPLASM' not in stain_df.index:
            viewer_1.add_labels(
                im_segmentation_stack['Cytoplasm'].astype(np.int32),
                name='CYTOPLASM (computed)', blending='additive', scale=scale_zoom,
            )
        if 'Aggregates' in im_segmentation_stack:
            viewer_1.add_labels(
                im_segmentation_stack['Aggregates'].astype(np.int32),
                name='Aggregates', blending='additive', scale=scale_zoom,
            )
        viewer_1.scale_bar.visible = True
        for layer in viewer_1.layers:
            layer.units = ('um', 'um', 'um')

    return viewer_0, viewer_1


def export_nucleus_vtk_crop(
    nuc_label,
    im_segmentation_stack,
    im_final_stack,
    stain_df,
    input_file,
    r_zxyz=(1.0, 1.0, 1.0),
    zoom_factors=(1.0, 1.0, 1.0),
    size=90,
):
    """
    Save a cubic sub-volume centred on a single nucleus as a VTK file.

    Each voxel stores the nuclei label, the cytoplasm label, and the equalized
    intensity for every non-NUCLEI channel.

    Parameters
    ----------
    nuc_label : int
        Nucleus label ID to export.
    im_segmentation_stack : dict
        Must contain 'Nuclei' and 'Cytoplasm'.
    im_final_stack : dict
        Must contain 'Equalized image'.
    stain_df : DataFrame
        Original staining metadata.
    input_file : str or Path
        Path to the source image file (used to derive output filename).
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Physical voxel sizes in micrometers on the working (post-zoom) grid.
        Used as the ImageData spacing so the sub-volume isn't rendered as
        unit cubes in ParaView, which stretches/squashes it along Z.
    zoom_factors : tuple of float [Z, Y, X]
        Zoom factors applied when resampling to the working grid (includes
        `scale_factor`, see Cell 3). `size` is defined in voxels at
        zoom_factors=[1,1,1]; the per-axis crop is scaled by zoom_factors so
        the exported sub-volume always covers the same physical region
        regardless of scale_factor, instead of shrinking/growing with it.
    size : int
        Side length of the export volume in voxels at zoom_factors=[1,1,1]
        (default 90).
    """
    import pyvista as pv
    from pathlib import Path as _Path

    r_zX, r_zY, r_zZ = r_zxyz
    size_z = max(1, int(round(size * zoom_factors[0])))
    size_y = max(1, int(round(size * zoom_factors[1])))
    size_x = max(1, int(round(size * zoom_factors[2])))

    nuc_vol = im_segmentation_stack['Nuclei']
    mask = nuc_vol == nuc_label
    if not np.any(mask):
        raise ValueError(f"Nucleus {nuc_label} not found in im_segmentation_stack['Nuclei'].")
    if 'Cytoplasm' not in im_segmentation_stack:
        raise ValueError("im_segmentation_stack['Cytoplasm'] is missing. Run cytoplasm segmentation first.")

    cyto_vol = im_segmentation_stack['Cytoplasm']
    coords = np.argwhere(mask)
    cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)
    print(f"Nucleus {nuc_label} - centroid  Z={cz}  Y={cy}  X={cx}")

    half_z, half_y, half_x = size_z // 2, size_y // 2, size_x // 2
    Zmax, Ymax, Xmax = nuc_vol.shape
    z0d, z1d = cz - half_z, cz - half_z + size_z
    y0d, y1d = cy - half_y, cy - half_y + size_y
    x0d, x1d = cx - half_x, cx - half_x + size_x

    z0c, z1c = max(0, z0d), min(Zmax, z1d)
    y0c, y1c = max(0, y0d), min(Ymax, y1d)
    x0c, x1c = max(0, x0d), min(Xmax, x1d)

    zp0, zp1 = z0c - z0d, z0c - z0d + (z1c - z0c)
    yp0, yp1 = y0c - y0d, y0c - y0d + (y1c - y0c)
    xp0, xp1 = x0c - x0d, x0c - x0d + (x1c - x0c)

    def _crop3d(vol):
        out = np.zeros((size_z, size_y, size_x), dtype=vol.dtype)
        out[zp0:zp1, yp0:yp1, xp0:xp1] = vol[z0c:z1c, y0c:y1c, x0c:x1c]
        return out

    nuclei_crop = _crop3d(nuc_vol).astype(np.int32)
    cytoplasm_crop = _crop3d(cyto_vol).astype(np.int32)

    eq_img = im_final_stack['Equalized image']
    marker_crops = {}
    for c_idx in range(eq_img.shape[3]):
        cond = stain_df.index[c_idx]
        if cond != 'NUCLEI':
            marker = stain_df['Marker'].iloc[c_idx]
            ch_name = f"{cond}_{marker}".replace(" ", "_").replace("-", "_")
            marker_crops[ch_name] = _crop3d(eq_img[:, :, :, c_idx]).astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = (size_x, size_y, size_z)
    grid.origin = (float(x0d), float(y0d), float(z0d))
    grid.spacing = (float(r_zX), float(r_zY), float(r_zZ))
    grid.point_data['Nuclei_label'] = nuclei_crop.ravel()
    grid.point_data['Cytoplasm_label'] = cytoplasm_crop.ravel()
    for ch_name, crop in marker_crops.items():
        grid.point_data[ch_name] = crop.ravel()

    out_path = str(_Path(input_file).stem + f"_nuc{nuc_label}_3Dcrop.vtk")
    grid.save(out_path)
    print(f"Saved: {out_path}  ({size_z}x{size_y}x{size_x} voxels, {2 + len(marker_crops)} channels)")


# ---------------------------------------------------------------------------
# VTK / STL / FEA export helpers
# ---------------------------------------------------------------------------

def build_vtk_volumes(
    im_segmentation_stack,
    labels_full_df,
    stain_complete_df,
    input_file,
    r_xyz,
    zoom_factors,
    progress=None,
    output_dir=None,
    live_counter=True,
):
    """Build labelled VTK volume meshes for nuclei, cytoplasm and PCM and save to disk.

    Parameters
    ----------
    im_segmentation_stack : dict
        Must contain 'Nuclei', 'Cytoplasm', and 'PCM'.
    labels_full_df : DataFrame
        Full quantification table produced by build_full_labels_dict / labels_dict_to_dataframe.
    stain_complete_df : DataFrame
        Staining metadata (columns 'Condition', 'Marker', …).
    input_file : str or Path
        Source image path (used to derive output filenames).
    r_xyz : tuple of float
        Physical voxel sizes (r_X, r_Y, r_Z) in µm/px.
    zoom_factors : list of float
        Zoom factors [Z, Y, X] applied during isotropic resampling.
    progress : callable or None
        tqdm-compatible wrapper used for the per-nucleus loop.
    output_dir : str or Path, optional
        Folder to write the .vtk files to (default: current working directory).
    live_counter : bool
        If True, clear the cell output and print a 'NUCLEI j / n' counter per
        nucleus. Batch mode sets this to False so earlier output is kept.
    """
    import pyvista as pv
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    import tempfile
    import shutil
    from pathlib import Path as _Path
    from IPython.display import clear_output

    r_X, r_Y, r_Z = r_xyz
    # Physical size of one voxel along each array axis (Z, Y, X) on the
    # segmentation grid — used as the mesh voxel size so the exported
    # geometry isn't stretched/squashed along Z relative to XY.
    mesh_voxel_size = mr.Vector3f(
        float(r_Z / zoom_factors[0]),
        float(r_Y / zoom_factors[1]),
        float(r_X / zoom_factors[2]),
    )

    # Per-nucleus meshes are written to disk (meshlib requires a file path)
    # and immediately re-read; they are scratch files, not pipeline output,
    # so they live in a temp dir that gets removed once the loop is done
    # instead of littering the project directory as part_*.stl.
    _stl_tmpdir = tempfile.mkdtemp(prefix="napari_seg_stl_")
    _nuclei_stl_path = str(_Path(_stl_tmpdir) / "part_nuclei_mesh.stl")
    _cyto_stl_path = str(_Path(_stl_tmpdir) / "part_cyto_mesh.stl")
    _pcm_stl_path = str(_Path(_stl_tmpdir) / "part_PCM_mesh.stl")

    nuc_max = int(np.max(im_segmentation_stack['Nuclei']))
    cyto_max = int(np.max(im_segmentation_stack['Cytoplasm']))
    pcm_max = int(np.max(im_segmentation_stack['PCM']))
    # The LIVE/DEAD notebook has no PCM (empty layer): mesh the cells only.
    has_pcm = pcm_max > 0

    blocks_nuclei = pv.MultiBlock()
    blocks_cyto = pv.MultiBlock()
    blocks_PCM = pv.MultiBlock()

    # Indexed by the running mesh counter k (up to nuc_max), not by the
    # label: sized for every layer, so an empty PCM (LIVE/DEAD notebook) or a
    # cytoplasm with fewer labels than nuclei can't run past the end. Without
    # a PCM its volume stays 0 (the cell volume alone in the relative volumes).
    n_slots = max(nuc_max, cyto_max, pcm_max) + 1
    nuc_vol = np.zeros((n_slots,))
    nuc_coord = np.zeros((n_slots, 3))
    cyto_vol = np.zeros((n_slots,))
    cyto_coord = np.zeros((n_slots, 3))
    PCM_vol = np.zeros((n_slots,))
    PCM_coord = np.zeros((n_slots, 3))

    k = 0
    for j in _progress_iter(range(1, nuc_max + 1), progress, desc='Step 30 - Build VTK Volumes'):
        if live_counter:
            clear_output(wait=True)
            print(f'NUCLEI {j} / {nuc_max}')

        # --- nuclei ---
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['Nuclei'] == j))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mesh_voxel_size
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, _nuclei_stl_path)

        mesh_nuclei = pv.read(_nuclei_stl_path)
        if mesh_nuclei.volume > 0.0:
            mesh_nuclei.decimate(target_reduction=0.8, inplace=True)
            # mesh_nuclei geometry is already scaled by mesh_voxel_size, so
            # volume is in um3 and center coordinates are in um directly.
            nuc_vol[k] = mesh_nuclei.volume
            nuc_coord[k] = mesh_nuclei.center

            mesh_nuclei.cell_data['ID'] = np.ones(mesh_nuclei.n_cells) * (k + 1)
            mesh_nuclei.cell_data['Nuclei volume (um3)'] = np.ones(mesh_nuclei.n_cells) * nuc_vol[k]
            mesh_nuclei.cell_data['Z nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][0]
            mesh_nuclei.cell_data['Y nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][1]
            mesh_nuclei.cell_data['X nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][2]
            blocks_nuclei.append(mesh_nuclei)
            k += 1

        # --- cytoplasm ---
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['Cytoplasm'] == j))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mesh_voxel_size
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, _cyto_stl_path)
        mesh_cyto = pv.read(_cyto_stl_path)

        # --- PCM (absent in the LIVE/DEAD notebook: no PCM mesh then) ---
        mesh_PCM = None
        if has_pcm:
            simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['PCM'] == j))
            floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
            _g2m_settings = mr.GridToMeshSettings()
            _g2m_settings.voxelSize = mesh_voxel_size
            _g2m_settings.isoValue = 0.5
            mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
            mr.saveMesh(mesh_stl, _pcm_stl_path)
            mesh_PCM = pv.read(_pcm_stl_path)

        def _set_pcm(name, value):
            if mesh_PCM is not None:
                mesh_PCM.cell_data[name] = np.ones(mesh_PCM.n_cells) * value

        if mesh_cyto.volume > 0.0 and (mesh_PCM is None or mesh_PCM.volume > 0.0):
            mesh_cyto.decimate(target_reduction=0.8, inplace=True)
            if mesh_PCM is not None:
                mesh_PCM.decimate(target_reduction=0.8, inplace=True)

            # Already in physical units (um / um3) since the mesh geometry
            # itself was built with the correct anisotropic voxel size.
            cyto_vol[k] = mesh_cyto.volume
            cyto_coord[k] = mesh_cyto.center
            if mesh_PCM is not None:
                PCM_vol[k] = mesh_PCM.volume
                PCM_coord[k] = mesh_PCM.center

            mesh_cyto.cell_data['ID'] = np.ones(mesh_cyto.n_cells) * (k + 1)
            _set_pcm('ID', k + 1)
            mesh_cyto.cell_data['Cellular volume (um3)'] = np.ones(mesh_cyto.n_cells) * cyto_vol[k]
            _set_pcm('PCM volume (um3)', PCM_vol[k])
            mesh_cyto.cell_data['Z cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][0]
            mesh_cyto.cell_data['Y cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][1]
            mesh_cyto.cell_data['X cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][2]
            _set_pcm('Z PCM (um)', PCM_coord[k][0])
            _set_pcm('Y PCM (um)', PCM_coord[k][1])
            _set_pcm('X PCM (um)', PCM_coord[k][2])

            for i, marker in enumerate(labels_full_df.index):
                row = labels_full_df.iloc[i]
                cond = row['Condition']
                if cond in ('NUCLEI', 'CYTOPLASM') or np.size(marker) != 1:
                    continue
                shared = list(row['Shared labels'])

                # Whole-cell intensity, positive or not (dense, indexed
                # directly by cell label j -- not gated by "shared").
                all_mean = row.get('Avg. marker intensity (all cells)', ())
                if len(all_mean) >= j and not np.isnan(all_mean[j - 1]):
                    mesh_cyto.cell_data[marker + ' avg. intensity all cells (-)'] = np.ones(mesh_cyto.n_cells) * all_mean[j - 1]
                    _set_pcm(marker + ' avg. intensity all cells (-)', all_mean[j - 1])

                if j in shared:
                    idx = shared.index(j)
                    vol_um3 = row['Marker size [um3]'][idx]
                    cyto_combined = cyto_vol[k] + PCM_vol[k]
                    mesh_cyto.cell_data[marker + ' volume (um3)'] = np.ones(mesh_cyto.n_cells) * vol_um3
                    _set_pcm(marker + ' volume (um3)', vol_um3)
                    mesh_cyto.cell_data[marker + ' volume cytoplasm (um3)'] = np.ones(mesh_cyto.n_cells) * row['Marker size cytoplasm [um3]'][idx]
                    mesh_cyto.cell_data[marker + ' rel. vol. (-)'] = np.ones(mesh_cyto.n_cells) * (vol_um3 / cyto_combined)
                    _set_pcm(marker + ' rel. vol. (-)', vol_um3 / cyto_combined)
                    mesh_cyto.cell_data[marker + ' rel. vol. cytoplasm (-)'] = np.ones(mesh_cyto.n_cells) * (row['Marker size cytoplasm [um3]'][idx] / cyto_vol[k])
                    mesh_cyto.cell_data[marker + ' avg. intensity (-)'] = np.ones(mesh_cyto.n_cells) * row['Avg. marker intensity'][idx]
                    _set_pcm(marker + ' avg. intensity (-)', row['Avg. marker intensity'][idx])
                    mesh_cyto.cell_data[marker + ' avg. cytoplasm int. (-)'] = np.ones(mesh_cyto.n_cells) * row['Avg. marker intensity cytoplasm'][idx]
                    if mesh_PCM is not None:
                        _set_pcm(marker + ' volume PCM (um3)', row['Marker size PCM [um3]'][idx])
                        _set_pcm(marker + ' rel. vol. PCM (-)', row['Marker size PCM [um3]'][idx] / PCM_vol[k])
                        _set_pcm(marker + ' avg. PCM int. (-)', row['Avg. marker intensity PCM'][idx])
                else:
                    mesh_cyto.cell_data[marker + ' expression (um3)'] = np.zeros(mesh_cyto.n_cells)
                    mesh_cyto.cell_data[marker + ' rel. expr. (-)'] = np.zeros(mesh_cyto.n_cells)

            blocks_cyto.append(mesh_cyto)
            if mesh_PCM is not None:
                blocks_PCM.append(mesh_PCM)

    def _to_surface(block):
        if hasattr(block, 'extract_surface'):
            try:
                return block.extract_surface(algorithm=None)
            except TypeError:
                return block.extract_surface()
        return block.extract_geometry()

    _to_surface(blocks_nuclei).save(_stem_output_path(input_file, '_NUCLEI_labelled.vtk', output_dir))
    _to_surface(blocks_cyto).save(_stem_output_path(input_file, '_CYTOPLASM_labelled.vtk', output_dir))
    if has_pcm:
        _to_surface(blocks_PCM).save(_stem_output_path(input_file, '_PCM_labelled.vtk', output_dir))

    shutil.rmtree(_stl_tmpdir, ignore_errors=True)


def export_marker_stl(
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    input_file,
    r_xyz=(1.0, 1.0, 1.0),
    zoom_factors=(1.0, 1.0, 1.0),
    progress=None,
    output_dir=None,
):
    """Export per-marker binary volumes as STL mesh files.

    Parameters
    ----------
    im_segmentation_stack : dict
        Segmentation stack containing per-marker threshold images.
    stain_df : DataFrame
        Staining DataFrame with condition index and 'Marker' column.
    stain_complete_df : DataFrame
        Complete staining DataFrame used to filter NUCLEI/CYTOPLASM/PCM.
    input_file : str or Path
        Source image path (used to derive output filenames).
    r_xyz : tuple of float
        Physical voxel sizes (r_X, r_Y, r_Z) in µm/px, used so the exported
        mesh geometry reflects the real (often anisotropic) voxel spacing
        instead of treating every voxel as a 1x1x1 cube.
    zoom_factors : list of float
        Zoom factors [Z, Y, X] applied during isotropic resampling.
    progress : callable or None
        tqdm-compatible wrapper.
    output_dir : str or Path, optional
        Folder to write the .stl files to (default: current working directory).
    """
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn

    r_X, r_Y, r_Z = r_xyz
    mesh_voxel_size = mr.Vector3f(
        float(r_Z / zoom_factors[0]),
        float(r_Y / zoom_factors[1]),
        float(r_X / zoom_factors[2]),
    )

    for c, _marker in _progress_iter(
        enumerate(stain_complete_df.index),
        progress,
        desc='Step 31 - Export Marker STL',
        total=len(stain_complete_df.index),
    ):
        if stain_complete_df.index[c] in ('NUCLEI', 'CYTOPLASM', 'PCM'):
            continue
        row = stain_complete_df.iloc[c]
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack[stain_df.index[c]] > 0))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mesh_voxel_size
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, _stem_output_path(input_file, "_" + row['Marker'] + "_mesh.stl", output_dir))


def export_fea_mesh(
    im_segmentation_stack,
    input_file,
    r_zxyz=None,
    progress=None,
    output_dir=None,
):
    """Build a FEA tetrahedral mesh from the nuclei segmentation and write an Abaqus .inp file.

    This function combines three steps:
    1. Tetrahedralize the nuclei surface mesh with TetGen.
    2. Assign tetrahedral elements to individual nucleus labels.
    3. Write a final ``_FEA.inp`` Abaqus file with per-nucleus element sets.

    Node coordinates are written in micrometers, in X, Y, Z order. Nuclei that
    receive no element get no element set (Abaqus rejects empty sets).

    Parameters
    ----------
    im_segmentation_stack : dict
        Must contain 'Nuclei'.
    input_file : str or Path
        Source image path (used to derive output filenames).
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ), optional
        Voxel sizes of the segmentation grid in micrometers (after
        resampling). Without it, coordinates stay in voxel units.
    progress : callable or None
        tqdm-compatible wrapper.
    output_dir : str or Path, optional
        Folder to write the final ``_FEA.inp`` to (default: current working
        directory).
    """
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    import meshio
    import tempfile
    import tetgen
    import statistics as st

    nuc_max = int(np.max(im_segmentation_stack['Nuclei']))

    # --- Step 1: tetrahedralize ---
    simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['Nuclei']))
    floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
    _g2m_settings = mr.GridToMeshSettings()
    _g2m_settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
    _g2m_settings.isoValue = 0.5
    mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)

    outVerts = mrn.getNumpyVerts(mesh_stl)
    outFaces = mrn.getNumpyFaces(mesh_stl.topology)

    tet = tetgen.TetGen(outVerts, outFaces)
    tet_result = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    if isinstance(tet_result, tuple):
        if len(tet_result) >= 2:
            nodes, elems = tet_result[0], tet_result[1]
        else:
            nodes, elems = tet.node, tet.elem
    else:
        nodes, elems = tet.node, tet.elem
    nodes = np.asarray(nodes, dtype=float)
    elems = np.asarray(elems, dtype=np.int64)

    # --- Step 2: assign elements to nucleus labels ---
    cell_elements = {c: [] for c in range(1, nuc_max + 1)}

    for ce, x in _progress_iter(
        enumerate(elems), progress, desc='Step 33B - Assign Elements To Cells', total=len(elems)
    ):
        coord = np.int16(np.round(np.mean(nodes[x], 0), 0))
        step = 0
        taken = False
        while not taken:
            step += 1
            coord[coord < step] = 1
            for k in range(3):
                if coord[k] >= np.shape(im_segmentation_stack['Nuclei'])[k] + 1 - step:
                    coord[k] = np.shape(im_segmentation_stack['Nuclei'])[k] - 1
            elemlist = im_segmentation_stack['Nuclei'][
                coord[0] - step:coord[0] + 1 + step,
                coord[1] - step:coord[1] + 1 + step,
                coord[2] - step:coord[2] + 1 + step,
            ].flatten()
            if sum(elemlist) > 0:
                c_el = st.mode(elemlist[elemlist != 0])
                taken = True
        if c_el != 0 and c_el in cell_elements:
            cell_elements[c_el].append(ce + 1)

    # --- Step 3: write nodes, elements and element sets ---
    # The mesh was built on the (Z, Y, X) array in voxel units: reorder the
    # node coordinates to X, Y, Z and scale them to micrometers.
    if r_zxyz is None:
        print("export_fea_mesh: no voxel size given (r_zxyz) -- coordinates are in voxel units.")
        r_zX = r_zY = r_zZ = 1.0
    else:
        r_zX, r_zY, r_zZ = r_zxyz
    points = nodes[:, [2, 1, 0]] * np.array([r_zX, r_zY, r_zZ], dtype=float)

    # Swapping the X and Z axes mirrors the mesh, which flips the orientation
    # of every tetrahedron; reorder the nodes of any element with a negative
    # volume so Abaqus (C3D4) gets positive volumes throughout.
    p = points[elems]
    volume = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flipped = volume < 0
    elems[flipped] = elems[flipped][:, [0, 2, 1, 3]]

    # meshio writes the *NODE / *ELEMENT blocks (1-based ids, matching the
    # element numbers collected in cell_elements) to a temporary file.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_inp = os.path.join(tmp_dir, "FE_segmentation.inp")
        meshio.Mesh(points, [("tetra", elems)]).write(tmp_inp, file_format="abaqus")
        with open(tmp_inp, "r") as f:
            lines = f.readlines()

    out_path = _stem_output_path(input_file, "_FEA.inp", output_dir)
    empty_sets = 0
    with open(out_path, "w") as f:
        for line in _progress_iter(lines, progress, desc='Step 33C - Write Nodes And Elements'):
            if line.strip().upper() == "*NODE":
                f.write("*PART, name=Part-1\n")
            f.write(line)
        for c in _progress_iter(range(1, nuc_max + 1), progress, desc='Step 33D - Write Element Sets'):
            ids = cell_elements[c]
            if not ids:
                empty_sets += 1
                continue
            f.write(f"*Elset, elset=cell{c}\n")
            for i in range(0, len(ids), 16):  # Abaqus: at most 16 entries per data line
                f.write(", ".join(str(e) for e in ids[i:i + 16]) + "\n")
        f.write("*END PART\n")

    units = "um" if r_zxyz is not None else "voxels"
    print(
        f"FEA mesh: {len(points)} nodes, {len(elems)} tetrahedra, coordinates in {units} (X, Y, Z). "
        f"Saved to {out_path}"
    )
    if empty_sets:
        print(f"{empty_sets} nucleus label(s) received no element and have no element set.")


# ---------------------------------------------------------------------------
# Batch mode: input_file is a folder
# ---------------------------------------------------------------------------

# .ome.tif/.ome.tiff are covered by .tif/.tiff. ".ldpair" = a LIVE/DEAD pair
# file (two images read as one, see `combine_live_dead_inputs`).
SUPPORTED_IMAGE_EXTENSIONS = (".nd2", ".tif", ".tiff", ".czi", ".lif", ".oib", LD_PAIR_EXTENSION)
_RAW_IMAGE_EXTENSIONS = tuple(e for e in SUPPORTED_IMAGE_EXTENSIONS if e != LD_PAIR_EXTENSION)


class BatchCompleted(Exception):
    """Raised by the batch cell once the folder is processed, so that 'Run All'
    stops there instead of running the single-file cells that follow.
    Rendered as a one-line message rather than a traceback."""

    def _render_traceback_(self):
        return [str(self) or "Batch processing finished."]


def is_batch_input(input_file):
    """True if *input_file* is a folder (batch mode) rather than a single file."""
    return os.path.isdir(input_file)


def get_batch_output_dir(input_folder):
    """Return the ``<folder>_output`` sibling folder used for batch outputs."""
    from pathlib import Path as _Path
    folder = _Path(input_folder).resolve()
    return folder.parent / f"{folder.name}_output"


def list_image_files(input_folder, extensions=SUPPORTED_IMAGE_EXTENSIONS):
    """Sorted list of the supported image files directly inside *input_folder*
    (subfolders are not searched)."""
    from pathlib import Path as _Path
    return sorted(
        p for p in _Path(input_folder).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def _ld_pair_stem(live_stem, dead_stem):
    """Name of a LIVE/DEAD pair: the parts the two file names share
    ('S1_live_20x' + 'S1_dead_20x' -> 'S1_20x'; the same name in two folders
    -> that name), or 'live+dead' when the names have no common parts."""
    split = re.compile(r'([_\-\s.]+)')
    a, b = split.split(live_stem), split.split(dead_stem)
    if len(a) == len(b):
        parts = []
        for i in range(0, len(a), 2):  # even = name parts, odd = separators
            if a[i].lower() == b[i].lower():
                if parts and i > 0:
                    parts.append(a[i - 1])
                parts.append(a[i])
        stem = "".join(parts).strip("_-. ")
        if stem:
            return stem
    return f"{live_stem}+{dead_stem}"


def _write_ld_pair_file(path, live_file, dead_file):
    record = {
        "description": "LIVE/DEAD pair: the two images are read as one multichannel image "
                       "(NucleoSeg3D_LD v1.2.2).",
        "files": [{"label": "LIVE", "path": os.path.abspath(str(live_file))},
                  {"label": "DEAD", "path": os.path.abspath(str(dead_file))}],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return str(path)


def combine_live_dead_inputs(input_file=None, input_file_live=None, input_file_dead=None,
                             pair_folder=None):
    """LIVE/DEAD v1.2.2, Cell 3: the input of the notebook, from either one
    stacked image or a separate LIVE and DEAD image.

    - ``input_file_live`` and ``input_file_dead`` both None: ``input_file``
      (a stacked image with every channel, or a folder of them) is returned
      unchanged.
    - two **files**: a pair file ``<name>.ldpair`` is written next to the
      LIVE file and returned; it is read as one image with the channels of
      the LIVE file, then those of the DEAD file (named ``'LIVE - ...'`` and
      ``'DEAD - ...'`` in the stain table). ``<name>`` is what the two file
      names share (`_ld_pair_stem`) and names every output.
    - two **folders** (batch mode): their images are paired in sorted name
      order (both folders must hold the same number of images; the pairs are
      printed -- check them), one pair file per pair is written into
      ``pair_folder`` (default: ``<LIVE folder>_LD_pairs`` next to the LIVE
      folder) and that folder is returned: the batch then runs on it and
      writes to ``<pair folder>_output``.

    The two images of a pair must have the same size and voxel size.
    """
    from pathlib import Path as _Path

    if input_file_live is None and input_file_dead is None:
        return input_file
    if input_file_live is None or input_file_dead is None:
        raise ValueError("Set both input_file_live and input_file_dead (or neither, to use "
                         "the stacked image in input_file).")
    live, dead = _Path(input_file_live), _Path(input_file_dead)
    for p, what in ((live, "input_file_live"), (dead, "input_file_dead")):
        if not p.exists():
            raise FileNotFoundError(f"{what} not found: {p}")
    if live.is_dir() != dead.is_dir():
        raise ValueError("input_file_live and input_file_dead must be two files or two folders.")

    if not live.is_dir():
        if live.suffix.lower() not in _RAW_IMAGE_EXTENSIONS or \
                dead.suffix.lower() not in _RAW_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file(s); supported: {', '.join(_RAW_IMAGE_EXTENSIONS)}")
        stem = _ld_pair_stem(_image_stem(live), _image_stem(dead))
        path = _write_ld_pair_file(live.parent / f"{stem}{LD_PAIR_EXTENSION}", live, dead)
        print(f"Separate LIVE and DEAD images, read as one image '{stem}' (pair file {path}):")
        print(f"  LIVE: {live}\n  DEAD: {dead}")
        return path

    live_files = list_image_files(live, extensions=_RAW_IMAGE_EXTENSIONS)
    dead_files = list_image_files(dead, extensions=_RAW_IMAGE_EXTENSIONS)
    if not live_files or len(live_files) != len(dead_files):
        raise ValueError(
            f"The LIVE folder holds {len(live_files)} image(s), the DEAD folder "
            f"{len(dead_files)}: they must hold the same number (> 0), paired in name order.")
    out = _Path(pair_folder) if pair_folder else live.parent / f"{live.name}_LD_pairs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"*{LD_PAIR_EXTENSION}"):  # pair files of an earlier run
        old.unlink()
    print(f"Batch of separate LIVE and DEAD images: {len(live_files)} pair(s), paired in "
          f"name order -- check that each LIVE image goes with its DEAD image:")
    used = set()
    for n, (lf, df) in enumerate(zip(live_files, dead_files), start=1):
        stem = _ld_pair_stem(_image_stem(lf), _image_stem(df))
        if stem.lower() in used:
            stem = f"{_image_stem(lf)}+{_image_stem(df)}"
        used.add(stem.lower())
        _write_ld_pair_file(out / f"{stem}{LD_PAIR_EXTENSION}", lf, df)
        print(f"  [{n}] {stem}:  LIVE {lf.name}  +  DEAD {df.name}")
    print(f"Pair files written to {out}; outputs will go to {get_batch_output_dir(out)}")
    return str(out)


def _print_channel_names(channels, source):
    """Print *channels* as a numbered list (the options of the stain table)."""
    print(f"\nChannels in {source}:")
    if not channels:
        print("  (no channel names found in the file metadata)")
        return
    for i, name in enumerate(channels):
        print(f"  [{i}] '{name}'")


def print_channel_list(input_file, file_meta=None):
    """Print the file's channel names as a reference for ``stain_dict``.

    Single file: uses *file_meta* (from Cell 4) if given, otherwise reads the
    metadata. Folder (batch mode): reads only the metadata of every image
    (no pixel data), prints the first image's channel list, and lists any
    image whose channels differ from it -- all images in a batch are expected
    to share the same channels, since one ``stain_dict`` is used for all.

    Returns
    -------
    list of str
        The channel names of the (first) image.
    """
    from pathlib import Path as _Path

    if not is_batch_input(input_file):
        if file_meta is None:
            file_meta = read_file_metadata(input_file, open_image_file(input_file))
        channels = list(file_meta.get("channels") or [])
        _print_channel_names(channels, _Path(input_file).name)
        return channels

    per_file = {}
    for image_path in list_image_files(input_file):
        try:
            meta = read_file_metadata(str(image_path), open_image_file(str(image_path)))
            per_file[image_path.name] = list(meta.get("channels") or [])
        except Exception as exc:
            print(f"[print_channel_list] Could not read channels of "
                  f"{image_path.name}: {exc}")
    if not per_file:
        print("[print_channel_list] No channel names could be read from the batch folder.")
        return []

    first_name, channels = next(iter(per_file.items()))
    _print_channel_names(channels, f"{first_name} (first image of the batch)")
    mismatched = {name: ch for name, ch in per_file.items() if ch != channels}
    if mismatched:
        print(f"\nWARNING: {len(mismatched)} image(s) have a different channel "
              f"list -- stain_dict may not match them:")
        for name, ch in mismatched.items():
            print(f"  {name}: {ch}")
    else:
        print(f"All {len(per_file)} image(s) in the batch share this channel list.")
    return channels


# ---------------------------------------------------------------------------
# Interactive stain table (Cell 4)
# ---------------------------------------------------------------------------
STAIN_COLORS = ("blue", "red", "green", "white", "yellow", "magenta")
_STAIN_NO_CHANNEL = "no"
_STAIN_FIXED_CONDITIONS = ("NUCLEI", "CYTOPLASM")
# (keywords in the channel name, default display color) -- first match wins.
_CHANNEL_COLOR_HINTS = (
    (("DAPI", "HOECHST", "405", "BFP"), "blue"),
    (("488", "GFP", "FITC"), "green"),
    (("555", "561", "568", "594", "CY3", "RFP", "MCHERRY", "TRITC", "TXRED"), "red"),
    (("633", "640", "647", "CY5"), "magenta"),
    (("BRIGHT", "TRANS", "DIC"), "white"),
)


class StainTableNotConfirmed(Exception):
    """Raised when the Cell 4 stain table is read before it was confirmed."""


def _guess_channel_color(channel, index):
    name = str(channel).upper()
    for keys, color in _CHANNEL_COLOR_HINTS:
        if any(key in name for key in keys):
            return color
    return STAIN_COLORS[index % len(STAIN_COLORS)]


def _default_stain_rows(channels):
    """Preliminary stain table: NUCLEI on the DAPI/Hoechst-like channel (or the
    first one), no CYTOPLASM, and one STAIN_<n> row per remaining channel."""
    nuclei_channel = next(
        (c for c in channels if any(k in c.upper() for k in ("DAPI", "HOECHST", "405"))),
        channels[0] if channels else _STAIN_NO_CHANNEL,
    )
    rest = [c for c in channels if c != nuclei_channel]
    rows = [
        dict(condition="NUCLEI", marker="", channel=nuclei_channel,
             color=_guess_channel_color(nuclei_channel, 0), intracellular=False,
             membrane=False),
        dict(condition="CYTOPLASM", marker="", channel=_STAIN_NO_CHANNEL,
             color="white", intracellular=False, membrane=False),
    ]
    for i, channel in enumerate(rest or [_STAIN_NO_CHANNEL], start=1):
        rows.append(dict(condition=f"STAIN_{i}", marker="", channel=channel,
                         color=_guess_channel_color(channel, i), intracellular=False,
                         membrane=False))
    return rows


class StainDictEditor:
    """Interactive table (ipywidgets) that builds ``stain_dict`` and
    ``cyto_markers`` from the channels found in the file.

    One row per condition: condition name (fixed for NUCLEI/CYTOPLASM), marker
    name (text), channel (drop-down of the file's channels, or 'no' if the
    condition isn't used), display color (drop-down), an 'intracellular?'
    tick that adds the marker to ``cyto_markers``, and a 'membrane?' tick for
    membrane stains: they are in ``cyto_markers`` too, their rings are
    filled to get the cytoplasm and the bright membrane marks the border
    between touching cells (see :attr:`membrane_markers`). The two ticks
    exclude each
    other. Clicking 'Confirm' checks
    the table and saves it to ``<name_setup>_stain_dict.json``; a saved table
    is loaded (and counts as confirmed) the next time the notebook runs.
    Read the result with :meth:`result`.
    """

    _COLUMNS = ("Condition", "Marker", "Channel", "Color", "Intracellular?", "Membrane?", "")

    def __init__(self, channels, name_setup=None, use_saved=True):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "The interactive stain table needs ipywidgets: "
                "uv pip install ipywidgets"
            ) from exc
        self._w = widgets
        self.channels = [str(c) for c in (channels or [])]
        self.save_path = f"{name_setup}_stain_dict.json" if name_setup else None
        self._rows = []
        self._confirmed = False
        self._load_notes = []

        loaded = self._load_saved() if use_saved else None
        for row in loaded or _default_stain_rows(self.channels):
            self._add_row(row, render=False)

        self._grid = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="120px 180px 240px 110px 110px 100px 40px",
            grid_gap="4px 10px",
            align_items="center",
        ))
        add_button = widgets.Button(description="Add stain", icon="plus")
        add_button.on_click(lambda _: self._add_row(self._new_stain_row()))
        confirm_button = widgets.Button(description="Confirm", icon="check",
                                        button_style="success")
        confirm_button.on_click(self._on_confirm)
        self._status = widgets.HTML()
        self._output = widgets.Output()
        self.widget = widgets.VBox([
            widgets.HTML("<b>Stain table</b> -- pick each condition's channel "
                         "('no' = not used) and color, type the marker names, "
                         "tick the intracellular and membrane markers, then click "
                         "<b>Confirm</b>."),
            self._grid,
            widgets.HBox([add_button, confirm_button]),
            self._status,
            self._output,
        ])
        self._render()
        errors, _ = self._validate()
        self._confirmed = loaded is not None and not errors and not self._load_notes
        self._update_status()

    # -- saved table --------------------------------------------------------
    def _load_saved(self):
        if not self.save_path or not os.path.exists(self.save_path):
            return None
        try:
            with open(self.save_path, encoding="utf-8") as f:
                rows = json.load(f)["rows"]
        except Exception as exc:
            print(f"[StainDictEditor] Could not read {self.save_path} ({exc}); "
                  f"starting from a new table.")
            return None
        # NUCLEI and CYTOPLASM always come first, even if missing from the file.
        saved = {r.get("condition"): r for r in rows}
        defaults = {r["condition"]: r for r in _default_stain_rows(self.channels)}
        rows = ([dict(saved.get(c, defaults[c])) for c in _STAIN_FIXED_CONDITIONS]
                + [dict(r) for r in rows if r.get("condition") not in _STAIN_FIXED_CONDITIONS])
        for row in rows:
            channel = row.get("channel", _STAIN_NO_CHANNEL)
            if channel != _STAIN_NO_CHANNEL and channel not in self.channels:
                self._load_notes.append(
                    f"{row.get('condition')}: saved channel '{channel}' is not in "
                    f"this file -- set to 'no'.")
                row["channel"] = _STAIN_NO_CHANNEL
        print(f"Stain table loaded from {self.save_path}"
              + ("" if not self._load_notes else " (needs review, see below)."))
        return rows

    # -- rows ---------------------------------------------------------------
    def _new_stain_row(self):
        taken = {r["condition"] for r in self._values()}
        n = 1
        while f"STAIN_{n}" in taken:
            n += 1
        return dict(condition=f"STAIN_{n}", marker="", channel=_STAIN_NO_CHANNEL,
                    color=STAIN_COLORS[n % len(STAIN_COLORS)], intracellular=False,
                    membrane=False)

    def _add_row(self, row, render=True):
        w = self._w
        fixed = row["condition"] in _STAIN_FIXED_CONDITIONS
        auto = w.Layout(width="auto")
        if fixed:
            condition = w.HTML(f"<b>{row['condition']}</b>")
        else:
            condition = w.Text(value=row["condition"], placeholder="e.g. STAIN_1", layout=auto)
        marker = w.Text(value=row.get("marker", ""), placeholder="marker name", layout=auto)
        options = [_STAIN_NO_CHANNEL] + self.channels
        channel_value = row.get("channel", _STAIN_NO_CHANNEL)
        channel = w.Dropdown(options=options, layout=auto,
                             value=channel_value if channel_value in options else _STAIN_NO_CHANNEL)
        color_value = row.get("color", "white")
        color = w.Dropdown(options=list(STAIN_COLORS), layout=auto,
                           value=color_value if color_value in STAIN_COLORS else "white")
        is_membrane = bool(row.get("membrane")) and not fixed
        intracellular = w.Checkbox(value=bool(row.get("intracellular")) and not fixed
                                   and not is_membrane,
                                   disabled=fixed, indent=False, description="")
        membrane = w.Checkbox(value=is_membrane, disabled=fixed, indent=False, description="")

        # A marker is either intracellular or membrane, not both.
        def _exclusive(change, other):
            if change["new"]:
                other.value = False
        intracellular.observe(lambda ch: _exclusive(ch, membrane), names="value")
        membrane.observe(lambda ch: _exclusive(ch, intracellular), names="value")
        if fixed:
            remove = w.HTML("")
        else:
            remove = w.Button(icon="times", tooltip="Remove this stain",
                              layout=w.Layout(width="36px"))
        entry = dict(fixed=fixed, name=row["condition"], condition=condition,
                     marker=marker, channel=channel, color=color,
                     intracellular=intracellular, membrane=membrane, remove=remove)
        if not fixed:
            remove.on_click(lambda _, e=entry: self._remove_row(e))
            condition.observe(self._on_change, names="value")
        for widget in (marker, channel, color, intracellular, membrane):
            widget.observe(self._on_change, names="value")
        self._rows.append(entry)
        if render:
            self._render()
            self._on_change()

    def _remove_row(self, entry):
        self._rows.remove(entry)
        self._render()
        self._on_change()

    def _render(self):
        header = [self._w.HTML(f"<b>{c}</b>") for c in self._COLUMNS]
        cells = []
        for e in self._rows:
            cells += [e["condition"], e["marker"], e["channel"], e["color"],
                      e["intracellular"], e["membrane"], e["remove"]]
        self._grid.children = header + cells

    def _values(self):
        return [
            dict(condition=(e["name"] if e["fixed"] else e["condition"].value).strip().upper(),
                 marker=e["marker"].value.strip(),
                 channel=e["channel"].value,
                 color=e["color"].value,
                 intracellular=e["intracellular"].value and not e["fixed"],
                 membrane=e["membrane"].value and not e["fixed"])
            for e in self._rows
        ]

    # -- validation / status ------------------------------------------------
    def _validate(self):
        rows = self._values()
        errors, notes = [], list(self._load_notes)
        names = [r["condition"] for r in rows]
        if "" in names:
            errors.append("Every stain needs a condition name.")
        dup_names = sorted({n for n in names if n and names.count(n) > 1})
        if dup_names:
            errors.append(f"Condition name used more than once: {dup_names}")
        used = [r for r in rows if r["channel"] != _STAIN_NO_CHANNEL]
        if not used:
            errors.append("Assign a channel to at least one condition.")
        for r in used:
            if not r["marker"]:
                errors.append(f"{r['condition'] or '(unnamed)'}: type a marker name.")
        used_channels = [r["channel"] for r in used]
        dup_channels = sorted({c for c in used_channels if used_channels.count(c) > 1})
        if dup_channels:
            errors.append(f"Channel assigned to more than one condition: {dup_channels}")
        markers = [r["marker"].upper() for r in used if r["marker"]]
        dup_markers = sorted({m for m in markers if markers.count(m) > 1})
        if dup_markers:
            errors.append(f"Marker name used more than once: {dup_markers}")

        by_name = {r["condition"]: r for r in rows}
        if by_name["NUCLEI"]["channel"] == _STAIN_NO_CHANNEL:
            notes.append("No NUCLEI channel: cells are segmented from the union "
                         "of all channels.")
        if (by_name["CYTOPLASM"]["channel"] != _STAIN_NO_CHANNEL
                and any(r["intracellular"] or r["membrane"] for r in used)):
            notes.append("A CYTOPLASM channel is set: from v1.6.2 the ticked markers "
                         "are added to it to build the cytoplasm (before v1.6.2 "
                         "only the CYTOPLASM channel is used).")
        unused = [c for c in self.channels if c not in used_channels]
        if unused:
            notes.append(f"Channel(s) not used (dropped from the analysis): {unused}")
        return errors, notes

    def _on_change(self, _change=None):
        self._confirmed = False
        self._load_notes = []
        self._update_status()

    def _update_status(self):
        import html
        errors, notes = self._validate()
        if errors:
            head = "<span style='color:#d9534f'><b>Fix before confirming:</b></span>"
            items = errors
        elif self._confirmed:
            head = "<span style='color:#2e8b57'><b>Confirmed.</b></span>"
            items = []
        else:
            head = ("<span style='color:#e69500'><b>Not confirmed yet</b> -- "
                    "click Confirm.</span>")
            items = []
        lines = [f"<li>{html.escape(t)}</li>" for t in items]
        lines += [f"<li><i>{html.escape(t)}</i></li>" for t in notes]
        self._status.value = head + (f"<ul>{''.join(lines)}</ul>" if lines else "")

    def _on_confirm(self, _button):
        errors, _ = self._validate()
        if errors:
            self._update_status()
            return
        self._confirmed = True
        self._update_status()
        self._output.clear_output()
        with self._output:
            if self.save_path:
                with open(self.save_path, "w", encoding="utf-8") as f:
                    json.dump({"rows": self._values()}, f, indent=2)
                print(f"Saved to {self.save_path}")
            print(f"stain_dict = {self.stain_dict}")
            print(f"cyto_markers = {self.cyto_markers}")
            print(f"membrane_markers = {self.membrane_markers}")

    # -- results ------------------------------------------------------------
    @property
    def stain_dict(self):
        """``{condition: [marker, channel, color]}`` for every used condition."""
        return {r["condition"]: [r["marker"], r["channel"], r["color"]]
                for r in self._values() if r["channel"] != _STAIN_NO_CHANNEL}

    @property
    def cyto_markers(self):
        """Marker names that make up the cytoplasm: ticked as intracellular or
        as membrane (upper case, as in stain_df)."""
        return [r["marker"].upper() for r in self._values()
                if r["channel"] != _STAIN_NO_CHANNEL
                and (r["intracellular"] or r["membrane"])]

    @property
    def membrane_markers(self):
        """Marker names ticked as membrane (a subset of ``cyto_markers``):
        their rings are filled to get the cytoplasm, and the membrane marks
        the border between touching cells."""
        return [r["marker"].upper() for r in self._values()
                if r["channel"] != _STAIN_NO_CHANNEL and r["membrane"]]

    def result(self):
        """Return ``(stain_dict, cyto_markers)``; raise StainTableNotConfirmed
        if the table has not been confirmed (or has been edited since)."""
        errors, _ = self._validate()
        if errors or not self._confirmed:
            raise StainTableNotConfirmed(
                "The stain table in Cell 5 is not confirmed yet: fill it in, "
                "click 'Confirm', then run the notebook again from Cell 6."
            )
        return self.stain_dict, self.cyto_markers


def edit_stain_dict(channels, name_setup=None, use_saved=True):
    """Display the interactive stain table for *channels* and return the
    :class:`StainDictEditor` (read it later with ``.result()``)."""
    from IPython.display import display
    editor = StainDictEditor(channels, name_setup=name_setup, use_saved=use_saved)
    display(editor.widget)
    return editor


# v1.6.4: NUCLEI settings of the nuclei notebook v1.6.4 for the batch
# mode (`run_batch_folder(nuclei_options=...)`), at their "off" values: the
# depth correction (Cell 10), the slice-wise haze removal (Cell 12), the
# threshold weights (Cell 15) and the segment_nuclei options (Cell 17).
BATCH_NUCLEI_OPTIONS = {
    "z_atten_model": "off", "z_atten_strength": 1.0, "z_atten_conditions": ["NUCLEI"],
    "z_bg_radius_um": 0.0, "z_bg_percentile": 25.0,
    "global_threshold_weight": 0.15, "sauvola_weight": 0.60,
    "z_elongation": 1.0, "nuclei_expand_um": 0.0, "nuclei_flare_trim_factor": 0.0,
    "nuclei_saddle_dip": 0.0, "nuclei_edge_fraction": 0.0, "nuclei_edge_margin_um": 1.0,
    "nuclei_tall_volume_factor": 1.15, "nuclei_tall_elongation_factor": 0.0,
    "nuclei_rescue_contrast": 0.0, "nuclei_rescue_grow_fraction": 0.35,
    "auto_adapt": False,  # v1.6.4: adapt the settings to each image (adapt_settings_to_image)
    # LD: False = keep the scale_factor as given (set in custom_settings)
    # when the settings are adapted to each image
    "adapt_scale": True,
    # v1.6.4: "calibrated" = NUCLEI contrast of every image from its own
    # histogram (`calibrated_contrast`), whatever the setup / automatic_contrast
    "nuclei_contrast": None,
    "nuclei_gamma": 1.0,  # v1.6.41: Gamma of the calibrated NUCLEI contrast
    "resample_edge_mode": "constant",  # v1.6.41: "nearest" keeps the last Z slice (resample_to_isotropic)
    # v1.6.41: {setting: rule} of settings that follow each image's channel
    # histograms (helpers/histogram_adapt.py); None = off
    "histogram_rules": None,
}
# v1.6.41: names of the optimal settings (histogram rules) in cyto_split_config
_BATCH_CYTO_RULE_KEYS = {
    "cyto_grow_um": "grow_um", "cyto_mask_close_um": "mask_close_um",
    "cyto_membrane_close_fraction": "membrane_close_fraction",
    "cyto_membrane_fill_area_factor": "membrane_fill_area_factor",
    "cyto_threshold": "cyto_threshold"}
_BATCH_SEGMENT_NUCLEI_KEYS = (
    "z_elongation", "nuclei_expand_um", "nuclei_flare_trim_factor", "nuclei_saddle_dip",
    "nuclei_edge_fraction", "nuclei_edge_margin_um", "nuclei_tall_volume_factor",
    "nuclei_tall_elongation_factor", "nuclei_rescue_contrast", "nuclei_rescue_grow_fraction")


def _process_single_image_batch(
    input_file,
    output_dir,
    roi_coords,
    stain_dict,
    cyto_markers,
    name_setup,
    use_setup,
    automatic_contrast,
    interactive_roi,
    nuclei_diameter,
    cell_diameter,
    scale_factor,
    zoom_factors,
    trig_cellpose,
    trig_stardist,
    trig_cellpose_cyto,
    multilabel,
    aggregate_distance_factor,
    nuclei_split_config,
    cyto_split_config,
    iterative_split,
    oversize_factor,
    max_split_iterations,
    merge_undersized,
    undersize_factor,
    size_reference,
    sigma,
    num_plateaus,
    plateau_factor,
    threshold_method,
    export_vtk,
    export_stl,
    export_fea,
    settings,
    napari_module,
    progress,
    membrane_markers=None,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.0,
    pcm_thickness_factor=0.05,
    nuclei_options=None,
):
    """Run the notebook pipeline (Cells 5-35) on one file, without viewers,
    plots or PNG subfolders, writing every output file into *output_dir*
    (this image's own subfolder of the batch output folder).
    ``split_gamma``, ``split_gamma_step`` and ``min_piece_fraction`` are the
    nuclei notebook v1.6.3 splitting settings (defaults: off, as in v1.6.2).
    ``nuclei_options``: the v1.6.4 NUCLEI settings, see
    `BATCH_NUCLEI_OPTIONS` (missing keys = off)."""
    opts = dict(BATCH_NUCLEI_OPTIONS, **(nuclei_options or {}))
    unknown = set(opts) - set(BATCH_NUCLEI_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown nuclei_options: {sorted(unknown)}")
    roi = list(roi_coords)
    if interactive_roi:
        roi = select_roi_interactively(input_file, roi, napari_module=napari_module,
                                       stain_table_path=(f"{name_setup}_stain_dict.json"
                                                         if use_setup else None))

    # Cell 5
    (meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
     cyto_factor, PCM_factor, zooms, used_lazy_loading) = initialize_dataset(
        input_file, roi,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        scale_factor=scale_factor,
        zoom_factors=list(zoom_factors),
    )
    # 2D image (single plane): segmented in 2D, sizes are areas, no 3D exports.
    is_2d = img.shape[0] == 1

    # Cells 7-8
    im_final_stack, _, _, _, _, stain_df, _ = prepare_and_preview(
        img,
        nuclei_diameter, cell_diameter,
        stain_dict, file_meta,
        napari_module=napari_module, r_xyz=(r_X, r_Y, r_Z), progress=progress,
        show_viewer=False,
    )
    del img
    stain_df, stain_complete_df, original_stain_complete_df = prepare_stain_settings(
        im_final_stack['Original image'],
        stain_df=stain_df,
        name_setup=name_setup,
        use_setup=use_setup,
        automatic_contrast=automatic_contrast,
        settings=settings,
        napari_module=napari_module,
        r_xyz=(r_X, r_Y, r_Z),
        progress=progress,
    )
    if opts["nuclei_contrast"] == "calibrated" and "NUCLEI" in stain_complete_df.index:  # v1.6.4
        nuc_contrast = calibrated_contrast(
            im_final_stack['Original image'][..., stain_complete_df.index.get_loc("NUCLEI")],
            gamma=opts["nuclei_gamma"])
        for col, value in nuc_contrast.items():
            stain_complete_df.loc["NUCLEI", col] = value
            original_stain_complete_df.loc["NUCLEI", col] = value
    elif opts["nuclei_contrast"] not in (None, "calibrated"):
        raise ValueError("nuclei_options['nuclei_contrast'] must be None or 'calibrated'")
    histogram_note = "off"
    if opts["histogram_rules"]:  # v1.6.41: settings from this image's channel histograms
        from helpers.histogram_adapt import adapt_settings, features_by_condition
        feats = features_by_condition(im_final_stack['Original image'], stain_complete_df)
        cyto_split_config = dict(cyto_split_config or get_cyto_split_config(profile="balanced"))
        nuc_rules = {k: r for k, r in opts["histogram_rules"].items() if k in opts}
        changes = adapt_settings(opts, nuc_rules, feats)
        cyto_rules = {}
        for k, r in opts["histogram_rules"].items():
            head, dot, rest = k.partition(".")
            if head in _BATCH_CYTO_RULE_KEYS:
                cyto_rules[_BATCH_CYTO_RULE_KEYS[head] + dot + rest] = r
        changes += adapt_settings(cyto_split_config, cyto_rules, feats)
        histogram_note = "; ".join(f"{n} {o} -> {v} ({s})" for n, o, v, s in changes) or "no change"
    adapted_note = "off"
    if opts["auto_adapt"] and not is_2d and "NUCLEI" in stain_complete_df.index:  # v1.6.4
        adapted, changes = adapt_settings_to_image(
            im_final_stack['Original image'], stain_complete_df, (r_Z, r_Y, r_X),
            {"nuclei_diameter": nuclei_diameter, "scale_factor": scale_factor,
             "z_bg_radius_um": opts["z_bg_radius_um"],
             "nuclei_edge_margin_um": opts["nuclei_edge_margin_um"]})
        nuclei_diameter = adapted["nuclei_diameter"]
        opts["z_bg_radius_um"] = adapted["z_bg_radius_um"]
        opts["nuclei_edge_margin_um"] = adapted["nuclei_edge_margin_um"]
        if adapted["scale_factor"] != scale_factor:
            scale_factor = adapted["scale_factor"]
            zooms = [z * scale_factor for z in zoom_factors]
        cyto_factor = int(np.round(cell_diameter / nuclei_diameter))
        adapted_note = "; ".join(f"{n} {o} -> {v}" for n, o, v, _ in changes) or "no change"
    elif opts["auto_adapt"] and is_2d:  # 2D: resolution only, for the given nucleus size
        scale_factor, changes = adapt_scale_2d(nuclei_diameter, scale_factor, r_X)
        zooms = [z * scale_factor for z in zoom_factors]
        zooms[0] = 1.0  # never resample a single plane along Z
        adapted_note = "; ".join(f"{n} {o} -> {v}" for n, o, v, _ in changes) or "no change"

    # Cells 10-16
    if opts["z_atten_model"] != "off" and im_final_stack['Original image'].shape[0] > 2:  # v1.6.4
        conds = list(stain_complete_df.index) if opts["z_atten_conditions"] == "all" else \
            [c for c in opts["z_atten_conditions"] if c in stain_complete_df.index]
        im_final_stack['Original image'], _ = correct_z_attenuation(
            im_final_stack['Original image'], model=opts["z_atten_model"],
            strength=opts["z_atten_strength"],
            channels=[stain_complete_df.index.get_loc(c) for c in conds], inplace=True)
    im_final_stack = run_normalize(im_final_stack, stain_complete_df=stain_complete_df)
    im_final_stack, r_zX, r_zY, r_zZ = run_resample(
        im_final_stack, stain_complete_df=stain_complete_df, zoom_factors=zooms, meta=meta,
        edge_mode=opts["resample_edge_mode"],
    )
    im_final_stack = run_denoise(im_final_stack, stain_complete_df=stain_complete_df)
    if opts["z_bg_radius_um"] > 0 and im_final_stack['Denoised image'].shape[0] > 1 \
            and "NUCLEI" in stain_complete_df.index:  # v1.6.4
        im_final_stack['Denoised image'] = correct_z_background(
            im_final_stack['Denoised image'], opts["z_bg_radius_um"], (r_zZ, r_zY, r_zX),
            percentile=opts["z_bg_percentile"],
            channels=[stain_complete_df.index.get_loc("NUCLEI")])
    im_final_stack = run_contrast_gamma(im_final_stack, stain_complete_df=stain_complete_df)
    im_final_stack = run_smooth(im_final_stack, stain_complete_df=stain_complete_df, sigma=sigma)
    im_final_stack = run_equalize(
        im_final_stack, stain_complete_df=stain_complete_df,
        num_plateaus=num_plateaus, plateau_factor=plateau_factor,
    )
    im_final_stack = run_threshold(
        im_final_stack,
        stain_complete_df=stain_complete_df,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        r_zX=r_zX, r_zY=r_zY, r_zZ=r_zZ,
        threshold_method=threshold_method,
        global_threshold_weight=opts["global_threshold_weight"],
        sauvola_weight=opts["sauvola_weight"],
        progress=progress,
    )

    # Cell 17
    export_channel_histograms(
        im_final_stack, stain_complete_df, input_file,
        ROI_print=ROI_print, lazy_loading_used=used_lazy_loading,
        name_setup=name_setup,
        nuclei_diameter=nuclei_diameter, cell_diameter=cell_diameter,
        cyto_factor=cyto_factor, pcm_thickness_factor=pcm_thickness_factor,
        zoom_factors=zooms, scale_factor=scale_factor,
        trig_cellpose=trig_cellpose, trig_stardist=trig_stardist,
        trig_cellpose_cyto=trig_cellpose_cyto,
        multilabel=multilabel, cyto_markers=cyto_markers,
        nuclei_split_config=nuclei_split_config,
        sigma=sigma, num_plateaus=num_plateaus, plateau_factor=plateau_factor,
        threshold_method=threshold_method,
        aggregate_distance_factor=aggregate_distance_factor,
        progress=progress,
        output_dir=output_dir,
        membrane_markers=list(membrane_markers or []),
        extra_params={
            'Split gamma': split_gamma,
            'Split gamma step': split_gamma_step,
            'Min split piece fraction': min_piece_fraction,
            **{f"Nuclei option {k}": v for k, v in opts.items()},
            'Settings adapted to the image (v1.6.4)': adapted_note,
            'Settings adapted to the histograms (v1.6.41)': histogram_note,
        },
    )

    # Cells 18-22
    im_segmentation_stack = segment_nuclei(
        im_final_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        nuclei_split_config=nuclei_split_config,
        r_zxyz=(r_zX, r_zY, r_zZ),
        nuclei_diameter=nuclei_diameter,
        trig_stardist=trig_stardist,
        trig_cellpose=trig_cellpose,
        iterative_split=iterative_split,
        oversize_factor=oversize_factor,
        max_split_iterations=max_split_iterations,
        merge_undersized=merge_undersized,
        undersize_factor=undersize_factor,
        size_reference=size_reference,
        progress=progress,
        split_gamma=split_gamma,
        split_gamma_step=split_gamma_step,
        min_piece_fraction=min_piece_fraction,
        **{k: opts[k] for k in _BATCH_SEGMENT_NUCLEI_KEYS},
    )
    im_segmentation_stack, stain_complete_df = segment_cytoplasm(
        im_final_stack,
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        cyto_markers=cyto_markers,
        cyto_factor=cyto_factor,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        trig_cellpose_cyto=trig_cellpose_cyto,
        r_zxyz=(r_zX, r_zY, r_zZ),
        membrane_markers=membrane_markers,
        progress=progress,
        split_gamma=split_gamma,
        **cyto_split_config,
    )
    im_segmentation_stack = segment_pcm(
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        cell_diameter=cell_diameter,
        r_zxyz=(r_zX, r_zY, r_zZ),
        pcm_thickness_factor=pcm_thickness_factor,
    )
    im_segmentation_stack = assign_channel_labels(
        im_final_stack,
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        progress=progress,
    )
    im_segmentation_stack = detect_aggregates(
        im_segmentation_stack,
        stain_df=stain_df,
        cell_diameter=cell_diameter,
        r_zxyz=(r_zX, r_zY, r_zZ),
        aggregate_distance_factor=aggregate_distance_factor,
    )

    # _context() looks at this module's globals first, which may still hold
    # the previous image's tables -- register this image's before quantifying.
    set_notebook_context(stain_df=stain_df, stain_complete_df=stain_complete_df)

    # Cells 24 and 26
    percell_mean_df, percell_std_df = compute_percell_marker_intensity_df(
        im_segmentation_stack, im_final_stack,
        stain_complete_df=stain_complete_df,
        progress=progress,
    )
    labels_full_df = build_full_labels_df(
        im_segmentation_stack, im_final_stack,
        stain_complete_df=stain_complete_df,
        r_zX=r_zX, r_zY=r_zY, r_zZ=r_zZ,
        zooms=zooms,
        percell_mean_df=percell_mean_df,
        percell_std_df=percell_std_df,
        progress=progress,
        is_2d=is_2d,
    )

    # Cell 30 -- saved as PNG only, not displayed
    plot_marker_intensity_clouds(
        percell_mean_df,
        stain_complete_df=stain_complete_df,
        im_segmentation_stack=im_segmentation_stack,
        r_xyz=(r_zX, r_zY, r_zZ),
        stain_df=stain_df,
        progress=progress,
        input_file=input_file,
        output_dir=output_dir,
        show=False,
        is_2d=is_2d,
    )
    plt.close("all")

    # Cell 34 (written first: it is the main result and the cheapest export)
    export_quantification_to_excel(
        input_file, original_stain_complete_df, labels_full_df,
        progress=progress, output_dir=output_dir,
    )

    # Cell 27 -- PDF only, PNGs go to a temporary folder
    build_histogram_report(
        im_segmentation_stack, im_final_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        input_file=input_file,
        progress=progress,
        output_dir=output_dir,
        keep_png=False,
        show_plot=False,
    )

    # Cells 31, 32, 35 -- 3D meshes, not made for a 2D image
    if is_2d and (export_vtk or export_stl or export_fea):
        print("2D image: VTK, STL and FEA mesh exports skipped (3D only).")
    if export_vtk and not is_2d:
        build_vtk_volumes(
            im_segmentation_stack,
            labels_full_df=labels_full_df,
            stain_complete_df=stain_complete_df,
            input_file=input_file,
            r_xyz=(r_X, r_Y, r_Z),
            zoom_factors=zooms,
            progress=progress,
            output_dir=output_dir,
            live_counter=False,
        )
    if export_stl and not is_2d:
        export_marker_stl(
            im_segmentation_stack,
            stain_df=stain_df,
            stain_complete_df=stain_complete_df,
            input_file=input_file,
            r_xyz=(r_X, r_Y, r_Z),
            zoom_factors=zooms,
            progress=progress,
            output_dir=output_dir,
        )
    if export_fea and not is_2d:
        export_fea_mesh(
            im_segmentation_stack,
            input_file=input_file,
            r_zxyz=(r_zX, r_zY, r_zZ),
            progress=progress,
            output_dir=output_dir,
        )


def run_batch_folder(
    input_folder,
    roi_coords,
    stain_dict,
    cyto_markers,
    name_setup,
    use_setup=True,
    automatic_contrast=False,
    interactive_roi=False,
    nuclei_diameter=10.0,
    cell_diameter=30.0,
    scale_factor=1.0,
    zoom_factors=None,
    trig_cellpose=False,
    trig_stardist=False,
    trig_cellpose_cyto=False,
    multilabel=True,
    aggregate_distance_factor=0.05,
    calibrated_contrast=None,
    nuclei_split_config=None,
    cyto_split_config=None,
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    size_reference="blend",
    sigma=0.5,
    num_plateaus=2,
    plateau_factor=0.7,
    threshold_method='otsu',
    export_vtk=True,
    export_stl=True,
    export_fea=True,
    settings=None,
    napari_module=None,
    progress=None,
    membrane_markers=None,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.0,
    pcm_thickness_factor=0.05,
    nuclei_options=None,
):
    """Process every supported image file in *input_folder* with the full
    pipeline and write each image's outputs into its own subfolder,
    ``<input_folder>_output/<image stem>/``.

    ``nuclei_options`` (v1.6.4): the NUCLEI settings added in the nuclei
    notebook v1.6.4 (depth correction, haze removal, threshold
    weights, Z elongation, flare trim, saddle / tall split, rescue, edges,
    label growth) as a dict - keys and "off" values in
    `BATCH_NUCLEI_OPTIONS`; missing keys stay off. Before v1.6.4 the batch
    mode ran without them.

    Each file goes through the same steps as the single-file notebook
    (Cells 5-35) with the same settings, but without napari viewers, inline
    plots, the optional single-nucleus export, or the crop_png/merged_png/
    density_png subfolders (the PNGs are only needed to build the PDF, so
    they are written to a temporary folder and deleted). Output files keep
    the ``<image stem>_...`` naming of the single-file mode.

    Contrast/gamma settings come from ``<name_setup>_setup.csv`` like in
    Cell 8: if it doesn't exist yet (or ``use_setup`` is False), the napari
    setup viewer opens -- or, with ``calibrated_contrast=True`` (or
    ``automatic_contrast``), the limits are picked from each channel's
    histogram -- and the CSV written for the first image is then reused for
    the following ones (as long as ``use_setup`` is True).

    ``membrane_markers`` (from the stain table) are filled as rings into the
    cytoplasm and used as cell borders when splitting it (Cell 19).

    A file that raises an error is reported and skipped; the batch carries on
    with the next one.

    Returns
    -------
    summary : DataFrame
        One row per file with its status ('OK'/'FAILED'), output subfolder
        and error message.
    """
    import traceback
    from helpers.notebook_setup_helpers import resolve_stardist_flag

    # Resolved once here so the warning isn't repeated for every image.
    trig_stardist = resolve_stardist_flag(trig_stardist)
    unknown = set(nuclei_options or {}) - set(BATCH_NUCLEI_OPTIONS)
    if unknown:  # fail before the first file, not once per file
        raise ValueError(f"Unknown nuclei_options: {sorted(unknown)}; "
                         f"known: {sorted(BATCH_NUCLEI_OPTIONS)}")
    automatic_contrast = _contrast_mode(automatic_contrast, calibrated_contrast)

    image_files = list_image_files(input_folder)
    if not image_files:
        raise FileNotFoundError(
            f"No supported image files ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) "
            f"found in {input_folder}"
        )

    output_dir = get_batch_output_dir(input_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if zoom_factors is None:
        zoom_factors = [1.0, 1.0, 1.0]
    if nuclei_split_config is None:
        nuclei_split_config = get_nuclei_split_config(profile="balanced")
    if cyto_split_config is None:
        cyto_split_config = get_cyto_split_config(profile="balanced")

    print(f"Batch mode: {len(image_files)} image file(s) found in {input_folder}")
    print(f"Outputs are written to: {output_dir}")

    results = []
    for n, image_path in enumerate(image_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{n}/{len(image_files)}] {image_path.name}")
        print("=" * 80)
        image_output_dir = output_dir / _image_stem(image_path)
        try:
            image_output_dir.mkdir(parents=True, exist_ok=True)
            _process_single_image_batch(
                str(image_path),
                image_output_dir,
                roi_coords=roi_coords,
                stain_dict=stain_dict,
                cyto_markers=cyto_markers,
                name_setup=name_setup,
                use_setup=use_setup,
                automatic_contrast=automatic_contrast,
                interactive_roi=interactive_roi,
                nuclei_diameter=nuclei_diameter,
                cell_diameter=cell_diameter,
                scale_factor=scale_factor,
                zoom_factors=zoom_factors,
                trig_cellpose=trig_cellpose,
                trig_stardist=trig_stardist,
                trig_cellpose_cyto=trig_cellpose_cyto,
                multilabel=multilabel,
                aggregate_distance_factor=aggregate_distance_factor,
                nuclei_split_config=nuclei_split_config,
                cyto_split_config=cyto_split_config,
                iterative_split=iterative_split,
                oversize_factor=oversize_factor,
                max_split_iterations=max_split_iterations,
                merge_undersized=merge_undersized,
                undersize_factor=undersize_factor,
                size_reference=size_reference,
                sigma=sigma,
                num_plateaus=num_plateaus,
                plateau_factor=plateau_factor,
                threshold_method=threshold_method,
                export_vtk=export_vtk,
                export_stl=export_stl,
                export_fea=export_fea,
                settings=settings,
                napari_module=napari_module,
                progress=progress,
                membrane_markers=membrane_markers,
                split_gamma=split_gamma,
                split_gamma_step=split_gamma_step,
                min_piece_fraction=min_piece_fraction,
                pcm_thickness_factor=pcm_thickness_factor,
                nuclei_options=nuclei_options,
            )
            results.append({
                "File": image_path.name,
                "Status": "OK",
                "Output folder": str(image_output_dir),
                "Error": "",
            })
        except Exception as exc:
            traceback.print_exc()
            print(f"[run_batch_folder] {image_path.name} FAILED -- continuing with the next file.")
            results.append({
                "File": image_path.name,
                "Status": "FAILED",
                "Output folder": str(image_output_dir),
                "Error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            # Figures are not shown in batch mode; closing them also stops
            # the inline backend from dumping them all at the end of the cell.
            plt.close("all")
            gc.collect()

    summary = pd.DataFrame(results)
    n_ok = int((summary["Status"] == "OK").sum())
    print("\n" + "=" * 80)
    print(f"Batch finished: {n_ok}/{len(summary)} file(s) processed successfully.")
    print(f"Outputs: {output_dir}")
    return summary


# ---------------------------------------------------------------------------
# LIVE/DEAD notebook (NucleoSeg3D_LD, since v1.2.1)
# ---------------------------------------------------------------------------
# The LD notebook follows the nuclei notebook (v1.6.2) step by step. Its stain
# table has fixed LIVE and DEAD rows (plus optional extra stains) and, instead
# of the intracellular/membrane ticks, asks whether each marker is NUCLEAR or
# CYTOPLASMIC. Everything else follows from that:
#   - the nuclear markers are merged into the nuclei mask (Cell 17);
#   - the cytoplasmic markers are merged into the cell-body mask (Cell 18),
#     and nuclei and cell bodies are combined into one label per cell;
#   - each marker's positivity is measured in its own compartment (nucleus
#     or whole cell) and every cell gets a binary LIVE/DEAD call (Cell 20).

LD_LOCALIZATIONS = ("nuclear", "cytoplasmic")
LD_RULES = ("dead_priority", "coverage")
_LD_FIXED_CONDITIONS = ("LIVE", "DEAD")
# Keywords in the channel name used to pre-fill the LIVE/DEAD rows.
_LD_CHANNEL_HINTS = {
    "LIVE": ("CALCEIN", "SYTO", "488", "GFP", "FITC", "GREEN"),
    "DEAD": ("ETHD", "ETHIDIUM", "PROPIDIUM", "7-AAD", "561", "555", "568", "594",
             "CY3", "RFP", "MCHERRY", "TRITC", "TXRED", "RED"),
}
_LD_NUCLEAR_HINTS = ("DAPI", "HOECHST", "405", "DRAQ", "SYTO", "ETHD", "PROPIDIUM", "7-AAD")


def ld_role(condition):
    """v1.2.2: the part a condition plays in the LIVE/DEAD call: 'LIVE' for
    LIVE and LIVE_<n> (e.g. a Hoechst counterstain added as LIVE_2: a cell
    with that stain only is LIVE), 'DEAD' for DEAD and DEAD_<n>, None for any
    other stain (segmented and quantified, not in the call)."""
    name = str(condition).upper()
    for role in _LD_FIXED_CONDITIONS:
        if name == role or re.fullmatch(rf"{role}_\d+", name):
            return role
    return None


def ld_call_conditions(conditions, role):
    """The conditions of *conditions* in the LIVE or DEAD group (`ld_role`)."""
    return [c for c in conditions if ld_role(c) == role]


def _guess_ld_channel(channels, condition, taken):
    free = [c for c in channels if c not in taken]
    keys = _LD_CHANNEL_HINTS.get(condition, ())
    # v1.2.2, separate LIVE and DEAD files: the channels of the LIVE (DEAD)
    # file are named 'LIVE - ...' ('DEAD - ...'): take one of those first.
    from_file = [c for c in free if str(c).upper().startswith(f"{condition} - ")]
    if from_file:
        hinted = [c for c in from_file if any(key in str(c).upper()[len(condition) + 3:]
                                              for key in keys)]
        return (hinted or from_file)[0]
    free = [c for c in free if not any(str(c).upper().startswith(f"{other} - ")
                                       for other in _LD_FIXED_CONDITIONS if other != condition)] \
        or free
    for channel in free:
        if any(key in str(channel).upper() for key in keys):
            return channel
    return free[0] if free else _STAIN_NO_CHANNEL


def _guess_ld_localization(channel, condition):
    if condition == "LIVE":
        return "cytoplasmic"
    if condition == "DEAD":
        return "nuclear"
    name = str(channel).upper()
    return "nuclear" if any(key in name for key in _LD_NUCLEAR_HINTS) else "cytoplasmic"


def _default_ld_stain_rows(channels):
    """Preliminary LIVE/DEAD stain table: LIVE on a green-like channel
    (cytoplasmic, e.g. calcein) and DEAD on a red-like channel (nuclear, e.g.
    ethidium homodimer or propidium iodide). v1.2.2: only these two rows --
    any other stain (e.g. a Hoechst counterstain) is added with 'Add stain';
    the channels left out are listed below the table."""
    taken, rows = [], []
    for condition in _LD_FIXED_CONDITIONS:
        channel = _guess_ld_channel(channels, condition, taken)
        if channel != _STAIN_NO_CHANNEL:
            taken.append(channel)
        rows.append(dict(condition=condition, marker="", channel=channel,
                         color="green" if condition == "LIVE" else "red",
                         localization=_guess_ld_localization(channel, condition)))
    return rows


class LDStainDictEditor:
    """Interactive LIVE/DEAD stain table (ipywidgets), the LD counterpart of
    :class:`StainDictEditor`.

    One row per condition: condition name (fixed for LIVE and DEAD), marker
    name (text), channel (drop-down of the file's channels, or 'no' if the
    condition isn't used), display color and **localization**: whether the
    marker stains the nucleus ('Nuclear', e.g. ethidium homodimer, propidium
    iodide, Hoechst) or the cytoplasm ('Cytoplasmic', e.g. calcein). Extra
    stains (e.g. a nuclear counterstain) can be added with 'Add stain': they
    help the segmentation and are quantified, but don't enter the LIVE/DEAD
    call. Clicking 'Confirm' checks the table and saves it to
    ``<name_setup>_ld_stain_dict.json``; a saved table is loaded (and counts
    as confirmed) the next time the notebook runs. Read the result with
    :meth:`result`.
    """

    _COLUMNS = ("Condition", "Marker", "Channel", "Color", "Localization", "")
    _LOCALIZATION_OPTIONS = (("Nuclear", "nuclear"), ("Cytoplasmic", "cytoplasmic"))
    _MARKER_HINTS = {"LIVE": "e.g. CALCEIN", "DEAD": "e.g. ETHD-1"}

    def __init__(self, channels, name_setup=None, use_saved=True):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "The interactive stain table needs ipywidgets: "
                "uv pip install ipywidgets"
            ) from exc
        self._w = widgets
        self.channels = [str(c) for c in (channels or [])]
        self.save_path = f"{name_setup}_ld_stain_dict.json" if name_setup else None
        self._rows = []
        self._confirmed = False
        self._load_notes = []

        loaded = self._load_saved() if use_saved else None
        for row in loaded or _default_ld_stain_rows(self.channels):
            self._add_row(row, render=False)

        self._grid = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="120px 180px 240px 110px 140px 40px",
            grid_gap="4px 10px",
            align_items="center",
        ))
        add_button = widgets.Button(description="Add other stain", icon="plus",
                                    tooltip="Segmented and quantified, not in the LIVE/DEAD call")
        add_button.on_click(lambda _: self._add_row(self._new_stain_row()))
        # v1.2.2: extra markers of the call, e.g. Hoechst as LIVE_2
        add_live = widgets.Button(description="Add LIVE stain", icon="plus",
                                  tooltip="Another LIVE marker (LIVE_2, ...): a cell positive "
                                          "for it only is LIVE, with a DEAD marker too DEAD")
        add_live.on_click(lambda _: self._add_row(self._new_stain_row("LIVE")))
        add_dead = widgets.Button(description="Add DEAD stain", icon="plus",
                                  tooltip="Another DEAD marker (DEAD_2, ...)")
        add_dead.on_click(lambda _: self._add_row(self._new_stain_row("DEAD")))
        confirm_button = widgets.Button(description="Confirm", icon="check",
                                        button_style="success")
        confirm_button.on_click(self._on_confirm)
        self._status = widgets.HTML()
        self._output = widgets.Output()
        self.widget = widgets.VBox([
            widgets.HTML("<b>LIVE/DEAD stain table</b> -- pick each condition's channel "
                         "('no' = not used) and color, type the marker names, choose "
                         "whether each marker is <b>nuclear</b> or <b>cytoplasmic</b>, "
                         "then click <b>Confirm</b>.<br>Extra stains: <b>Add LIVE stain</b> "
                         "(LIVE_2, ...; e.g. Hoechst: a cell with it only is LIVE, with a DEAD "
                         "marker too it is DEAD), <b>Add DEAD stain</b> (DEAD_2, ...) or "
                         "<b>Add other stain</b> (not in the call)."),
            self._grid,
            widgets.HBox([add_live, add_dead, add_button, confirm_button]),
            self._status,
            self._output,
        ])
        self._render()
        errors, _ = self._validate()
        self._confirmed = loaded is not None and not errors and not self._load_notes
        self._update_status()

    # -- saved table --------------------------------------------------------
    def _load_saved(self):
        if not self.save_path or not os.path.exists(self.save_path):
            return None
        try:
            with open(self.save_path, encoding="utf-8") as f:
                rows = json.load(f)["rows"]
        except Exception as exc:
            print(f"[LDStainDictEditor] Could not read {self.save_path} ({exc}); "
                  f"starting from a new table.")
            return None
        # LIVE and DEAD always come first, even if missing from the file.
        saved = {r.get("condition"): r for r in rows}
        defaults = {r["condition"]: r for r in _default_ld_stain_rows(self.channels)}
        rows = ([dict(saved.get(c, defaults[c])) for c in _LD_FIXED_CONDITIONS]
                + [dict(r) for r in rows if r.get("condition") not in _LD_FIXED_CONDITIONS])
        for row in rows:
            channel = row.get("channel", _STAIN_NO_CHANNEL)
            if channel != _STAIN_NO_CHANNEL and channel not in self.channels:
                self._load_notes.append(
                    f"{row.get('condition')}: saved channel '{channel}' is not in "
                    f"this file -- set to 'no'.")
                row["channel"] = _STAIN_NO_CHANNEL
            if row.get("localization") not in LD_LOCALIZATIONS:
                row["localization"] = _guess_ld_localization(channel, row.get("condition", ""))
        print(f"Stain table loaded from {self.save_path}"
              + ("" if not self._load_notes else " (needs review, see below)."))
        return rows

    # -- rows ---------------------------------------------------------------
    def _new_stain_row(self, role=None):
        taken = {r["condition"] for r in self._values()}
        if role:  # v1.2.2: LIVE_2, DEAD_2, ... (extra markers of the call)
            n = 2
            while f"{role}_{n}" in taken:
                n += 1
            used = {r["channel"] for r in self._values()}
            free = [c for c in self.channels if c not in used]
            channel = free[0] if free else _STAIN_NO_CHANNEL
            return dict(condition=f"{role}_{n}", marker="", channel=channel,
                        color=_guess_channel_color(channel, n + 1),
                        localization=_guess_ld_localization(channel, ""))
        n = 1
        while f"STAIN_{n}" in taken:
            n += 1
        # v1.2.2: the first channel not used yet, with its guessed color and
        # localization (e.g. a Hoechst / DAPI channel -> nuclear, blue).
        used = {r["channel"] for r in self._values()}
        free = [c for c in self.channels if c not in used]
        if free:
            return dict(condition=f"STAIN_{n}", marker="", channel=free[0],
                        color=_guess_channel_color(free[0], n + 1),
                        localization=_guess_ld_localization(free[0], ""))
        return dict(condition=f"STAIN_{n}", marker="", channel=_STAIN_NO_CHANNEL,
                    color=STAIN_COLORS[n % len(STAIN_COLORS)], localization="nuclear")

    def _add_row(self, row, render=True):
        w = self._w
        fixed = row["condition"] in _LD_FIXED_CONDITIONS
        auto = w.Layout(width="auto")
        if fixed:
            condition = w.HTML(f"<b>{row['condition']}</b>")
        else:
            condition = w.Text(value=row["condition"], placeholder="e.g. STAIN_1", layout=auto)
        marker = w.Text(value=row.get("marker", ""), layout=auto,
                        placeholder=self._MARKER_HINTS.get(row["condition"], "marker name"))
        options = [_STAIN_NO_CHANNEL] + self.channels
        channel_value = row.get("channel", _STAIN_NO_CHANNEL)
        channel = w.Dropdown(options=options, layout=auto,
                             value=channel_value if channel_value in options else _STAIN_NO_CHANNEL)
        color_value = row.get("color", "white")
        color = w.Dropdown(options=list(STAIN_COLORS), layout=auto,
                           value=color_value if color_value in STAIN_COLORS else "white")
        loc_value = row.get("localization")
        localization = w.Dropdown(options=list(self._LOCALIZATION_OPTIONS), layout=auto,
                                  value=loc_value if loc_value in LD_LOCALIZATIONS else "nuclear")
        if fixed:
            remove = w.HTML("")
        else:
            remove = w.Button(icon="times", tooltip="Remove this stain",
                              layout=w.Layout(width="36px"))
        entry = dict(fixed=fixed, name=row["condition"], condition=condition,
                     marker=marker, channel=channel, color=color,
                     localization=localization, remove=remove)
        if not fixed:
            remove.on_click(lambda _, e=entry: self._remove_row(e))
            condition.observe(self._on_change, names="value")
        for widget in (marker, channel, color, localization):
            widget.observe(self._on_change, names="value")
        self._rows.append(entry)
        if render:
            self._render()
            self._on_change()

    def _remove_row(self, entry):
        self._rows.remove(entry)
        self._render()
        self._on_change()

    def _render(self):
        header = [self._w.HTML(f"<b>{c}</b>") for c in self._COLUMNS]
        cells = []
        for e in self._rows:
            cells += [e["condition"], e["marker"], e["channel"], e["color"],
                      e["localization"], e["remove"]]
        self._grid.children = header + cells

    def _values(self):
        return [
            dict(condition=(e["name"] if e["fixed"] else e["condition"].value).strip().upper(),
                 marker=e["marker"].value.strip(),
                 channel=e["channel"].value,
                 color=e["color"].value,
                 localization=e["localization"].value)
            for e in self._rows
        ]

    # -- validation / status ------------------------------------------------
    def _validate(self):
        rows = self._values()
        errors, notes = [], list(self._load_notes)
        names = [r["condition"] for r in rows]
        if "" in names:
            errors.append("Every stain needs a condition name.")
        dup_names = sorted({n for n in names if n and names.count(n) > 1})
        if dup_names:
            errors.append(f"Condition name used more than once: {dup_names}")
        used = [r for r in rows if r["channel"] != _STAIN_NO_CHANNEL]
        for r in used:
            if not r["marker"]:
                errors.append(f"{r['condition'] or '(unnamed)'}: type a marker name.")
        used_channels = [r["channel"] for r in used]
        dup_channels = sorted({c for c in used_channels if used_channels.count(c) > 1})
        if dup_channels:
            errors.append(f"Channel assigned to more than one condition: {dup_channels}")
        markers = [r["marker"].upper() for r in used if r["marker"]]
        dup_markers = sorted({m for m in markers if markers.count(m) > 1})
        if dup_markers:
            errors.append(f"Marker name used more than once: {dup_markers}")

        # v1.2.2: LIVE_<n> / DEAD_<n> rows count as LIVE / DEAD markers
        live_used = any(ld_role(r["condition"]) == "LIVE" for r in used)
        dead_used = any(ld_role(r["condition"]) == "DEAD" for r in used)
        if not (live_used or dead_used):
            errors.append("Assign a channel to LIVE and/or DEAD: the live/dead call "
                          "needs at least one of them.")
        elif not dead_used:
            notes.append("No DEAD channel: cells that are not LIVE-positive are called DEAD.")
        elif not live_used:
            notes.append("No LIVE channel: cells that are not DEAD-positive are called LIVE.")
        else:
            notes.append("Cells positive for no LIVE and no DEAD marker are UNCLASSIFIED "
                         "(left out of the viability). A counterstain that should make a cell "
                         "LIVE (e.g. Hoechst) goes in with 'Add LIVE stain'.")
        if used and not any(r["localization"] == "nuclear" for r in used):
            notes.append("No nuclear marker: each cell's nucleus is estimated as a "
                         "nucleus-sized sphere at the centre of its cell body.")
        if used and not any(r["localization"] == "cytoplasmic" for r in used):
            notes.append("No cytoplasmic marker: each cell is its stained nucleus.")
        unused = [c for c in self.channels if c not in used_channels]
        if unused:
            notes.append(f"Channel(s) not used (dropped from the analysis): {unused}")
        return errors, notes

    def _on_change(self, _change=None):
        self._confirmed = False
        self._load_notes = []
        self._update_status()

    def _update_status(self):
        import html
        errors, notes = self._validate()
        if errors:
            head = "<span style='color:#d9534f'><b>Fix before confirming:</b></span>"
            items = errors
        elif self._confirmed:
            head = "<span style='color:#2e8b57'><b>Confirmed.</b></span>"
            items = []
        else:
            head = ("<span style='color:#e69500'><b>Not confirmed yet</b> -- "
                    "click Confirm.</span>")
            items = []
        lines = [f"<li>{html.escape(t)}</li>" for t in items]
        lines += [f"<li><i>{html.escape(t)}</i></li>" for t in notes]
        self._status.value = head + (f"<ul>{''.join(lines)}</ul>" if lines else "")

    def _on_confirm(self, _button):
        errors, _ = self._validate()
        if errors:
            self._update_status()
            return
        self._confirmed = True
        self._update_status()
        self._output.clear_output()
        with self._output:
            if self.save_path:
                with open(self.save_path, "w", encoding="utf-8") as f:
                    json.dump({"rows": self._values()}, f, indent=2)
                print(f"Saved to {self.save_path}")
            print(f"stain_dict = {self.stain_dict}")
            print(f"localization = {self.localization}")

    # -- results ------------------------------------------------------------
    @property
    def stain_dict(self):
        """``{condition: [marker, channel, color]}`` for every used condition."""
        return {r["condition"]: [r["marker"], r["channel"], r["color"]]
                for r in self._values() if r["channel"] != _STAIN_NO_CHANNEL}

    @property
    def localization(self):
        """``{condition: 'nuclear' | 'cytoplasmic'}`` for every used condition."""
        return {r["condition"]: r["localization"]
                for r in self._values() if r["channel"] != _STAIN_NO_CHANNEL}

    def result(self):
        """Return ``(stain_dict, localization)``; raise StainTableNotConfirmed
        if the table has not been confirmed (or has been edited since)."""
        errors, _ = self._validate()
        if errors or not self._confirmed:
            raise StainTableNotConfirmed(
                "The stain table in Cell 4 is not confirmed yet: fill it in, "
                "click 'Confirm', then run the notebook again from Cell 5."
            )
        return self.stain_dict, self.localization


def edit_ld_stain_dict(channels, name_setup=None, use_saved=True):
    """Display the interactive LIVE/DEAD stain table for *channels* and return
    the :class:`LDStainDictEditor` (read it later with ``.result()``)."""
    from IPython.display import display
    editor = LDStainDictEditor(channels, name_setup=name_setup, use_saved=use_saved)
    display(editor.widget)
    return editor


def _ld_localization_map(localization):
    return {str(k).upper(): str(v).lower() for k, v in (localization or {}).items()}


def _ld_channel_groups(stain_df, localization):
    """Channel indices ``(nuclear, cytoplasmic)`` -- channel c of the image is
    row c of *stain_df*."""
    loc = _ld_localization_map(localization)
    nuclear, cytoplasmic = [], []
    for c, cond in enumerate(stain_df.index):
        value = loc.get(str(cond).upper())
        if value == "nuclear":
            nuclear.append(c)
        elif value == "cytoplasmic":
            cytoplasmic.append(c)
        else:
            raise ValueError(
                f"No localization ('nuclear' or 'cytoplasmic') for condition '{cond}'. "
                "Confirm the stain table in Cell 4 and re-run Cell 5."
            )
    return nuclear, cytoplasmic


def _ld_marker_names(stain_df, channels):
    return ", ".join(f"{stain_df.index[c]} ({stain_df['Marker'].iloc[c]})" for c in channels)


def _ld_merged_substack(im_final_stack, channels):
    """One-channel stack (Z, Y, X, 1) merging *channels*: union of their
    threshold masks, per-voxel max of their filtered and denoised
    intensities. Lets `segment_nuclei` run on a group of markers as if they
    were one NUCLEI channel."""
    thr = im_final_stack['Threshold image']
    merged = {
        'Threshold image': np.any(thr[..., channels] > 0, axis=-1)[..., None].astype(np.uint8),
    }
    for key in ('Filtered image', 'Denoised image'):
        merged[key] = np.max(im_final_stack[key][..., channels], axis=-1)[..., None]
    return merged


def _ld_single_channel_df(marker):
    return pd.DataFrame({'Marker': [marker], 'Laser': [''], 'Color': ['']},
                        index=pd.Index(['NUCLEI'], name='Condition'))


# -- 2D images (single plane) ------------------------------------------------
# The 3D watershed and its size refinement erode along Z, which wipes out a
# one-plane image, so 2D images get their own segmentation: the same steps
# (watershed on the distance map with an intensity-aware split, reference
# size, merge of undersized fragments, escalating re-split of oversized
# labels) with every size measured as an area.

_LD_SPLIT_LEVELS_2D = (
    # (name, peak spacing as a fraction of the radius, intensity weight,
    #  minimum intensity dip between two seeds)
    ("aggressive", 0.4, 1.0, 0.06),
    ("very aggressive", 0.3, 1.5, 0.04),
    ("extreme", 0.2, 2.0, 0.02),
)
# First pass: two intensity maxima become separate seeds when the signal
# between them dips by at least this fraction of the object's intensity range.
_LD_INTENSITY_DIP_2D = 0.1


def _ld_edt(region, spacing):
    """Distance map of *region* (any dimension), padded so objects touching
    the array edge are not treated as extending beyond it."""
    inner = tuple(slice(1, -1) for _ in range(region.ndim))
    return ndi.distance_transform_edt(np.pad(region, 1), sampling=spacing)[inner]


def _ld_norm_in_mask(arr, mask):
    values = arr[mask]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _ld_disk_area_px(diameter_um, pixel_yx):
    return max(1.0, np.pi * (diameter_um / 2.0) ** 2 / (pixel_yx[0] * pixel_yx[1]))


def absorb_small_split_pieces(labels, parents, min_size):
    """Second merge round, limited to the children of each split (2D or 3D).

    *parents* labels the regions each split came from (e.g. the connected
    islands of the threshold mask). Within every parent, the pieces of
    *labels* smaller than *min_size* pixels/voxels are merged into the
    closest big piece of the same parent (`_absorb_small_pieces`); pieces of
    different parents are never merged. Label ids are kept (no renumbering).

    Returns ``(labels, n_absorbed)``.
    """
    labels = np.array(labels, dtype=np.int32, copy=True)
    n_absorbed = 0
    for i, sl in enumerate(ndi.find_objects(parents), start=1):
        if sl is None:
            continue
        inside = parents[sl] == i
        crop = np.where(inside, labels[sl], 0)
        if np.count_nonzero(np.unique(crop)) < 2:
            continue
        crop, n = _absorb_small_pieces(crop, min_size)
        if n:
            labels[sl][inside] = crop[inside]
            n_absorbed += n
    return labels, n_absorbed


def _absorb_small_pieces(pieces, min_size):
    """Second merge round, limited to the children of one split.

    *pieces* holds the pieces of ONE split object (0 = outside). Every piece
    smaller than *min_size* pixels -- a one- or two-pixel sliver left by an
    over-eager split -- is merged, smallest first, into the closest big
    piece: the big piece it shares the longest border with, or, if it touches
    none, the nearest big piece. (If no piece is big enough, the biggest
    touching one is used.) Unlike the fragment merge of the size refinement
    there is no seam test: pieces this small are never a cell on their own.
    Label ids are kept (no renumbering). Returns ``(pieces, n_absorbed)``.
    """
    pieces = np.array(pieces, dtype=np.int32, copy=True)
    ids, counts = np.unique(pieces[pieces > 0], return_counts=True)
    area = dict(zip(ids.tolist(), counts.tolist()))
    if len(area) < 2:
        return pieces, 0
    struct = ndi.generate_binary_structure(pieces.ndim, 1)
    n_absorbed = 0
    while len(area) > 1:
        small = [i for i in sorted(area, key=area.get) if area[i] < min_size]
        if not small:
            break
        a = small[0]
        mask_a = pieces == a
        ring = ndi.binary_dilation(mask_a, structure=struct) & ~mask_a & (pieces > 0)
        nb_ids, contact = np.unique(pieces[ring], return_counts=True)
        if nb_ids.size:
            touching = list(zip(contact.tolist(), nb_ids.tolist()))
            big = [(c, n) for c, n in touching if area[n] >= min_size]
            target = max(big or touching)[1]
        else:
            others = [i for i in area if i != a]
            candidates = [i for i in others if area[i] >= min_size] or others
            dist, idx = ndi.distance_transform_edt(~np.isin(pieces, candidates),
                                                   return_indices=True)
            pos = np.unravel_index(int(np.argmin(np.where(mask_a, dist, np.inf))), pieces.shape)
            target = int(pieces[tuple(i[pos] for i in idx)])
        pieces[mask_a] = target
        area[target] += area.pop(a)
        n_absorbed += 1
    return pieces, n_absorbed


def _ld_watershed_2d(mask, intensity, pixel_yx, diameter_um, peak_fraction=0.5,
                     intensity_weight=1.0, intensity_dip=_LD_INTENSITY_DIP_2D,
                     min_piece_px=None):
    """Split a 2D mask into objects of about *diameter_um* and flood them on
    a surface combining the distance and the intensity, so touching objects
    split along the dimmer seam between them.

    Seeds, per island of the mask:

    - **intensity maxima**: when the island holds two or more maxima of the
      (smoothed) intensity separated by a dip of at least ``intensity_dip``
      x the intensity range inside the mask (the object itself, when a
      single label is re-split), each maximum is a seed. Touching cells
      that form one round blob have a single distance peak but one bright
      centre each, so this is what splits them; a higher split gamma deepens
      the dips and gives more seeds;
    - otherwise **distance peaks**: one per peak of the distance map (at
      least ``peak_fraction`` x the radius apart), at least one per island.

    With ``min_piece_px``, the pieces of each island smaller than that are
    then merged into the closest big piece of the same island
    (`absorb_small_split_pieces`).

    Returns ``(labels, n_absorbed)``.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import relabel_sequential

    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return labels, 0
    radius_px = diameter_um / 2.0 / float(np.mean(pixel_yx))
    distance = ndi.gaussian_filter(_ld_edt(mask, pixel_yx), sigma=1.0)
    smoothed = ndi.gaussian_filter(np.asarray(intensity, dtype=np.float32), sigma=1.0)
    islands, n_islands = ndi.label(mask)

    # Intensity maxima separated by a dip of at least intensity_dip, per island.
    inten_norm = np.where(mask, _ld_norm_in_mask(smoothed, mask), 0.0)
    maxima, n_max = ndi.label(h_maxima(inten_norm.astype(np.float64), float(intensity_dip)) > 0)
    maxima[~mask] = 0
    max_island = np.zeros(n_max + 1, dtype=np.int64)
    if n_max:
        max_island[1:] = np.asarray(
            ndi.maximum(islands, labels=maxima, index=np.arange(1, n_max + 1)), dtype=np.int64
        )
    maxima_per_island = np.bincount(max_island[1:], minlength=n_islands + 1)
    use_intensity = maxima_per_island >= 2
    use_intensity[0] = False

    markers = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    if n_max:
        lut = np.zeros(n_max + 1, dtype=np.int32)
        for k in range(1, n_max + 1):
            if use_intensity[max_island[k]]:
                lut[k] = next_id
                next_id += 1
        markers = lut[maxima]

    # Distance peaks for the islands without several intensity maxima.
    peaks = peak_local_max(
        distance,
        min_distance=max(1, int(round(peak_fraction * radius_px))),
        labels=islands,
        exclude_border=False,
    )
    for coord in np.asarray(peaks).reshape(-1, 2):
        if not use_intensity[islands[tuple(coord)]]:
            markers[tuple(coord)] = next_id
            next_id += 1
    has_marker = np.zeros(n_islands + 1, dtype=bool)
    has_marker[np.unique(islands[markers > 0])] = True
    island_slices = ndi.find_objects(islands)
    for i in np.flatnonzero(~has_marker[1:]) + 1:
        sl = island_slices[i - 1]
        local = np.where(islands[sl] == i, distance[sl], -1.0)
        pos = np.unravel_index(int(np.argmax(local)), local.shape)
        markers[sl][pos] = next_id
        next_id += 1
    elevation = -(_ld_norm_in_mask(distance, mask)
                  + intensity_weight * _ld_norm_in_mask(smoothed, mask))
    labels = watershed(elevation, markers, mask=mask).astype(np.int32)

    # Second merge round: only the pieces of each split island are involved.
    n_absorbed = 0
    if min_piece_px:
        labels, n_absorbed = absorb_small_split_pieces(labels, islands, min_piece_px)
        labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32), n_absorbed


def _ld_reference_area_2d(labels, diameter_um, pixel_yx, size_reference="blend",
                          plausible_range=(0.25, 4.0), min_labels=5):
    """2D counterpart of `estimate_reference_nucleus_size`: the reference
    object area (pixels) from a disk of *diameter_um* and/or the median area
    of the plausible labels."""
    choices = ("blend", "input", "detected")
    if size_reference not in choices:
        raise ValueError(f"Unknown size_reference '{size_reference}'. Choose from {choices}.")
    input_px = _ld_disk_area_px(diameter_um, pixel_yx)
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    areas = areas[areas > 0]
    lo, hi = plausible_range
    plausible = areas[(areas >= lo * input_px) & (areas <= hi * input_px)]
    detected_px = float(np.median(plausible)) if plausible.size >= min_labels else None
    used = size_reference
    if detected_px is None or size_reference == "input":
        reference_px, used = input_px, "input"
    elif size_reference == "detected":
        reference_px = detected_px
    else:
        reference_px = float(np.sqrt(input_px * detected_px))
    px_um2 = pixel_yx[0] * pixel_yx[1]
    print(f"Reference size: input diameter {diameter_um:g} um -> {input_px * px_um2:.1f} um2")
    if detected_px is None:
        print(f"  Detected: only {plausible.size} plausible label(s) (< {min_labels}) "
              "-> using the input size.")
    else:
        print(f"  Detected: median of {plausible.size} plausible labels = "
              f"{detected_px * px_um2:.1f} um2 ({detected_px / input_px:.2f}x input)")
    print(f"  Reference ('{used}'): {reference_px * px_um2:.1f} um2 "
          f"(~{diameter_um * np.sqrt(reference_px / input_px):.1f} um diameter)")
    return reference_px


def _ld_split_oversized_2d(labels, intensity, pixel_yx, reference_px, oversize_factor=2.0,
                           max_iterations=3, progress=None, level_gammas=None,
                           min_piece_fraction=0.2):
    """Re-split labels bigger than ``oversize_factor`` x the reference area
    with increasingly aggressive settings (`_LD_SPLIT_LEVELS_2D`). With
    ``level_gammas`` the intensity crop of each label gets that level's gamma
    (`apply_split_gamma`), so the seams darken more at every level. Pieces of
    a split smaller than ``min_piece_fraction`` x the reference area are
    merged back into the closest big piece of the same split
    (`absorb_small_split_pieces`); a label left with one piece is not split.
    Labels that can't be split are kept."""
    from skimage.segmentation import relabel_sequential

    labels = np.asarray(labels, dtype=np.int32).copy()
    ref_diameter = 2.0 * np.sqrt(reference_px * pixel_yx[0] * pixel_yx[1] / np.pi)
    next_id = int(labels.max()) + 1
    min_piece_px = max(1.0, float(min_piece_fraction or 0.0) * reference_px)
    n_levels = int(np.clip(max_iterations, 1, len(_LD_SPLIT_LEVELS_2D)))
    for level in range(n_levels):
        name, peak_fraction, weight, dip = _LD_SPLIT_LEVELS_2D[level]
        gamma = level_gammas[min(level, len(level_gammas) - 1)] if level_gammas else None
        areas = np.bincount(labels.ravel())
        big = np.flatnonzero(areas > oversize_factor * reference_px)
        big = big[big > 0]
        if big.size == 0:
            break
        slices = ndi.find_objects(labels)
        n_split = 0
        n_absorbed = 0
        for lab in _progress_iter(big.tolist(), progress,
                                  desc=f'Split oversized ({name})', leave=False):
            sl = slices[lab - 1]
            if sl is None:
                continue
            region = labels[sl] == lab
            parts, absorbed = _ld_watershed_2d(
                region, apply_split_gamma(intensity[sl], gamma), pixel_yx, ref_diameter,
                peak_fraction=peak_fraction, intensity_weight=weight, intensity_dip=dip,
                min_piece_px=min_piece_px,
            )
            n_absorbed += absorbed
            ids = np.unique(parts[parts > 0])
            if ids.size < 2:
                continue
            view = labels[sl]
            for pid in ids[1:]:
                view[parts == pid] = next_id
                next_id += 1
            n_split += 1
        print(f"  Split level {level + 1} ('{name}'"
              + (f", intensity gamma {gamma:.2f}" if gamma else "")
              + f"): {big.size} oversized label(s) "
              f"(> {oversize_factor:g}x reference), {n_split} split"
              + (f", {n_absorbed} tiny piece(s) merged back" if n_absorbed else ""))
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


def _ld_cellpose_2d(image_2d, diameter_um, pixel_yx, model_type):
    from cellpose import models

    model = models.Cellpose(model_type=model_type, gpu=True)
    diameter_px = diameter_um / float(np.mean(pixel_yx))
    labels, _, _, _ = model.eval(image_2d, diameter=diameter_px, channels=[0, 0])
    print(f"Cellpose 2D (model={model_type}, diameter_px={diameter_px:.1f})")
    return np.asarray(labels, dtype=np.int32)


def _ld_stardist_2d(image_2d, model_name="2D_versatile_fluo"):
    try:
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise ImportError(
            "StarDist requires TensorFlow, csbdeep and stardist, which are not "
            "installed (TensorFlow has no wheels for Python 3.14). Use Python "
            "3.10-3.13 for StarDist, or set trig_stardist=False."
        ) from exc
    model = StarDist2D.from_pretrained(model_name)
    labels, _ = model.predict_instances(normalize(image_2d, 1, 99.8, axis=None))
    return np.asarray(labels, dtype=np.int32)


def _ld_split_gammas(split_gamma=1.0, split_gamma_step=0.0, n_levels=3):
    """Gamma of the first pass and of each re-split level:
    ``split_gamma`` + ``split_gamma_step`` x level (level 1, 2, 3)."""
    base = float(split_gamma or 1.0)
    step = float(split_gamma_step or 0.0)
    return base, [base + step * (level + 1) for level in range(n_levels)]


def _ld_print_split_gamma(what, split_gamma, split_gamma_step, applied=True):
    base, levels = _ld_split_gammas(split_gamma, split_gamma_step)
    if abs(base - 1.0) < 1e-6 and not split_gamma_step:
        return
    first = (f"gamma {base:g} on the intensity used to split touching {what}"
             if applied else
             f"no gamma on the deep-learning input (split_gamma {base:g} not applied)")
    print(f"Split gamma: {first}; re-split levels: "
          + ", ".join(f"{g:g}" for g in levels))


def _ld_segment_objects_2d(
    mask,
    intensity,
    pixel_yx,
    diameter_um,
    what,
    method="watershed",
    cellpose_model="nuclei",
    size_reference="blend",
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.2,
    progress=None,
):
    """Segment a 2D image into objects of about *diameter_um* (nuclei or cell
    bodies) and refine them by area, like the 3D nuclei segmentation:
    watershed (or Cellpose / StarDist in 2D), then undersized fragments
    merged into a touching neighbour when there is no dark seam between them
    (`merge_undersized_nuclei`, run on the single plane) and oversized labels
    re-split (`_ld_split_oversized_2d`). Labels are never deleted.

    ``split_gamma`` is applied to the intensity used by the watershed split
    and by the merge test (dark seams between objects get darker), and grows
    by ``split_gamma_step`` at each re-split level. The deep-learning models
    get the original intensity.

    After every watershed split, the pieces of one split smaller than
    ``min_piece_fraction`` x the expected size (the input diameter for the
    first pass, the reference size for the re-split levels) are merged into
    the closest big piece of the same split (`absorb_small_split_pieces`)."""
    base_gamma, level_gammas = _ld_split_gammas(split_gamma, split_gamma_step)
    split_intensity = apply_split_gamma(intensity, base_gamma)
    _ld_print_split_gamma(what, split_gamma, split_gamma_step)
    if method == "cellpose":
        labels = _ld_cellpose_2d(intensity, diameter_um, pixel_yx, cellpose_model)
    elif method == "stardist":
        labels = _ld_stardist_2d(intensity)
    else:
        filled = _fill_small_holes_per_slice(
            np.asarray(mask, dtype=bool)[np.newaxis], _ld_disk_area_px(diameter_um, pixel_yx)
        )[0]
        min_piece_px = max(1.0, float(min_piece_fraction or 0.0)
                           * _ld_disk_area_px(diameter_um, pixel_yx))
        labels, n_absorbed = _ld_watershed_2d(filled, split_intensity, pixel_yx, diameter_um,
                                              min_piece_px=min_piece_px)
        if n_absorbed:
            print(f"Split pieces smaller than {min_piece_fraction:g}x the expected size "
                  f"({min_piece_px:.0f} px) merged back into a neighbouring piece: {n_absorbed}")
    print(f"Total {what} found: {int(np.count_nonzero(np.unique(labels)))}")
    if not (iterative_split or merge_undersized) or not labels.any():
        return labels

    reference_px = _ld_reference_area_2d(labels, diameter_um, pixel_yx, size_reference)
    if merge_undersized:
        merged, _ = merge_undersized_nuclei(
            labels[np.newaxis], split_intensity[np.newaxis], reference_px,
            undersize_factor=undersize_factor, progress=progress,
        )
        labels = np.asarray(merged[0], dtype=np.int32)
    if iterative_split:
        labels = _ld_split_oversized_2d(
            labels, intensity, pixel_yx, reference_px,
            oversize_factor=oversize_factor, max_iterations=max_split_iterations,
            progress=progress, level_gammas=level_gammas,
            min_piece_fraction=min_piece_fraction,
        )
    print(f"Total {what} after size refinement: {int(np.count_nonzero(np.unique(labels)))}")
    return labels


# -- LIVE/DEAD v1.2.2: the validated NUCLEI settings of the nuclei notebook
# v1.6.4, applied to the markers set as nuclear ---------------------------------
# In the nuclei notebook the per-image steps (calibrated contrast, size
# adaptation, haze removal, depth correction) act on the NUCLEI channel; in
# the LIVE/DEAD notebook the nuclei are segmented from the merged nuclear
# markers, so the same steps act on every nuclear marker.

def ld_nuclear_conditions(stain_df, localization):
    """Conditions of *stain_df* set as nuclear, in channel order."""
    nuclear, _ = _ld_channel_groups(stain_df, localization)
    return [stain_df.index[c] for c in nuclear]


def ld_calibrated_nuclear_contrast(im_original, stain_complete_df, original_stain_complete_df,
                                   nuclear_conditions, gamma=1.0):
    """Contrast / gamma of every nuclear marker from its own histogram (the
    'calibrated' rule the NUCLEI settings were validated with,
    `calibrated_contrast`; ``gamma`` = ``nuclei_gamma`` of the settings),
    written into both stain tables. Returns ``{condition: contrast}``."""
    out = {}
    for cond in nuclear_conditions:
        contrast = calibrated_contrast(im_original[..., stain_complete_df.index.get_loc(cond)],
                                       gamma=gamma)
        for col, value in contrast.items():
            stain_complete_df.loc[cond, col] = value
            original_stain_complete_df.loc[cond, col] = value
        out[cond] = contrast
        print(f"{cond} (nuclear) contrast from its own histogram: Cont_min {contrast['Cont_min']}, "
              f"Cont_max {contrast['Cont_max']}, Gamma {contrast['Gamma']:g}")
    if out:
        print("(the nuclei segmentation settings were validated with this contrast rule)")
    return out


def ld_histogram_features(original_image, stain_complete_df, nuclear_conditions):
    """Histogram features (`helpers.histogram_adapt.channel_histogram_features`)
    of every channel, plus ``'NUCLEI'``: the merged nuclear markers (each
    min-max normalized, then the per-voxel maximum), the channel the NUCLEI
    histogram rules of the settings read in the LIVE/DEAD notebook."""
    from helpers.histogram_adapt import channel_histogram_features, features_by_condition
    feats = features_by_condition(original_image, stain_complete_df)
    merged = None
    for cond in nuclear_conditions:
        ch = np.asarray(original_image[..., stain_complete_df.index.get_loc(cond)], dtype=np.float32)
        lo, hi = float(ch.min()), float(ch.max())
        ch = (ch - lo) / (hi - lo + 1e-8)
        merged = ch if merged is None else np.maximum(merged, ch)
    if merged is not None:
        feats["NUCLEI"] = channel_histogram_features(merged)
    return feats


def ld_adapt_to_histograms(settings, rules, original_image, stain_complete_df,
                           nuclear_conditions, keep=()):
    """Apply the NUCLEI histogram rules (``channel`` 'NUCLEI') of the settings
    to ``settings`` (in place), read on the merged nuclear markers
    (`ld_histogram_features`). The cell-body rules are not used: the
    LIVE/DEAD cell bodies have their own segmentation. Returns
    ``(changes, features)``."""
    from helpers.histogram_adapt import adapt_settings
    nuc_rules = {k: r for k, r in (rules or {}).items()
                 if r.get("channel", "NUCLEI") == "NUCLEI" and k.partition(".")[0] in settings}
    if not nuc_rules or not nuclear_conditions:
        return [], {}
    feats = ld_histogram_features(original_image, stain_complete_df, nuclear_conditions)
    return adapt_settings(settings, nuc_rules, feats, keep=keep), feats


def _ld_nuclear_view(original_image, stain_complete_df, nuclear_conditions):
    """The merged nuclear markers as the segmentation sees them: every
    nuclear channel min-max normalized to 0-255 with its own contrast and
    gamma (stain table), then the per-voxel maximum."""
    merged = None
    for cond in nuclear_conditions:
        row = stain_complete_df.loc[cond]
        ch = np.asarray(original_image[..., stain_complete_df.index.get_loc(cond)], dtype=np.float32)
        lo, hi = float(ch.min()), float(ch.max())
        ch = (ch - lo) / (hi - lo + 1e-8) * 255.0
        ch = napari_contrast_gamma_uint8(ch, (float(row["Cont_min"]), float(row["Cont_max"])),
                                         float(row["Gamma"]))
        merged = ch if merged is None else np.maximum(merged, ch)
    return merged


def ld_adapt_settings_to_image(original_image, stain_complete_df, voxel_size_zyx,
                               nuclear_conditions, settings, verbose=True, adapt_scale=True):
    """`adapt_settings_to_image` for the LIVE/DEAD notebook: the typical
    nucleus is measured in the merged nuclear markers (`_ld_nuclear_view`)
    and ``settings`` (``nuclei_diameter``, ``scale_factor`` and the um
    settings) adapted to it. Returns ``(adapted settings, changes)``; nothing
    changes without nuclear markers or when too few nuclei are measured
    (e.g. a DEAD marker alone, with few dead cells). ``adapt_scale=False``
    (a ``scale_factor`` set in ``custom_settings``) keeps the
    ``scale_factor`` as given."""
    if not nuclear_conditions:
        if verbose:
            print("No nuclear marker: settings not adapted to the image.")
        return dict(settings), []
    if verbose:
        print("Typical nucleus measured in the nuclear marker(s): " + ", ".join(nuclear_conditions))
    view = _ld_nuclear_view(original_image, stain_complete_df, nuclear_conditions)
    measurement = measure_nuclei_in_image(view, voxel_size_zyx, contrast=None, verbose=verbose)
    del view
    return adapt_nuclei_settings(measurement, settings, voxel_size_zyx, adapt_scale=adapt_scale,
                                 verbose=verbose)


def ld_remove_nuclear_haze(denoised, stain_complete_df, nuclear_conditions, radius_um,
                           voxel_size_zyx, percentile=25.0):
    """Slice-by-slice haze removal (`correct_z_background`) of the nuclear
    markers' 'Denoised image' channels. Returns the (new) array."""
    if not radius_um or radius_um <= 0 or denoised.shape[0] <= 1 or not nuclear_conditions:
        return denoised
    out = correct_z_background(
        denoised, radius_um, tuple(voxel_size_zyx), percentile=percentile,
        channels=[stain_complete_df.index.get_loc(c) for c in nuclear_conditions])
    print(f"Haze of the nuclear marker(s) {', '.join(nuclear_conditions)} removed slice by "
          f"slice (radius {radius_um:.3g} um).")
    return out


def ld_correct_depth(original, stain_complete_df, nuclear_conditions, conditions, model="exp",
                     strength=1.0):
    """Depth (Z attenuation) correction, `correct_z_attenuation`, of the LIVE/DEAD
    channels: ``conditions`` = 'all' or a list of conditions, where 'NUCLEI'
    stands for every nuclear marker. Returns ``(image, profiles, names)``."""
    if conditions == "all":
        conds = list(stain_complete_df.index)
    else:
        conds = []
        for c in conditions:
            for cond in (nuclear_conditions if str(c).upper() == "NUCLEI" else [c]):
                if cond in stain_complete_df.index and cond not in conds:
                    conds.append(cond)
    if not conds or original.shape[0] <= 2:
        return original, None, conds
    image, profiles = correct_z_attenuation(
        original, model=model, strength=strength,
        channels=[stain_complete_df.index.get_loc(c) for c in conds], inplace=True)
    return image, profiles, conds


def ld_binary_threshold(im_final_stack, stain_complete_df, nuclei_diameter, r_zxyz,
                        conditions=None, nuclear_conditions=None,
                        min_brightness=0.3, edge_brightness=0.15, min_snr=5.0):
    """LIVE/DEAD v1.2.2, Cell 15: keep only the clearly bright signal of the
    LIVE/DEAD stains in their threshold masks.

    The threshold of Cell 15 (validated on nuclei) follows the intensity
    gradients, so it also keeps dim objects: out-of-focus cells (in a 2D image
    often light from another depth) and faint halos. LIVE/DEAD stains are
    binary -- a cell is bright or it is not -- so for each condition in
    ``conditions`` (default: every marker of the call, LIVE / LIVE_<n> /
    DEAD / DEAD_<n>, `ld_role`; e.g. a Hoechst counterstain added as LIVE_2),
    on its 'Denoised image' channel (linear intensity, lightly smoothed):

    - background = median and noise (1.4826 x MAD) outside the mask;
    - the *typical bright cell* = the 90th percentile of the peak intensity
      of the mask's objects;
    - an object is kept only if its peak is at least ``min_brightness`` x
      (typical bright cell - background) above the background, **and**
      ``min_snr`` x the noise above it (so a channel without real cells, e.g.
      no dead cells, ends up empty instead of keeping its brightest noise);
    - inside the kept objects, voxels dimmer than ``edge_brightness`` x
      (typical bright cell - background) above the background are background.

    The pieces left smaller than the small-island limit of Cell 15 are
    removed. Every factor at 0 = that test off. The masks of
    ``im_final_stack['Threshold image']`` are changed in place: they are what
    the segmentation (Cells 17-18) and the positivity of the LIVE/DEAD call
    (Cell 19) use. Returns ``im_final_stack``.
    """
    if not (min_brightness or edge_brightness or min_snr):
        print("LIVE/DEAD bright-signal filter off.")
        return im_final_stack
    if conditions is None:
        conditions = [c for c in stain_complete_df.index if ld_role(c)]
    thr = im_final_stack['Threshold image']
    den = im_final_stack['Denoised image']
    r_zX, r_zY, r_zZ = r_zxyz
    is_2d = thr.shape[0] == 1
    if is_2d:
        nucleus_vox = np.pi * (nuclei_diameter / 2.0) ** 2 / (r_zX * r_zY)
    else:
        nucleus_vox = _sphere_volume_um3(nuclei_diameter) / (r_zX * r_zY * r_zZ)
    nuclear = {str(c).upper() for c in (nuclear_conditions or [])}
    rows = []
    for cond in conditions:
        if cond not in stain_complete_df.index:
            continue
        c = stain_complete_df.index.get_loc(cond)
        mask = thr[..., c] > 0
        if not mask.any():
            rows.append({"Condition": cond, "Objects kept": 0, "Objects removed": 0})
            continue
        sm = ndi.gaussian_filter(np.asarray(den[..., c], dtype=np.float32),
                                 sigma=(0, 1, 1) if is_2d else 1.0)
        outside = sm[~mask]
        if outside.size == 0:
            outside = sm.ravel()
        if outside.size > 2_000_000:
            outside = outside[::max(1, outside.size // 2_000_000)]
        bg = float(np.median(outside))
        noise = max(0.5, 1.4826 * float(np.median(np.abs(outside - bg))))
        del outside
        labels, n = ndi.label(mask)
        peaks = np.asarray(ndi.maximum(sm, labels, index=np.arange(1, n + 1)), dtype=float)
        ref = float(np.percentile(peaks, 90)) if n >= 3 else float(peaks.max())
        span = max(ref - bg, 1e-6)
        cut_obj = max(bg + float(min_brightness or 0) * span, bg + float(min_snr or 0) * noise)
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = peaks >= cut_obj
        new = keep[labels]
        del labels
        cut_vox = bg + float(edge_brightness or 0) * span
        if edge_brightness:
            new &= sm >= cut_vox
        del sm
        min_size = np.ceil((0.4 if str(cond).upper() in nuclear else 0.8) * nucleus_vox)
        new = remove_small_islands(new, min_size)
        n_vox = int(np.count_nonzero(mask))
        thr[..., c] = new.astype(thr.dtype)
        rows.append({
            "Condition": cond, "Background": round(bg, 1), "Noise": round(noise, 2),
            "Typical bright cell": round(ref, 1), "Object cutoff": round(cut_obj, 1),
            "Voxel cutoff": round(cut_vox, 1) if edge_brightness else None,
            "Objects kept": int(np.count_nonzero(keep[1:])),
            "Objects removed": int(n - np.count_nonzero(keep[1:])),
            "Mask voxels removed [%]": round(100.0 * (1 - np.count_nonzero(new) / max(n_vox, 1)), 1),
        })
        del mask, new
    print(f"LIVE/DEAD bright-signal filter (objects >= {min_brightness:g} x the typical bright "
          f"cell and >= {min_snr:g} x noise above background; voxels >= {edge_brightness:g} x):")
    if rows:
        _display_if_notebook(pd.DataFrame(rows).set_index("Condition"))
        for r in rows:
            if r.get("Objects kept") == 0 and r.get("Objects removed"):
                print(f"  {r['Condition']}: no object stands out clearly from the background "
                      "-- the channel has no positive cells now. Lower ld_min_brightness / "
                      "ld_min_snr if that is wrong.")
    return im_final_stack


def segment_ld_nuclei(
    im_final_stack,
    stain_df,
    localization,
    r_zxyz,
    nuclei_diameter,
    nuclei_split_config=None,
    trig_stardist=False,
    trig_cellpose=False,
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    size_reference="blend",
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.2,
    progress=None,
    **nuclei_options,
):
    """LD Cell 17: segment the nuclei from the markers set as **nuclear**.

    v1.2.2: ``nuclei_options`` are the v1.6.4 NUCLEI options of
    `segment_nuclei` (``z_elongation``, ``nuclei_flare_trim_factor``,
    ``nuclei_saddle_dip``, ``nuclei_tall_*``, ``nuclei_rescue_*``,
    ``nuclei_edge_*``, ``nuclei_expand_um``), applied to the merged nuclear
    markers on 3D images (off when not given, as in v1.2.1). The split gamma
    is applied by `segment_nuclei` itself, so these steps see the linear
    intensity.

    Their threshold masks (Cell 15) are merged into one nuclei mask, which is
    segmented exactly like the NUCLEI channel of the nuclei notebook
    (`segment_nuclei`: watershed, Cellpose or StarDist, then the size
    refinement). Without nuclear markers no nuclei are segmented here: the
    nucleus of each cell is then estimated in Cell 18.

    ``split_gamma`` (> 1) is applied to the intensity used to split touching
    nuclei (watershed and merge test; not the deep-learning input), so the
    dark seams between them get darker faster than the nuclei; it grows by
    ``split_gamma_step`` at each re-split level of the size refinement.

    On a 2D image, the pieces of each split smaller than
    ``min_piece_fraction`` x the expected nucleus size are merged into the
    closest big piece of the same split.

    Returns
    -------
    im_segmentation_stack : dict with 'Nuclei' (Z, Y, X) int32.
    """
    nuclear, _ = _ld_channel_groups(stain_df, localization)
    shape = im_final_stack['Threshold image'].shape[:3]
    if not nuclear:
        print("No nuclear marker: no nuclei segmented here. Each cell's nucleus is "
              "estimated from its cell body in Cell 18.")
        return {'Nuclei': np.zeros(shape, dtype=np.int32)}

    print(f"Nuclei segmented from the nuclear marker(s): {_ld_marker_names(stain_df, nuclear)}")
    if shape[0] == 1:
        r_zX, r_zY, _ = r_zxyz
        merged = _ld_merged_substack(im_final_stack, nuclear)
        method = 'cellpose' if trig_cellpose else ('stardist' if trig_stardist else 'watershed')
        labels = _ld_segment_objects_2d(
            merged['Threshold image'][0, ..., 0] > 0,
            merged['Filtered image'][0, ..., 0],
            pixel_yx=(r_zY, r_zX),
            diameter_um=nuclei_diameter,
            what="nuclei",
            method=method,
            cellpose_model='nuclei',
            size_reference=size_reference,
            iterative_split=iterative_split,
            oversize_factor=oversize_factor,
            max_split_iterations=max_split_iterations,
            merge_undersized=merge_undersized,
            undersize_factor=undersize_factor,
            split_gamma=split_gamma,
            split_gamma_step=split_gamma_step,
            min_piece_fraction=min_piece_fraction,
            progress=progress,
        )
        return {'Nuclei': labels[np.newaxis]}

    unknown = set(nuclei_options) - set(_BATCH_SEGMENT_NUCLEI_KEYS)
    if unknown:
        raise TypeError(f"segment_ld_nuclei: unknown option(s) {sorted(unknown)}")
    merged = _ld_merged_substack(im_final_stack, nuclear)
    sub_df = _ld_single_channel_df('+'.join(str(stain_df['Marker'].iloc[c]) for c in nuclear))
    # The split gamma is applied inside segment_nuclei (NUCLEI path): only to
    # the intensity used for the split, so the v1.6.4 steps after the
    # watershed (saddle / tall split, rescue, edges) get the linear signal.
    seg = segment_nuclei(
        merged,
        stain_df=sub_df,
        stain_complete_df=sub_df,
        r_zxyz=r_zxyz,
        nuclei_diameter=nuclei_diameter,
        nuclei_split_config=nuclei_split_config,
        trig_stardist=trig_stardist,
        trig_cellpose=trig_cellpose,
        iterative_split=iterative_split,
        oversize_factor=oversize_factor,
        max_split_iterations=max_split_iterations,
        merge_undersized=merge_undersized,
        undersize_factor=undersize_factor,
        size_reference=size_reference,
        progress=progress,
        split_gamma=split_gamma,
        split_gamma_step=split_gamma_step,
        min_piece_fraction=min_piece_fraction,
        **nuclei_options,
    )
    return {'Nuclei': np.asarray(seg['Nuclei'], dtype=np.int32)}


def _ld_gamma_substack(merged, what, split_gamma, split_gamma_step, deep_learning=False):
    """Apply the first-pass ``split_gamma`` to the merged intensities used to
    split touching objects in 3D (not when a deep-learning model segments
    them: it gets the original image). Returns the substack and the extra
    gamma per re-split level, relative to the gamma already applied, for
    `segment_nuclei`'s ``refine_gamma_levels`` (None when no gamma is set)."""
    base, levels = _ld_split_gammas(split_gamma, split_gamma_step)
    if abs(base - 1.0) < 1e-6 and not split_gamma_step:
        return merged, None
    applied = 1.0 if deep_learning else base
    _ld_print_split_gamma(what, split_gamma, split_gamma_step, applied=not deep_learning)
    if not deep_learning:
        merged = dict(merged)
        for key in ('Filtered image', 'Denoised image'):
            merged[key] = apply_split_gamma(merged[key], base)
    return merged, [g / applied for g in levels]


def _ld_union_slices(slices):
    slices = [s for s in slices if s is not None]
    return tuple(
        slice(min(s[d].start for s in slices), max(s[d].stop for s in slices))
        for d in range(len(slices[0]))
    )


def _ld_combine_nuclei_and_bodies(nuclei, bodies, spacing, nuclei_diameter,
                                  min_nucleus_overlap=0.5, progress=None,
                                  body_intensity=None):
    """Combine the nuclei (nuclear markers) and the cell bodies (cytoplasmic
    markers) into one label per cell:

    - a nucleus lying for at least ``min_nucleus_overlap`` of its volume in a
      cell body belongs to that cell;
    - a body holding two or more nuclei is split between them by a watershed
      seeded by the nuclei. With ``body_intensity`` (the cytoplasmic signal,
      gamma-adjusted by the caller) the border follows the cytoplasm, as in
      the nuclei notebook's cytoplasm split (`_gradient_watershed_elevation`:
      distance + dim signal + steep intensity change); without it, the
      distance map alone;
    - a nucleus outside every body is a cell without stained cytoplasm: the
      cell is the nucleus;
    - a body with no nucleus is a cell whose nucleus is not stained: its
      nucleus is *estimated* as a sphere (a disk in 2D) of ``nuclei_diameter``
      around the innermost point of the body. It gives the cell a position and the
      compartment where the nuclear markers are measured.

    Returns
    -------
    cells : (Z, Y, X) int32 -- whole cells, nucleus included, labels 1..K
    nuclei_out : (Z, Y, X) int32 -- stained or estimated nucleus, same labels
    stained : (Z, Y, X) int32 -- stained nuclei only, same labels
    info : dict of counts
    """
    shape = nuclei.shape
    cells = np.zeros(shape, dtype=np.int32)
    nuc_out = np.zeros(shape, dtype=np.int32)
    n_nuc = int(nuclei.max()) if nuclei.size else 0
    n_body = int(bodies.max()) if bodies.size else 0

    # Each nucleus -> the body it overlaps most (if enough of it lies inside).
    owner = np.zeros(n_nuc + 1, dtype=np.int64)
    if n_nuc and n_body:
        both = (nuclei > 0) & (bodies > 0)
        pairs = nuclei[both].astype(np.int64) * (n_body + 1) + bodies[both]
        del both
        pair_ids, pair_counts = np.unique(pairs, return_counts=True)
        del pairs
        nuc_size = np.bincount(nuclei.ravel(), minlength=n_nuc + 1)
        best = {}
        for pid, count in zip(pair_ids.tolist(), pair_counts.tolist()):
            n, b = divmod(pid, n_body + 1)
            if n not in best or count > best[n][1]:
                best[n] = (b, count)
        for n, (b, count) in best.items():
            if count >= min_nucleus_overlap * nuc_size[n]:
                owner[n] = b

    nuclei_of_body = defaultdict(list)
    for n in range(1, n_nuc + 1):
        if owner[n]:
            nuclei_of_body[int(owner[n])].append(n)
    nuc_slices = ndi.find_objects(nuclei) if n_nuc else []
    body_slices = ndi.find_objects(bodies) if n_body else []

    next_id = 1
    n_split_bodies = 0
    for b in _progress_iter(range(1, n_body + 1), progress,
                            desc='Step 18B - Combine Nuclei And Cell Bodies', leave=False):
        if body_slices[b - 1] is None:
            continue
        owned = nuclei_of_body.get(b, [])
        sl = _ld_union_slices([body_slices[b - 1]] + [nuc_slices[n - 1] for n in owned])
        body = bodies[sl] == b
        nuc_sub = nuclei[sl]
        cell_view = cells[sl]
        nuc_view = nuc_out[sl]
        if len(owned) <= 1:
            region = body
            if owned:
                nuc_mask = nuc_sub == owned[0]
                region = body | nuc_mask
                nuc_view[nuc_mask] = next_id
            cell_view[region] = next_id
            next_id += 1
            continue
        # Several nuclei in one body: split it between them.
        markers = np.zeros(body.shape, dtype=np.int32)
        for k, n in enumerate(owned, start=1):
            markers[nuc_sub == n] = k
        region = body | (markers > 0)
        if body_intensity is not None:
            # Padded by one voxel so the crop edge isn't taken as inside the body.
            inner = tuple(slice(1, -1) for _ in range(region.ndim))
            region_p = np.pad(region, 1)
            elevation = _gradient_watershed_elevation(
                [np.pad(np.asarray(body_intensity[sl], dtype=np.float32), 1, mode='edge')],
                region_p, spacing=spacing,
            )
            parts = watershed(elevation, np.pad(markers, 1), mask=region_p)[inner]
        else:
            parts = watershed(-_ld_edt(region, spacing), markers, mask=region)
        for k in range(1, len(owned) + 1):
            cell_view[parts == k] = next_id
            nuc_view[markers == k] = next_id
            next_id += 1
        n_split_bodies += 1

    n_nucleus_only = 0
    for n in range(1, n_nuc + 1):
        if owner[n] or nuc_slices[n - 1] is None:
            continue
        sl = nuc_slices[n - 1]
        nuc_mask = nuclei[sl] == n
        cells[sl][nuc_mask] = next_id
        nuc_out[sl][nuc_mask] = next_id
        next_id += 1
        n_nucleus_only += 1

    # A nucleus always belongs to its own cell (a neighbouring body may have
    # been written over it), then renumber the cells 1..K.
    has_nuc = nuc_out > 0
    cells[has_nuc] = nuc_out[has_nuc]
    del has_nuc
    present = np.unique(cells)
    present = present[present > 0]
    lut = np.zeros(next_id, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    cells = lut[cells]
    nuc_out = lut[nuc_out]
    stained = nuc_out.copy()
    n_cells = int(present.size)

    # Cells without a stained nucleus: nucleus-sized sphere at the innermost
    # point of the cell.
    with_nucleus = np.zeros(n_cells + 1, dtype=bool)
    with_nucleus[np.unique(nuc_out)] = True
    missing = np.flatnonzero(~with_nucleus)
    radius = nuclei_diameter / 2.0
    cell_slices = ndi.find_objects(cells)
    for j in _progress_iter(missing.tolist(), progress,
                            desc='Step 18C - Estimate Unstained Nuclei', leave=False):
        sl = cell_slices[j - 1]
        if sl is None:
            continue
        region = cells[sl] == j
        distance = _ld_edt(region, spacing)
        center = np.unravel_index(int(np.argmax(distance)), distance.shape)
        grids = np.ogrid[tuple(slice(0, n) for n in region.shape)]
        d2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, center, spacing))
        nuc_out[sl][region & (d2 <= radius ** 2)] = j

    info = {
        'nuclei': n_nuc,
        'bodies': n_body,
        'cells': n_cells,
        'estimated nuclei': int(missing.size),
        'nucleus only': n_nucleus_only,
        'bodies split': n_split_bodies,
    }
    return cells, nuc_out, stained, info


def segment_ld_cells(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    localization,
    r_zxyz,
    nuclei_diameter,
    cell_diameter,
    trig_cellpose_cyto=False,
    nuclei_split_config=None,
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    size_reference="blend",
    min_nucleus_overlap=0.5,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.2,
    progress=None,
):
    """LD Cell 18: segment the cell bodies from the markers set as
    **cytoplasmic** and combine them with the nuclei of Cell 17 into one
    label per cell.

    The cytoplasmic threshold masks (Cell 15) are merged and segmented like
    the nuclei, at the scale of ``cell_diameter`` (watershed and size
    refinement; Cellpose 'cyto3' with ``trig_cellpose_cyto``), with the same
    ``split_gamma`` / ``split_gamma_step`` as the nuclei (Cell 17). Nuclei and
    bodies are then combined (`_ld_combine_nuclei_and_bodies`): a body with
    several nuclei is split between them along the cytoplasm border (the
    cytoplasmic intensity with ``split_gamma``), a nucleus outside every body is a
    cell on its own, and a body without a stained nucleus gets an estimated
    one (a nucleus-sized sphere at its centre).

    Adds to ``im_segmentation_stack``:

    - 'Cytoplasm': the whole cells (nucleus included), labels 1..K -- the
      same layout as the cytoplasm of the nuclei notebook;
    - 'Nuclei': each cell's nucleus, stained or estimated, same labels;
    - 'Nuclei (stained)': the stained nuclei only;
    - 'PCM': empty (no pericellular matrix in the LD analysis).

    A placeholder CYTOPLASM row is added to ``stain_complete_df`` (as
    `segment_cytoplasm` does when there is no CYTOPLASM channel).
    """
    r_zX, r_zY, r_zZ = r_zxyz
    _, cytoplasmic = _ld_channel_groups(stain_df, localization)
    if 'Nuclei' not in im_segmentation_stack:
        raise RuntimeError("Run Cell 17 first so im_segmentation_stack contains 'Nuclei'.")
    nuclei = np.asarray(im_segmentation_stack['Nuclei'], dtype=np.int32)
    is_2d = nuclei.shape[0] == 1

    if cytoplasmic:
        print(f"Cell bodies segmented from the cytoplasmic marker(s): "
              f"{_ld_marker_names(stain_df, cytoplasmic)}")
        merged = _ld_merged_substack(im_final_stack, cytoplasmic)
        # Cytoplasmic signal (with the first-pass split gamma) used to split a
        # body between several nuclei along the cytoplasm border.
        body_intensity = apply_split_gamma(
            merged['Filtered image'][..., 0], _ld_split_gammas(split_gamma, split_gamma_step)[0]
        )
        if is_2d:
            bodies = _ld_segment_objects_2d(
                merged['Threshold image'][0, ..., 0] > 0,
                merged['Filtered image'][0, ..., 0],
                pixel_yx=(r_zY, r_zX),
                diameter_um=cell_diameter,
                what="cell bodies",
                method='cellpose' if trig_cellpose_cyto else 'watershed',
                cellpose_model='cyto3',
                size_reference=size_reference,
                iterative_split=iterative_split,
                oversize_factor=oversize_factor,
                max_split_iterations=max_split_iterations,
                merge_undersized=merge_undersized,
                undersize_factor=undersize_factor,
                split_gamma=split_gamma,
                split_gamma_step=split_gamma_step,
                min_piece_fraction=min_piece_fraction,
                progress=progress,
            )[np.newaxis]
        elif trig_cellpose_cyto:
            intensity = merged['Filtered image'][..., 0]
            bodies = segment_nuclei_cellpose(
                intensity,
                nuclei_diameter=cell_diameter,
                voxel_size=(r_zZ, r_zY, r_zX),
                model_type='cyto3',
            )
            if iterative_split or merge_undersized:
                # Cellpose got the original image; the refinement splits on
                # the gamma-adjusted one.
                base_gamma, level_gammas = _ld_split_gammas(split_gamma, split_gamma_step)
                if abs(base_gamma - 1.0) > 1e-6 or split_gamma_step:
                    _ld_print_split_gamma("cell bodies", split_gamma, split_gamma_step)
                    refine_gammas = [g / base_gamma for g in level_gammas]
                else:
                    refine_gammas = None
                bodies, _ = refine_nuclei_by_size(
                    bodies,
                    intensity_img=apply_split_gamma(intensity, base_gamma),
                    intensity_img_raw=apply_split_gamma(merged['Denoised image'][..., 0], base_gamma),
                    intensity_gamma_levels=refine_gammas,
                    r_zxyz=r_zxyz,
                    nuclei_diameter=cell_diameter,
                    base_split_config=nuclei_split_config,
                    size_reference=size_reference,
                    merge_undersized=merge_undersized,
                    undersize_factor=undersize_factor,
                    iterative_split=iterative_split,
                    oversize_factor=oversize_factor,
                    max_split_iterations=max_split_iterations,
                    progress=progress,
                )
        else:
            merged, refine_gammas = _ld_gamma_substack(
                merged, "cell bodies", split_gamma, split_gamma_step,
            )
            sub_df = _ld_single_channel_df(
                '+'.join(str(stain_df['Marker'].iloc[c]) for c in cytoplasmic))
            bodies = segment_nuclei(
                merged,
                stain_df=sub_df,
                stain_complete_df=sub_df,
                r_zxyz=r_zxyz,
                nuclei_diameter=cell_diameter,
                nuclei_split_config=nuclei_split_config,
                iterative_split=iterative_split,
                oversize_factor=oversize_factor,
                max_split_iterations=max_split_iterations,
                merge_undersized=merge_undersized,
                undersize_factor=undersize_factor,
                size_reference=size_reference,
                progress=progress,
                refine_gamma_levels=refine_gammas,
                min_piece_fraction=min_piece_fraction,
            )['Nuclei']
        del merged
        bodies = np.asarray(bodies, dtype=np.int32)
    else:
        print("No cytoplasmic marker: each cell is its stained nucleus.")
        bodies = np.zeros_like(nuclei)
        body_intensity = None

    if is_2d:
        # Combine in the image plane (a 3D distance map would see the empty
        # planes above and below the single slice).
        cells, nuclei_out, stained, info = _ld_combine_nuclei_and_bodies(
            nuclei[0], bodies[0],
            spacing=(r_zY, r_zX),
            nuclei_diameter=nuclei_diameter,
            min_nucleus_overlap=min_nucleus_overlap,
            progress=progress,
            body_intensity=None if body_intensity is None else body_intensity[0],
        )
        cells, nuclei_out, stained = (a[np.newaxis] for a in (cells, nuclei_out, stained))
    else:
        cells, nuclei_out, stained, info = _ld_combine_nuclei_and_bodies(
            nuclei, bodies,
            spacing=(r_zZ, r_zY, r_zX),
            nuclei_diameter=nuclei_diameter,
            min_nucleus_overlap=min_nucleus_overlap,
            progress=progress,
            body_intensity=body_intensity,
        )
    del body_intensity
    n_cells = info['cells']
    n_both = n_cells - info['estimated nuclei'] - info['nucleus only']
    print(f"{info['nuclei']} nuclei + {info['bodies']} cell bodies -> {n_cells} cells:")
    print(f"  {n_both} with a stained nucleus inside a cell body "
          f"({info['bodies split']} bodies split between 2+ nuclei)")
    print(f"  {info['nucleus only']} stained nucleus only (no cytoplasmic signal)")
    print(f"  {info['estimated nuclei']} cell body only (nucleus estimated at the cell centre)")

    if 'CYTOPLASM' not in stain_complete_df.index:
        stain_complete_df = stain_complete_df.copy()
        stain_complete_df.loc['CYTOPLASM'] = ['', '', '', 0, 255, 1.0]

    im_segmentation_stack = dict(im_segmentation_stack)
    im_segmentation_stack['Nuclei'] = nuclei_out
    im_segmentation_stack['Nuclei (stained)'] = stained
    im_segmentation_stack['Cytoplasm'] = cells
    # No pericellular matrix in the LD analysis: an empty PCM layer keeps the
    # shared quantification/export helpers (which read 'PCM') working.
    im_segmentation_stack['PCM'] = np.zeros_like(cells)
    return im_segmentation_stack, stain_complete_df


def _ld_z_chunk(shape, max_voxels=20_000_000):
    return max(1, int(max_voxels // max(1, int(shape[1]) * int(shape[2]))))


def _ld_label_sums(labels, values, n_labels):
    """Per-label sum of *values* over labels 0..n_labels (voxel count if
    *values* is None), computed in Z slabs to bound memory."""
    out = np.zeros(n_labels + 1, dtype=np.float64)
    chunk = _ld_z_chunk(labels.shape)
    for z0 in range(0, labels.shape[0], chunk):
        lab = labels[z0:z0 + chunk].ravel()
        weights = None if values is None else values[z0:z0 + chunk].ravel().astype(np.float64)
        out += np.bincount(lab, weights=weights, minlength=n_labels + 1)[:n_labels + 1]
    return out


def _ld_label_centroids(labels, n_labels):
    """Mean (z, y, x) voxel position of labels 1..n_labels, shape (n, 3)."""
    counts = np.zeros(n_labels + 1, dtype=np.float64)
    sums = np.zeros((3, n_labels + 1), dtype=np.float64)
    nz, ny, nx = labels.shape
    chunk = _ld_z_chunk(labels.shape)
    for z0 in range(0, nz, chunk):
        lab = labels[z0:z0 + chunk]
        flat = lab.ravel()
        n_planes = lab.shape[0]
        coords = (
            np.repeat(np.arange(z0, z0 + n_planes, dtype=np.float64), ny * nx),
            np.tile(np.repeat(np.arange(ny, dtype=np.float64), nx), n_planes),
            np.tile(np.arange(nx, dtype=np.float64), n_planes * ny),
        )
        counts += np.bincount(flat, minlength=n_labels + 1)[:n_labels + 1]
        for d in range(3):
            sums[d] += np.bincount(flat, weights=coords[d], minlength=n_labels + 1)[:n_labels + 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        return (sums / counts).T[1:]


def _ld_size_names(is_2d):
    """Column names and units of the per-cell sizes: areas for 2D images,
    volumes for 3D ones."""
    if is_2d:
        return {'cell': 'Cell area [um2]', 'nucleus': 'Nucleus area [um2]',
                'word': 'area', 'unit': 'um2', 'pretty': 'um²'}
    return {'cell': 'Cell volume [um3]', 'nucleus': 'Nucleus volume [um3]',
            'word': 'volume', 'unit': 'um3', 'pretty': 'um³'}


def _ld_intensity_column(condition, compartment):
    return f"{condition} mean int. ({'nucleus' if compartment == 'nuclear' else 'cell'})"


# v1.2.2: with both a LIVE and a DEAD channel, a cell positive for neither
# marker gets no call: UNCLASSIFIED, reported but left out of the viability.
LD_UNCLASSIFIED = "UNCLASSIFIED"
LD_STATUSES = ("LIVE", "DEAD", LD_UNCLASSIFIED)


def _ld_counts(ld_cells_df):
    """Cells per status and the viability, LIVE / (LIVE + DEAD) in % (the
    UNCLASSIFIED cells are left out; NaN when no cell is LIVE or DEAD)."""
    counts = ld_cells_df['Status'].value_counts() if len(ld_cells_df) else {}
    out = {s: int(counts.get(s, 0)) for s in LD_STATUSES}
    out['total'] = int(len(ld_cells_df))
    out['classified'] = out['LIVE'] + out['DEAD']
    out['viability'] = (100.0 * out['LIVE'] / out['classified']) if out['classified'] else np.nan
    return out


def _ld_pct(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.1f}%"


def _ld_rule_text(rule, has_live, has_dead):
    neither = (f"; positive for neither -> {LD_UNCLASSIFIED} (not in the viability)"
               if has_live and has_dead else "")
    if rule == "coverage":
        return ("positive for one marker only -> that one; for both -> "
                "the marker covering more of its compartment" + neither)
    if has_dead and has_live:
        return ("positive for a DEAD marker -> DEAD, for a LIVE marker only -> LIVE"
                + neither)
    if has_dead:
        return "DEAD-positive -> DEAD, otherwise LIVE"
    return "LIVE-positive -> LIVE, otherwise DEAD"


def classify_live_dead(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    localization,
    r_zxyz,
    min_positive_fraction=0.2,
    ld_rule="dead_priority",
    progress=None,
):
    """LD Cell 19: marker positivity of every cell and the binary LIVE/DEAD call.

    Each marker is measured in its own compartment: a nuclear marker in the
    cell's nucleus ('Nuclei', stained or estimated), a cytoplasmic marker in
    the whole cell ('Cytoplasm', nucleus included). A cell is positive for a
    marker when the marker's threshold mask (Cell 15) covers at least
    ``min_positive_fraction`` of that compartment (0 = any voxel, as in the
    nuclei notebook).

    The call (``ld_rule``) uses the LIVE and DEAD conditions only:

    - ``"dead_priority"`` (default): DEAD-positive -> DEAD, otherwise LIVE
      (a cell whose membrane lets the dead stain in is dead, even with some
      calcein left). Without a DEAD channel: LIVE-positive -> LIVE,
      otherwise DEAD.
    - ``"coverage"``: positive for one of the two only -> that one; positive
      for both -> the marker covering the larger fraction of its
      compartment. Needs both channels (otherwise "dead_priority" is used).

    v1.2.2: with both a LIVE and a DEAD channel, a cell positive for neither
    marker (e.g. a nucleus of the counterstain only, or a cell whose signal
    was too dim for the bright-signal filter) is ``UNCLASSIFIED``: reported,
    but left out of the viability, LIVE / (LIVE + DEAD).

    Adds to ``im_segmentation_stack`` one label layer per condition -- the
    marker's voxels in the compartment of its positive cells, also stored as
    '<condition>_cyto' for the shared quantification -- and 'Status' (each
    cell's voxels: 1 = LIVE, 2 = DEAD, 3 = UNCLASSIFIED).

    Returns
    -------
    im_segmentation_stack : dict
    ld_cells_df : DataFrame
        One row per cell (index 'Cell label' 1..K): status, position and
        size of the cell (volume, or area for a 2D image), nucleus
        (stained/estimated) and, per condition,
        coverage, positivity and mean intensity. The settings are stored in
        ``ld_cells_df.attrs``.
    """
    if ld_rule not in LD_RULES:
        raise ValueError(f"ld_rule must be one of {LD_RULES}, got '{ld_rule}'")
    loc = _ld_localization_map(localization)
    r_zX, r_zY, r_zZ = r_zxyz
    thr = im_final_stack['Threshold image']
    filt = im_final_stack['Filtered image']
    cells = im_segmentation_stack['Cytoplasm']
    nuclei = im_segmentation_stack['Nuclei']
    stained = im_segmentation_stack.get('Nuclei (stained)', nuclei)
    n_cells = int(cells.max())
    if n_cells == 0:
        raise RuntimeError("No cells were segmented: check the thresholds (Cell 15) "
                           "and the segmentation (Cells 17-18).")

    cell_vox = _ld_label_sums(cells, None, n_cells)
    nuc_vox = _ld_label_sums(nuclei, None, n_cells)
    stained_vox = _ld_label_sums(stained, None, n_cells)[1:]
    centroids = _ld_label_centroids(cells, n_cells)
    # 2D image: sizes are areas (pixel area), not volumes.
    is_2d = cells.shape[0] == 1
    size = _ld_size_names(is_2d)
    unit_um = float(r_zX) * float(r_zY) * (1.0 if is_2d else float(r_zZ))

    df = pd.DataFrame(index=pd.RangeIndex(1, n_cells + 1, name='Cell label'))
    df['X [um]'] = centroids[:, 2] * r_zX
    df['Y [um]'] = centroids[:, 1] * r_zY
    if not is_2d:
        df['Z [um]'] = centroids[:, 0] * r_zZ
    df[size['cell']] = cell_vox[1:] * unit_um
    df['Nucleus'] = np.where(stained_vox > 0, 'stained', 'estimated')
    df[size['nucleus']] = np.where(stained_vox > 0, stained_vox * unit_um, np.nan)

    seg = dict(im_segmentation_stack)
    coverage, positive = {}, {}
    for c, cond in _progress_iter(list(enumerate(stain_df.index)), progress,
                                  desc='Step 20 - Marker Positivity Per Cell'):
        compartment = loc.get(str(cond).upper())
        if compartment not in LD_LOCALIZATIONS:
            raise ValueError(f"No localization ('nuclear' or 'cytoplasmic') for condition '{cond}'.")
        lab = nuclei if compartment == 'nuclear' else cells
        denom = nuc_vox if compartment == 'nuclear' else cell_vox
        mask = thr[..., c] > 0
        intensity = filt[..., c]
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = np.where(denom > 0, _ld_label_sums(lab, mask, n_cells) / denom, 0.0)[1:]
            mean_comp = np.where(denom > 0, _ld_label_sums(lab, intensity, n_cells) / denom, np.nan)[1:]
            mean_cell = np.where(cell_vox > 0, _ld_label_sums(cells, intensity, n_cells) / cell_vox, np.nan)[1:]
        pos = (cov > 0) & (cov >= min_positive_fraction)

        df[f'{cond} coverage'] = cov
        df[f'{cond} positive'] = pos
        df[_ld_intensity_column(cond, compartment)] = mean_comp
        if compartment == 'nuclear':
            df[f'{cond} mean int. (cell)'] = mean_cell
        coverage[cond] = cov
        positive[cond] = pos

        pos_lut = np.zeros(n_cells + 1, dtype=bool)
        pos_lut[1:] = pos
        layer = np.where(mask & pos_lut[lab], lab, 0).astype(np.int32)
        seg[cond] = layer
        seg[cond + '_cyto'] = layer
        del mask, layer

    # v1.2.2: the LIVE group (LIVE, LIVE_2, ... e.g. Hoechst) and the DEAD
    # group (DEAD, DEAD_2, ...): a cell is positive for a group when it is
    # positive for any of its markers; the group coverage is the largest one.
    groups = {role: ld_call_conditions(list(positive), role) for role in ('LIVE', 'DEAD')}
    group_pos, group_cov = {}, {}
    for role, conds in groups.items():
        if conds:
            group_pos[role] = np.logical_or.reduce([positive[c] for c in conds])
            group_cov[role] = np.max(np.vstack([coverage[c] for c in conds]), axis=0)
            df[f'{role} call positive'] = group_pos[role]
            df[f'{role} call coverage'] = group_cov[role]
    has_live, has_dead = 'LIVE' in group_pos, 'DEAD' in group_pos
    if not (has_live or has_dead):
        raise ValueError("The stain table needs a LIVE and/or a DEAD condition.")
    rule = ld_rule
    if rule == 'coverage' and not (has_live and has_dead):
        print("ld_rule='coverage' needs both a LIVE and a DEAD channel: "
              "using 'dead_priority' instead.")
        rule = 'dead_priority'
    if rule == 'coverage':
        live_pos, dead_pos = group_pos['LIVE'], group_pos['DEAD']
        is_dead = np.where(live_pos != dead_pos, dead_pos, group_cov['DEAD'] > group_cov['LIVE'])
    elif has_dead:
        is_dead = group_pos['DEAD']
    else:
        is_dead = ~group_pos['LIVE']

    status = np.where(is_dead, 'DEAD', 'LIVE').astype(object)
    if has_live and has_dead:
        live_pos, dead_pos = group_pos['LIVE'], group_pos['DEAD']
        markers = np.full(n_cells, 'neither', dtype=object)
        markers[live_pos & ~dead_pos] = 'LIVE only'
        markers[dead_pos & ~live_pos] = 'DEAD only'
        markers[live_pos & dead_pos] = 'both'
        # v1.2.2: positive for neither marker -> no call (left out of the viability)
        status[~live_pos & ~dead_pos] = LD_UNCLASSIFIED
    elif has_dead:
        markers = np.where(group_pos['DEAD'], 'DEAD+', 'DEAD-')
    else:
        markers = np.where(group_pos['LIVE'], 'LIVE+', 'LIVE-')
    df.insert(0, 'Status', status)
    df.insert(1, 'LIVE/DEAD markers', markers)

    # Status layer: 1 = LIVE, 2 = DEAD, 3 = UNCLASSIFIED
    status_lut = np.zeros(n_cells + 1, dtype=np.uint8)
    status_lut[1:] = np.select([status == 'LIVE', status == 'DEAD'], [1, 2], default=3)
    seg['Status'] = status_lut[cells]

    df.attrs = {
        'is_2d': bool(is_2d),
        'ld_rule': rule,
        'rule_text': _ld_rule_text(rule, has_live, has_dead),
        'min_positive_fraction': float(min_positive_fraction),
        'localization': {str(c): loc.get(str(c).upper()) for c in stain_df.index},
        'markers': {str(c): str(stain_df.loc[c, 'Marker']) for c in stain_df.index},
        'call_groups': {role: list(conds) for role, conds in groups.items() if conds},
    }
    if any(len(v) > 1 for v in groups.values()):
        print("LIVE/DEAD markers of the call: " + "; ".join(
            f"{role} = {' or '.join(map(str, conds))}" for role, conds in groups.items() if conds))
    c = _ld_counts(df)
    print(f"LIVE/DEAD call ({df.attrs['rule_text']}; positive = marker covers >= "
          f"{100 * min_positive_fraction:.0f}% of its compartment): "
          f"{c['LIVE']} LIVE, {c['DEAD']} DEAD"
          + (f", {c[LD_UNCLASSIFIED]} UNCLASSIFIED" if c[LD_UNCLASSIFIED] else "")
          + f" of {n_cells} cells ({_ld_pct(c['viability'])} viability"
          + (", unclassified cells left out" if c[LD_UNCLASSIFIED] else "") + ")")
    return seg, df


def view_ld_results(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    r_xyz,
    r_zxyz,
    napari_module,
    progress=None,
):
    """LD Cell 20: the viewers of the nuclei notebook (`view_processing_results`:
    processing stages, threshold islands, marker labels, cells) plus the
    LIVE/DEAD layers: stained and estimated nuclei, and the LIVE and DEAD
    cells of the call. The empty PCM layer is left out."""
    viewer_0, viewer_1 = view_processing_results(
        im_final_stack,
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        r_xyz=r_xyz,
        r_zxyz=r_zxyz,
        napari_module=napari_module,
        progress=progress,
    )
    if viewer_1 is None:
        return viewer_0, viewer_1

    for layer in [l for l in viewer_1.layers if l.name == 'PCM']:
        viewer_1.layers.remove(layer)

    r_zX, r_zY, r_zZ = r_zxyz
    scale = (r_zZ, r_zY, r_zX)
    nuclei = im_segmentation_stack['Nuclei']
    stained = im_segmentation_stack.get('Nuclei (stained)')
    if stained is not None:
        viewer_1.add_labels(stained.astype(np.int32), name='NUCLEI (stained)',
                            blending='additive', scale=scale)
        estimated = np.where(stained > 0, 0, nuclei).astype(np.int32)
        if np.any(estimated):
            viewer_1.add_labels(estimated, name='NUCLEI (estimated)',
                                blending='additive', scale=scale)
    else:
        viewer_1.add_labels(nuclei.astype(np.int32), name='NUCLEI',
                            blending='additive', scale=scale)
    status = im_segmentation_stack.get('Status')
    if status is not None:
        for value, name, colormap in ((1, 'LIVE', 'green'), (2, 'DEAD', 'red'),
                                      (3, LD_UNCLASSIFIED, 'gray')):
            if value == 3 and not np.any(status == 3):
                continue
            viewer_1.add_image((status == value).astype(np.uint8), name=f'{name} cells (call)',
                               colormap=colormap, blending='additive', opacity=0.5,
                               contrast_limits=(0, 1), scale=scale)
    for layer in viewer_1.layers:
        layer.units = ('um', 'um', 'um')
    return viewer_0, viewer_1


def _ld_conditions(ld_cells_df):
    return [c for c in ld_cells_df.attrs.get('localization', {})
            if f'{c} positive' in ld_cells_df.columns]


def _ld_status_table(ld_cells_df):
    """Population table: one row per status (LIVE, DEAD and, if any,
    UNCLASSIFIED) and ALL cells."""
    loc = ld_cells_df.attrs.get('localization', {})
    size = _ld_size_names(ld_cells_df.attrs.get('is_2d', False))
    word, unit = size['word'], size['unit']
    total = len(ld_cells_df)
    rows = {}
    names = ['LIVE', 'DEAD']
    if (ld_cells_df['Status'] == LD_UNCLASSIFIED).any():
        names.append(LD_UNCLASSIFIED)
    for name in names + ['ALL']:
        sub = ld_cells_df if name == 'ALL' else ld_cells_df[ld_cells_df['Status'] == name]
        n = len(sub)
        nuc = sub[size['nucleus']].dropna()
        row = {
            'Cells': n,
            '% of cells': 100.0 * n / total if total else np.nan,
            f'Cell {word} mean [{unit}]': sub[size['cell']].mean() if n else np.nan,
            f'Cell {word} SD [{unit}]': sub[size['cell']].std(ddof=0) if n else np.nan,
            'Stained nuclei': int(len(nuc)),
            f'Nucleus {word} mean [{unit}]': nuc.mean() if len(nuc) else np.nan,
            f'Nucleus {word} SD [{unit}]': nuc.std(ddof=0) if len(nuc) else np.nan,
        }
        for cond in _ld_conditions(ld_cells_df):
            int_col = _ld_intensity_column(cond, loc.get(cond))
            row[f'{cond} positive cells'] = int(sub[f'{cond} positive'].sum())
            row[f'{cond} mean coverage'] = sub[f'{cond} coverage'].mean() if n else np.nan
            row[int_col.replace('mean int.', 'mean intensity')] = sub[int_col].mean() if n else np.nan
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.index.name = 'Population'
    return table


def _ld_stats_line(values, label, unit):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return
    print(f"{label}:  mean = {np.mean(arr):.2f} {unit}  |  std = {np.std(arr):.2f} {unit}"
          f"  |  median = {np.median(arr):.2f} {unit}"
          f"  |  [min = {np.min(arr):.2f},  max = {np.max(arr):.2f}]")


def print_ld_summary(ld_cells_df, stain_df):
    """LD Cell 22: the LIVE/DEAD call (counts, percentages, viability), how
    the marker combinations were called, the positivity of every marker and,
    per population, cell/nucleus size and marker intensity."""
    attrs = ld_cells_df.attrs
    loc = attrs.get('localization', {})
    size = _ld_size_names(attrs.get('is_2d', False))
    total = len(ld_cells_df)

    if attrs.get('is_2d'):
        print("2D IMAGE: cell and nucleus sizes are areas [um²].")
    print("CHANNELS:")
    for i, cond in enumerate(stain_df.index):
        row = stain_df.loc[cond]
        print(f"  [{i}] {cond:<12} {row['Marker']:<16} ({row['Laser']})  {loc.get(cond, '?')}")
    print("_" * 80)
    n_stained = int((ld_cells_df['Nucleus'] == 'stained').sum())
    print(f"TOT CELLS = {total}  ({n_stained} with a stained nucleus, "
          f"{total - n_stained} with the nucleus estimated from the cell body)")
    if total == 0:
        print("_" * 80)
        return
    print(f"LIVE/DEAD call: {attrs.get('rule_text', '')}")
    print(f"  (positive = marker covers >= {100 * attrs.get('min_positive_fraction', 0):.0f}% "
          "of its compartment: nucleus for nuclear, whole cell for cytoplasmic markers)")
    print(" ")
    c = _ld_counts(ld_cells_df)
    for status in LD_STATUSES:
        n = c[status]
        if status == LD_UNCLASSIFIED and not n:
            continue
        print(f"  {status:<12} = {n:>6} cells  ({100.0 * n / total:.1f} % of all cells)")
    print(f"  VIABILITY = {_ld_pct(c['viability'])}  (LIVE / (LIVE + DEAD), "
          f"{c['classified']} classified cells"
          + (f"; {c[LD_UNCLASSIFIED]} unclassified left out)" if c[LD_UNCLASSIFIED] else ")"))
    print(" ")
    print("  Marker combination -> call:")
    for combo, sub in ld_cells_df.groupby('LIVE/DEAD markers', sort=False):
        calls = ", ".join(f"{k} {v}" for k, v in sub['Status'].value_counts().items())
        print(f"    {combo:<10} {len(sub):>6} cells -> {calls}")
    print("_" * 80)

    print("MARKER POSITIVITY:")
    for cond in _ld_conditions(ld_cells_df):
        marker = attrs.get('markers', {}).get(cond, '')
        n_pos = int(ld_cells_df[f'{cond} positive'].sum())
        print(f"  {cond} ({marker}, {loc.get(cond)}) = {100.0 * n_pos / total:.1f} %  "
              f"({n_pos} cells)  |  mean coverage = {ld_cells_df[f'{cond} coverage'].mean():.2f}")
    print("_" * 80)

    for status in LD_STATUSES:
        sub = ld_cells_df[ld_cells_df['Status'] == status]
        if sub.empty:
            continue
        print(f"\n {status}  —  {len(sub)} cells  ({100.0 * len(sub) / total:.1f} %)")
        _ld_stats_line(sub[size['cell']], f"   Cell {size['word']}", size['pretty'])
        _ld_stats_line(sub[size['nucleus']], f"   Nucleus {size['word']} (stained)", size['pretty'])
        for cond in _ld_conditions(ld_cells_df):
            col = _ld_intensity_column(cond, loc.get(cond))
            _ld_stats_line(sub[col], f"   {col}", "a.u.")
    print("_" * 80)


def _ld_status_colors(stain_complete_df):
    colors = {}
    for status, default in (('LIVE', 'green'), ('DEAD', 'red')):
        if status in stain_complete_df.index:
            colors[status] = _condition_color(status, stain_complete_df)
        else:
            colors[status] = default
    colors[LD_UNCLASSIFIED] = '0.6'  # grey
    return colors


def _ld_present_statuses(ld_cells_df):
    """LIVE, DEAD and, if any cell has it, UNCLASSIFIED."""
    statuses = ['LIVE', 'DEAD']
    if (ld_cells_df['Status'] == LD_UNCLASSIFIED).any():
        statuses.append(LD_UNCLASSIFIED)
    return statuses


def plot_live_dead_classification(ld_cells_df, stain_complete_df, input_file=None,
                                  output_dir=None, show=True):
    """LD Cell 27: how the cells were called. Left: LIVE/DEAD counts. Right:
    each cell's LIVE vs DEAD coverage (fraction of its compartment covered by
    the marker), coloured by the call, with the positivity threshold as
    dashed lines (a histogram of the coverage when only one of the two
    channels is used). Saved as ``<file>_live_dead.png`` when *input_file*
    is given."""
    colors = _ld_status_colors(stain_complete_df)
    thr = ld_cells_df.attrs.get('min_positive_fraction', 0.0)
    total = len(ld_cells_df)
    fig, (ax_bar, ax) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={'width_ratios': (1, 2.2)})

    c = _ld_counts(ld_cells_df)
    statuses = _ld_present_statuses(ld_cells_df)
    values = [c[s] for s in statuses]
    bars = ax_bar.bar(['UNCLASS.' if s == LD_UNCLASSIFIED else s for s in statuses], values,
                      color=[colors[s] for s in statuses], edgecolor='black')
    for bar, value in zip(bars, values):
        pct = 100.0 * value / total if total else 0.0
        ax_bar.annotate(f"{value}\n({pct:.1f} %)", (bar.get_x() + bar.get_width() / 2, value),
                        ha='center', va='bottom', fontsize=10)
    ax_bar.set_ylabel("Cells")
    ax_bar.set_ylim(0, max(values + [1]) * 1.25)
    ax_bar.set_title(f"LIVE/DEAD call ({total} cells, viability {_ld_pct(c['viability'])})")

    # v1.2.2: coverage of the call groups (LIVE + LIVE_2 ..., DEAD + DEAD_2 ...:
    # the largest of their markers); older tables only have 'LIVE coverage'.
    def _cov_col(c):
        return f'{c} call coverage' if f'{c} call coverage' in ld_cells_df.columns else f'{c} coverage'
    groups = ld_cells_df.attrs.get('call_groups', {})

    def _cov_label(c):
        names = groups.get(c) or [c]
        return f"{' / '.join(map(str, names))} coverage" + (" (largest)" if len(names) > 1 else "")
    has = [c for c in ('LIVE', 'DEAD') if _cov_col(c) in ld_cells_df.columns]
    if len(has) == 2:
        for status in statuses:
            sub = ld_cells_df[ld_cells_df['Status'] == status]
            ax.scatter(sub[_cov_col('LIVE')], sub[_cov_col('DEAD')], s=12, alpha=0.6,
                       color=colors[status], edgecolors='none', label=f"{status} ({len(sub)})")
        ax.axvline(thr, color='0.3', linestyle='--', linewidth=1.2)
        ax.axhline(thr, color='0.3', linestyle='--', linewidth=1.2)
        ax.set_xlabel(f"{_cov_label('LIVE')} (fraction of compartment)")
        ax.set_ylabel(f"{_cov_label('DEAD')} (fraction of compartment)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    elif has:
        cond = has[0]
        bins = np.linspace(0, 1, 41)
        for status in statuses:
            sub = ld_cells_df[ld_cells_df['Status'] == status]
            ax.hist(sub[_cov_col(cond)], bins=bins, color=colors[status], alpha=0.6,
                    label=f"{status} ({len(sub)})")
        ax.axvline(thr, color='0.3', linestyle='--', linewidth=1.2)
        ax.set_xlabel(f"{_cov_label(cond)} (fraction of compartment)")
        ax.set_ylabel("Cells")
    ax.set_title(f"Per-cell coverage -- positive >= {thr:.2f} (dashed)")
    ax.grid(alpha=0.2)
    ax.legend(title="Call", loc="upper right")
    plt.tight_layout()
    if input_file is not None:
        png_path = _stem_output_path(input_file, "_live_dead.png", output_dir)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"LIVE/DEAD plot saved to: {png_path}")
    if show:
        plt.show()
    return fig, (ax_bar, ax)


def plot_ld_spatial_distributions(ld_cells_df, stain_complete_df, volume_shape, r_zxyz, bins=30):
    """LD Cell 25: where the LIVE and DEAD (and UNCLASSIFIED) cells are along
    X, Y and Z (number of cells, left axis) and the viability along each
    direction (LIVE / (LIVE + DEAD) in %, dashed, right axis) -- e.g.
    viability vs depth in a 3D construct. A 2D image has only X and Y."""
    colors = _ld_status_colors(stain_complete_df)
    statuses = _ld_present_statuses(ld_cells_df)
    r_zX, r_zY, r_zZ = r_zxyz
    extents = {'X': volume_shape[2] * r_zX, 'Y': volume_shape[1] * r_zY, 'Z': volume_shape[0] * r_zZ}
    axis_names = 'XYZ' if 'Z [um]' in ld_cells_df.columns else 'XY'
    fig, axs = plt.subplots(len(axis_names), 1, figsize=(15, 5 * len(axis_names)))
    for ax, axis_name in zip(axs, axis_names):
        edges = np.linspace(0, extents[axis_name], bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        counts = {}
        for status in statuses:
            values = ld_cells_df.loc[ld_cells_df['Status'] == status, f'{axis_name} [um]']
            counts[status], _ = np.histogram(values, bins=edges)
            ax.plot(centers, counts[status], color=colors[status], label=f'{status} cells',
                    linestyle=':' if status == LD_UNCLASSIFIED else '-')
        total = counts['LIVE'] + counts['DEAD']  # unclassified cells left out
        with np.errstate(invalid='ignore', divide='ignore'):
            viability = np.where(total > 0, 100.0 * counts['LIVE'] / total, np.nan)
        tax = ax.twinx()
        tax.plot(centers, viability, color='white', linestyle='--', marker='o', markersize=3,
                 label='Viability (LIVE / (LIVE + DEAD))')
        tax.set_ylim(0, 105)
        tax.set_ylabel("Viability [%]")
        ax.set_title(f"LIVE/DEAD {axis_name} DISTRIBUTION")
        ax.set_xlabel("[μm]")
        ax.set_ylabel("Cell count")
        ax.set_facecolor("black")
        handles, names = ax.get_legend_handles_labels()
        t_handles, t_names = tax.get_legend_handles_labels()
        tax.legend(handles + t_handles, names + t_names, loc="upper right", fontsize="small")
    return fig, axs


def plot_ld_size_distributions(ld_cells_df, stain_complete_df, bins=30):
    """LD Cell 26: cell and (stained) nucleus size of the LIVE and DEAD cells:
    volumes, or areas for a 2D image."""
    colors = _ld_status_colors(stain_complete_df)
    size = _ld_size_names(ld_cells_df.attrs.get('is_2d', False))
    word = size['word'].upper()
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    for ax, col, title in ((axs[0], size['cell'], f'CELL {word} DISTRIBUTION'),
                           (axs[1], size['nucleus'], f'NUCLEUS {word} DISTRIBUTION (stained nuclei)')):
        values = ld_cells_df[col].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
        else:
            edges = np.linspace(0, float(values.max()), bins + 1)
            for status in _ld_present_statuses(ld_cells_df):
                sub = ld_cells_df.loc[ld_cells_df['Status'] == status, col].dropna()
                if len(sub):
                    ax.hist(sub, bins=edges, color=colors[status], alpha=0.5,
                            label=f"{status} ({len(sub)})")
            ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_xlabel(f"[{size['pretty'].replace('um', 'μm')}]")
        ax.set_ylabel("Cells")
    return fig, axs


def export_live_dead_to_excel(input_file, original_stain_complete_df, ld_cells_df,
                              parameters=None, output_dir=None):
    """LD Cell 32: write the LIVE/DEAD results to ``<file>_live_dead.xlsx``
    (next to *input_file*, or inside *output_dir*):

    * **Summary** -- the call (counts, %, viability, settings) and the
      population table (LIVE, DEAD, ALL: size, positivity, intensity).
    * **Cells** -- one row per cell (``ld_cells_df``).
    * **Setup** -- stain table with each marker's localization, and the
      parameters (*parameters*, a dict).

    Returns
    -------
    output_path : str
    """
    from pathlib import Path as _Path

    if output_dir is not None:
        output_path = _stem_output_path(input_file, '_live_dead.xlsx', output_dir)
    else:
        output_path = (_Path(input_file).parent / _image_stem(input_file)).as_posix() + '_live_dead.xlsx'

    attrs = ld_cells_df.attrs
    total = len(ld_cells_df)
    c = _ld_counts(ld_cells_df)
    n_live, n_dead, n_unc = c['LIVE'], c['DEAD'], c[LD_UNCLASSIFIED]
    summary_rows = [
        ('Image', '2D (sizes are areas, um2)' if attrs.get('is_2d') else '3D (sizes are volumes, um3)'),
        ('Total cells', total),
        ('LIVE cells', n_live),
        ('DEAD cells', n_dead),
        ('UNCLASSIFIED cells (positive for neither)', n_unc),
        ('LIVE [% of all cells]', round(100.0 * n_live / total, 2) if total else ''),
        ('DEAD [% of all cells]', round(100.0 * n_dead / total, 2) if total else ''),
        ('UNCLASSIFIED [% of all cells]', round(100.0 * n_unc / total, 2) if total else ''),
        ('Viability [%] = LIVE / (LIVE + DEAD)',
         round(c['viability'], 2) if np.isfinite(c['viability']) else ''),
        ('Cells with a stained nucleus', int((ld_cells_df['Nucleus'] == 'stained').sum())),
        ('Cells with an estimated nucleus', int((ld_cells_df['Nucleus'] == 'estimated').sum())),
        ('Call rule', attrs.get('rule_text', '')),
        ('Positive = coverage of compartment >=', attrs.get('min_positive_fraction', '')),
    ]
    for combo, n in ld_cells_df['LIVE/DEAD markers'].value_counts().items():
        summary_rows.append((f'Markers: {combo}', int(n)))
    summary_df = pd.DataFrame(summary_rows, columns=['Item', 'Value'])
    status_table = _ld_status_table(ld_cells_df).reset_index()

    setup = original_stain_complete_df.copy()
    setup.insert(0, 'Localization', [attrs.get('localization', {}).get(str(c), '') for c in setup.index])
    setup = setup.reset_index()
    params_df = pd.DataFrame(list((parameters or {}).items()), columns=['Parameter', 'Value'])
    params_df['Value'] = params_df['Value'].astype(str)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        wb = writer.book
        title_fmt = wb.add_format({'bold': True, 'font_size': 12})
        live_fmt = wb.add_format({'bg_color': '#C6EFCE'})
        dead_fmt = wb.add_format({'bg_color': '#FFC7CE'})
        unc_fmt = wb.add_format({'bg_color': '#D9D9D9'})

        summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=1)
        start = len(summary_df) + 4
        status_table.to_excel(writer, sheet_name='Summary', index=False, startrow=start)
        ws = writer.sheets['Summary']
        ws.write(0, 0, 'LIVE/DEAD CALL', title_fmt)
        ws.write(start - 1, 0, 'POPULATIONS', title_fmt)
        ws.set_column(0, 0, 38)
        ws.set_column(1, status_table.shape[1], 18)

        cells_out = ld_cells_df.reset_index()
        cells_out.to_excel(writer, sheet_name='Cells', index=False)
        ws = writer.sheets['Cells']
        ws.set_column(0, cells_out.shape[1], 16)
        ws.freeze_panes(1, 1)
        if total:
            col = cells_out.columns.get_loc('Status')
            for value, fmt in (('LIVE', live_fmt), ('DEAD', dead_fmt), (LD_UNCLASSIFIED, unc_fmt)):
                ws.conditional_format(1, col, total, col, {
                    'type': 'cell', 'criteria': '==', 'value': f'"{value}"', 'format': fmt,
                })

        setup.to_excel(writer, sheet_name='Setup', index=False, startrow=1)
        p_start = len(setup) + 4
        params_df.to_excel(writer, sheet_name='Setup', index=False, startrow=p_start)
        ws = writer.sheets['Setup']
        ws.write(0, 0, 'STAIN TABLE', title_fmt)
        ws.write(p_start - 1, 0, 'PARAMETERS', title_fmt)
        ws.set_column(0, 0, 32)
        ws.set_column(1, setup.shape[1], 16)

    print(f"LIVE/DEAD report saved to: {output_path}")
    return output_path


def _ld_parameters(
    nuclei_diameter, cell_diameter, trig_cellpose,
    trig_stardist, trig_cellpose_cyto, localization, min_positive_fraction,
    ld_rule, size_reference, iterative_split, oversize_factor,
    max_split_iterations, merge_undersized, undersize_factor,
):
    loc = _ld_localization_map(localization)
    return {
        'Nuclei diameter (um)': nuclei_diameter,
        'Cell diameter (um)': cell_diameter,
        'Use Cellpose (nuclei)': trig_cellpose,
        'Use StarDist (nuclei)': trig_stardist,
        'Use Cellpose (cell bodies)': trig_cellpose_cyto,
        'Nuclear markers': [c for c, v in loc.items() if v == 'nuclear'],
        'Cytoplasmic markers': [c for c, v in loc.items() if v == 'cytoplasmic'],
        'Min positive fraction': min_positive_fraction,
        'LIVE/DEAD rule': ld_rule,
        'Size reference': size_reference,
        'Iterative split': iterative_split,
        'Oversize factor': oversize_factor,
        'Max split iterations': max_split_iterations,
        'Merge undersized': merge_undersized,
        'Undersize factor': undersize_factor,
    }


def _process_single_image_ld_batch(
    input_file,
    output_dir,
    roi_coords,
    stain_dict,
    localization,
    name_setup,
    use_setup,
    automatic_contrast,
    interactive_roi,
    nuclei_diameter,
    cell_diameter,
    scale_factor,
    zoom_factors,
    trig_cellpose,
    trig_stardist,
    trig_cellpose_cyto,
    multilabel,
    min_positive_fraction,
    ld_rule,
    iterative_split,
    oversize_factor,
    max_split_iterations,
    merge_undersized,
    undersize_factor,
    size_reference,
    sigma,
    num_plateaus,
    plateau_factor,
    threshold_method,
    export_vtk,
    export_stl,
    export_fea,
    settings,
    napari_module,
    progress,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.2,
    nuclei_split_config=None,
    nuclei_options=None,
    ld_min_brightness=0.0,
    ld_edge_brightness=0.0,
    ld_min_snr=0.0,
):
    """Run the LIVE/DEAD notebook (Cells 4-33) on one file, without viewers,
    inline plots or PNG subfolders, writing every output file into
    *output_dir*. Returns this image's LIVE/DEAD counts. A 2D image (single
    plane) is segmented in 2D, its sizes are areas and the 3D mesh exports
    are skipped.

    v1.2.2: ``nuclei_split_config`` (watershed split configuration, nuclei
    and cell bodies) and ``nuclei_options`` (the v1.6.4 NUCLEI options of
    `BATCH_NUCLEI_OPTIONS`, applied to the nuclear markers; missing keys =
    off, as in v1.2.1)."""
    opts = dict(BATCH_NUCLEI_OPTIONS, **(nuclei_options or {}))
    unknown = set(opts) - set(BATCH_NUCLEI_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown nuclei_options: {sorted(unknown)}")
    if opts["nuclei_contrast"] not in (None, "calibrated"):
        raise ValueError("nuclei_options['nuclei_contrast'] must be None or 'calibrated'")
    roi = list(roi_coords)
    if interactive_roi:
        roi = select_roi_interactively(input_file, roi, napari_module=napari_module,
                                       stain_table_path=(f"{name_setup}_ld_stain_dict.json"
                                                         if use_setup else None))

    # Cell 4
    (meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
     _, _, zooms, used_lazy_loading) = initialize_dataset(
        input_file, roi,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        scale_factor=scale_factor,
        zoom_factors=list(zoom_factors),
    )
    is_2d = img.shape[0] == 1

    # Cells 6-7
    im_final_stack, _, _, _, _, stain_df, _ = prepare_and_preview(
        img,
        nuclei_diameter, cell_diameter,
        stain_dict, file_meta,
        napari_module=napari_module, r_xyz=(r_X, r_Y, r_Z), progress=progress,
        show_viewer=False,
    )
    del img
    stain_df, stain_complete_df, original_stain_complete_df = prepare_stain_settings(
        im_final_stack['Original image'],
        stain_df=stain_df,
        name_setup=name_setup,
        use_setup=use_setup,
        automatic_contrast=automatic_contrast,
        settings=settings,
        napari_module=napari_module,
        r_xyz=(r_X, r_Y, r_Z),
        progress=progress,
    )

    # v1.2.2: nuclear markers' contrast and size adaptation (Cell 8)
    nuclear_conditions = [c for c, v in _ld_localization_map(localization).items() if v == 'nuclear']
    nuclear_in_image = ld_nuclear_conditions(stain_df, localization)
    if opts["nuclei_contrast"] == "calibrated":
        ld_calibrated_nuclear_contrast(im_final_stack['Original image'], stain_complete_df,
                                       original_stain_complete_df, nuclear_in_image,
                                       gamma=opts["nuclei_gamma"])
    histogram_note = "off"
    if opts["histogram_rules"]:  # v1.6.41 settings: NUCLEI rules on the nuclear markers
        changes, _ = ld_adapt_to_histograms(
            opts, opts["histogram_rules"], im_final_stack['Original image'], stain_complete_df,
            nuclear_in_image)
        histogram_note = "; ".join(f"{n} {o} -> {v} ({s})" for n, o, v, s in changes) or "no change"
    adapted_note = "off"
    if opts["auto_adapt"] and not is_2d and nuclear_in_image:
        adapted, changes = ld_adapt_settings_to_image(
            im_final_stack['Original image'], stain_complete_df, (r_Z, r_Y, r_X),
            nuclear_in_image,
            {"nuclei_diameter": nuclei_diameter, "scale_factor": scale_factor,
             "z_bg_radius_um": opts["z_bg_radius_um"],
             "nuclei_edge_margin_um": opts["nuclei_edge_margin_um"]},
            adapt_scale=opts["adapt_scale"])
        nuclei_diameter = adapted["nuclei_diameter"]
        opts["z_bg_radius_um"] = adapted["z_bg_radius_um"]
        opts["nuclei_edge_margin_um"] = adapted["nuclei_edge_margin_um"]
        if adapted["scale_factor"] != scale_factor:
            scale_factor = adapted["scale_factor"]
            zooms = [z * scale_factor for z in zoom_factors]
        adapted_note = "; ".join(f"{n} {o} -> {v}" for n, o, v, _ in changes) or "no change"
    elif opts["auto_adapt"] and is_2d and opts["adapt_scale"]:  # 2D: resolution only, for the given nucleus size
        scale_factor, changes = adapt_scale_2d(nuclei_diameter, scale_factor, r_X)
        zooms = [z * scale_factor for z in zoom_factors]
        zooms[0] = 1.0  # never resample a single plane along Z
        adapted_note = "; ".join(f"{n} {o} -> {v}" for n, o, v, _ in changes) or "no change"

    # Cells 9-15
    if opts["z_atten_model"] != "off":  # v1.2.2
        im_final_stack['Original image'], _, _ = ld_correct_depth(
            im_final_stack['Original image'], stain_complete_df, nuclear_in_image,
            opts["z_atten_conditions"], model=opts["z_atten_model"],
            strength=opts["z_atten_strength"])
    im_final_stack = run_normalize(im_final_stack, stain_complete_df=stain_complete_df)
    im_final_stack, r_zX, r_zY, r_zZ = run_resample(
        im_final_stack, stain_complete_df=stain_complete_df, zoom_factors=zooms, meta=meta,
        edge_mode=opts["resample_edge_mode"],
    )
    im_final_stack = run_denoise(im_final_stack, stain_complete_df=stain_complete_df)
    im_final_stack['Denoised image'] = ld_remove_nuclear_haze(  # v1.2.2
        im_final_stack['Denoised image'], stain_complete_df, nuclear_in_image,
        opts["z_bg_radius_um"], (r_zZ, r_zY, r_zX), percentile=opts["z_bg_percentile"])
    im_final_stack = run_contrast_gamma(im_final_stack, stain_complete_df=stain_complete_df)
    im_final_stack = run_smooth(im_final_stack, stain_complete_df=stain_complete_df, sigma=sigma)
    im_final_stack = run_equalize(
        im_final_stack, stain_complete_df=stain_complete_df,
        num_plateaus=num_plateaus, plateau_factor=plateau_factor,
    )
    im_final_stack = run_threshold(
        im_final_stack,
        stain_complete_df=stain_complete_df,
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        r_zX=r_zX, r_zY=r_zY, r_zZ=r_zZ,
        threshold_method=threshold_method,
        global_threshold_weight=opts["global_threshold_weight"],
        sauvola_weight=opts["sauvola_weight"],
        nuclear_conditions=nuclear_conditions,
        progress=progress,
    )
    im_final_stack = ld_binary_threshold(  # v1.2.2: clearly bright LIVE/DEAD signal only
        im_final_stack, stain_complete_df, nuclei_diameter, (r_zX, r_zY, r_zZ),
        nuclear_conditions=nuclear_conditions, min_brightness=ld_min_brightness,
        edge_brightness=ld_edge_brightness, min_snr=ld_min_snr)

    # Cell 16
    ld_params = _ld_parameters(
        nuclei_diameter, cell_diameter, trig_cellpose,
        trig_stardist, trig_cellpose_cyto, localization, min_positive_fraction,
        ld_rule, size_reference, iterative_split, oversize_factor,
        max_split_iterations, merge_undersized, undersize_factor,
    )
    ld_params['Split gamma'] = split_gamma
    ld_params['Split gamma step'] = split_gamma_step
    ld_params['Min split piece fraction'] = min_piece_fraction
    ld_params['Scale factor'] = scale_factor
    ld_params['LD min brightness'] = ld_min_brightness
    ld_params['LD edge brightness'] = ld_edge_brightness
    ld_params['LD min SNR'] = ld_min_snr
    if nuclei_split_config is not None:
        ld_params['Watershed split config'] = str(nuclei_split_config)
    ld_params.update({f"Nuclei option {k}": (str(v) if isinstance(v, (list, dict)) else v)
                      for k, v in opts.items()})
    ld_params['Settings adapted to the image'] = adapted_note
    ld_params['Settings adapted to the histograms'] = histogram_note
    export_channel_histograms(
        im_final_stack, stain_complete_df, input_file,
        ROI_print=ROI_print, lazy_loading_used=used_lazy_loading,
        name_setup=name_setup,
        nuclei_diameter=nuclei_diameter, cell_diameter=cell_diameter,
        zoom_factors=zooms, scale_factor=scale_factor,
        trig_cellpose=trig_cellpose, trig_stardist=trig_stardist,
        trig_cellpose_cyto=trig_cellpose_cyto,
        multilabel=multilabel,
        sigma=sigma, num_plateaus=num_plateaus, plateau_factor=plateau_factor,
        threshold_method=threshold_method,
        global_threshold_weight=opts["global_threshold_weight"],
        sauvola_weight=opts["sauvola_weight"],
        nuclei_split_config=nuclei_split_config,
        aggregate_distance_factor=None,  # no aggregates / PCM in the LD analysis
        progress=progress,
        output_dir=output_dir,
        extra_params=ld_params,
    )

    # Cells 17-19
    size_kwargs = dict(
        iterative_split=iterative_split,
        oversize_factor=oversize_factor,
        max_split_iterations=max_split_iterations,
        merge_undersized=merge_undersized,
        undersize_factor=undersize_factor,
        size_reference=size_reference,
        split_gamma=split_gamma,
        split_gamma_step=split_gamma_step,
        min_piece_fraction=min_piece_fraction,
        nuclei_split_config=nuclei_split_config,
        progress=progress,
    )
    im_segmentation_stack = segment_ld_nuclei(
        im_final_stack,
        stain_df=stain_df,
        localization=localization,
        r_zxyz=(r_zX, r_zY, r_zZ),
        nuclei_diameter=nuclei_diameter,
        trig_stardist=trig_stardist,
        trig_cellpose=trig_cellpose,
        **size_kwargs,
        **{k: opts[k] for k in _BATCH_SEGMENT_NUCLEI_KEYS},  # v1.2.2
    )
    im_segmentation_stack, stain_complete_df = segment_ld_cells(
        im_final_stack,
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        localization=localization,
        r_zxyz=(r_zX, r_zY, r_zZ),
        nuclei_diameter=nuclei_diameter,
        cell_diameter=cell_diameter,
        trig_cellpose_cyto=trig_cellpose_cyto,
        **size_kwargs,
    )
    im_segmentation_stack, ld_cells_df = classify_live_dead(
        im_final_stack,
        im_segmentation_stack=im_segmentation_stack,
        stain_df=stain_df,
        localization=localization,
        r_zxyz=(r_zX, r_zY, r_zZ),
        min_positive_fraction=min_positive_fraction,
        ld_rule=ld_rule,
        progress=progress,
    )

    # _context() looks at this module's globals first, which may still hold
    # the previous image's tables -- register this image's before quantifying.
    set_notebook_context(stain_df=stain_df, stain_complete_df=stain_complete_df)

    # Cells 21 and 23
    percell_mean_df, percell_std_df = compute_percell_marker_intensity_df(
        im_segmentation_stack, im_final_stack,
        stain_complete_df=stain_complete_df,
        progress=progress,
    )
    labels_full_df = build_full_labels_df(
        im_segmentation_stack, im_final_stack,
        stain_complete_df=stain_complete_df,
        r_zX=r_zX, r_zY=r_zY, r_zZ=r_zZ,
        zooms=zooms,
        percell_mean_df=percell_mean_df,
        percell_std_df=percell_std_df,
        progress=progress,
        is_2d=is_2d,
    )

    # Cells 27-28 -- saved as PNG only, not displayed
    plot_live_dead_classification(
        ld_cells_df, stain_complete_df,
        input_file=input_file, output_dir=output_dir, show=False,
    )
    plot_marker_intensity_clouds(
        percell_mean_df,
        stain_complete_df=stain_complete_df,
        im_segmentation_stack=im_segmentation_stack,
        r_xyz=(r_zX, r_zY, r_zZ),
        stain_df=stain_df,
        progress=progress,
        input_file=input_file,
        output_dir=output_dir,
        show=False,
        is_2d=is_2d,
    )
    plt.close("all")

    # Cell 32 (written first: it is the main result and the cheapest export)
    export_live_dead_to_excel(
        input_file, original_stain_complete_df, ld_cells_df,
        parameters=ld_params, output_dir=output_dir,
    )

    # Cell 24 -- PDF only, PNGs go to a temporary folder
    build_histogram_report(
        im_segmentation_stack, im_final_stack,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        input_file=input_file,
        progress=progress,
        output_dir=output_dir,
        keep_png=False,
        show_plot=False,
    )

    # Cells 29, 30, 33 -- 3D meshes, not made for a 2D image
    if is_2d and (export_vtk or export_stl or export_fea):
        print("2D image: VTK, STL and FEA mesh exports skipped (3D only).")
    if export_vtk and not is_2d:
        build_vtk_volumes(
            im_segmentation_stack,
            labels_full_df=labels_full_df,
            stain_complete_df=stain_complete_df,
            input_file=input_file,
            r_xyz=(r_X, r_Y, r_Z),
            zoom_factors=zooms,
            progress=progress,
            output_dir=output_dir,
            live_counter=False,
        )
    if export_stl and not is_2d:
        export_marker_stl(
            im_segmentation_stack,
            stain_df=stain_df,
            stain_complete_df=stain_complete_df,
            input_file=input_file,
            r_xyz=(r_X, r_Y, r_Z),
            zoom_factors=zooms,
            progress=progress,
            output_dir=output_dir,
        )
    if export_fea and not is_2d:
        export_fea_mesh(
            {'Nuclei': im_segmentation_stack['Cytoplasm']},
            input_file=input_file,
            r_zxyz=(r_zX, r_zY, r_zZ),
            progress=progress,
            output_dir=output_dir,
        )

    c = _ld_counts(ld_cells_df)
    return {
        'Image': '2D' if is_2d else '3D',
        'Cells': c['total'],
        'LIVE': c['LIVE'],
        'DEAD': c['DEAD'],
        'UNCLASSIFIED': c[LD_UNCLASSIFIED],
        # LIVE / (LIVE + DEAD): the unclassified cells are left out
        'Viability [%]': round(c['viability'], 2) if np.isfinite(c['viability']) else np.nan,
    }


def run_ld_batch_folder(
    input_folder,
    roi_coords,
    stain_dict,
    localization,
    name_setup,
    use_setup=True,
    automatic_contrast=False,
    interactive_roi=False,
    nuclei_diameter=10.0,
    cell_diameter=30.0,
    scale_factor=1.0,
    zoom_factors=None,
    trig_cellpose=False,
    trig_stardist=False,
    trig_cellpose_cyto=False,
    multilabel=True,
    min_positive_fraction=0.2,
    ld_rule="dead_priority",
    iterative_split=True,
    oversize_factor=2.0,
    max_split_iterations=3,
    merge_undersized=True,
    undersize_factor=0.5,
    size_reference="blend",
    sigma=0.5,
    num_plateaus=2,
    plateau_factor=0.7,
    threshold_method='otsu',
    export_vtk=True,
    export_stl=True,
    export_fea=True,
    settings=None,
    napari_module=None,
    progress=None,
    split_gamma=1.0,
    split_gamma_step=0.0,
    min_piece_fraction=0.2,
    nuclei_split_config=None,
    nuclei_options=None,
    ld_min_brightness=0.0,
    ld_edge_brightness=0.0,
    ld_min_snr=0.0,
    calibrated_contrast=None,
):
    """Process every supported image file in *input_folder* with the full
    LIVE/DEAD pipeline and write each image's outputs into its own
    subfolder, ``<input_folder>_output/<image stem>/`` -- the LD counterpart
    of `run_batch_folder` (same rules for the contrast setup, failed files
    and outputs). The folder may hold LIVE/DEAD pair files (``.ldpair``,
    `combine_live_dead_inputs`) instead of stacked images.

    v1.2.2: ``nuclei_split_config`` and ``nuclei_options`` (see
    `_process_single_image_ld_batch`) carry the validated settings of the
    nuclei notebook v1.6.4 (`helpers.optimal_settings_ld.ld_batch_settings`);
    every image gets its own nuclear-marker contrast and size adaptation.
    ``ld_min_brightness``, ``ld_edge_brightness``, ``ld_min_snr``: the
    bright-signal filter of the LIVE/DEAD masks (`ld_binary_threshold`;
    default 0 = off).

    Returns
    -------
    summary : DataFrame
        One row per file: status ('OK'/'FAILED'), cells, LIVE, DEAD,
        viability, output subfolder and error message. Also saved as
        ``live_dead_batch_summary.xlsx`` in the output folder.
    """
    import traceback

    automatic_contrast = _contrast_mode(automatic_contrast, calibrated_contrast)
    image_files = list_image_files(input_folder)
    if not image_files:
        raise FileNotFoundError(
            f"No supported image files ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) "
            f"found in {input_folder}"
        )

    output_dir = get_batch_output_dir(input_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    if zoom_factors is None:
        zoom_factors = [1.0, 1.0, 1.0]

    print(f"Batch mode: {len(image_files)} image file(s) found in {input_folder}")
    print(f"Outputs are written to: {output_dir}")

    results = []
    for n, image_path in enumerate(image_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{n}/{len(image_files)}] {image_path.name}")
        print("=" * 80)
        image_output_dir = output_dir / _image_stem(image_path)
        row = {'File': image_path.name, 'Status': 'OK', 'Image': '', 'Cells': np.nan, 'LIVE': np.nan,
               'DEAD': np.nan, 'UNCLASSIFIED': np.nan, 'Viability [%]': np.nan,
               'Output folder': str(image_output_dir), 'Error': ''}
        try:
            image_output_dir.mkdir(parents=True, exist_ok=True)
            row.update(_process_single_image_ld_batch(
                str(image_path),
                image_output_dir,
                roi_coords=roi_coords,
                stain_dict=stain_dict,
                localization=localization,
                name_setup=name_setup,
                use_setup=use_setup,
                automatic_contrast=automatic_contrast,
                interactive_roi=interactive_roi,
                nuclei_diameter=nuclei_diameter,
                cell_diameter=cell_diameter,
                scale_factor=scale_factor,
                zoom_factors=zoom_factors,
                trig_cellpose=trig_cellpose,
                trig_stardist=trig_stardist,
                trig_cellpose_cyto=trig_cellpose_cyto,
                multilabel=multilabel,
                min_positive_fraction=min_positive_fraction,
                ld_rule=ld_rule,
                split_gamma=split_gamma,
                split_gamma_step=split_gamma_step,
                min_piece_fraction=min_piece_fraction,
                nuclei_split_config=nuclei_split_config,
                nuclei_options=nuclei_options,
                ld_min_brightness=ld_min_brightness,
                ld_edge_brightness=ld_edge_brightness,
                ld_min_snr=ld_min_snr,
                iterative_split=iterative_split,
                oversize_factor=oversize_factor,
                max_split_iterations=max_split_iterations,
                merge_undersized=merge_undersized,
                undersize_factor=undersize_factor,
                size_reference=size_reference,
                sigma=sigma,
                num_plateaus=num_plateaus,
                plateau_factor=plateau_factor,
                threshold_method=threshold_method,
                export_vtk=export_vtk,
                export_stl=export_stl,
                export_fea=export_fea,
                settings=settings,
                napari_module=napari_module,
                progress=progress,
            ))
        except Exception as exc:
            traceback.print_exc()
            print(f"[run_ld_batch_folder] {image_path.name} FAILED -- continuing with the next file.")
            row['Status'] = 'FAILED'
            row['Error'] = f"{type(exc).__name__}: {exc}"
        finally:
            plt.close("all")
            gc.collect()
        results.append(row)

    summary = pd.DataFrame(results)
    try:
        summary_path = output_dir / 'live_dead_batch_summary.xlsx'
        summary.to_excel(summary_path, index=False)
        print(f"Batch summary saved to: {summary_path}")
    except Exception as exc:
        print(f"[run_ld_batch_folder] Could not save the batch summary: {exc}")
    n_ok = int((summary["Status"] == "OK").sum())
    print("\n" + "=" * 80)
    print(f"Batch finished: {n_ok}/{len(summary)} file(s) processed successfully.")
    print(f"Outputs: {output_dir}")
    return summary

