from collections import defaultdict
import os

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
    "build_labels_df",
    "build_labels_dict",
    "collect_histogram_data",
    "compute_full_marker_stats_for_marker",
    "compute_marker_stats_for_marker",
    "compute_nuclei_cytoplasm_stats",
    "contr_limit",
    "contr_stretch",
    "create_row_pdf",
    "crop_nucleus_with_padding",
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
    "get_nuclei_split_config",
    "get_stain_name",
    "grow_labels",
    "grow_markers_within_islands_limited",
    "hist_plot",
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
    "segment_nuclei",
    "segment_nuclei_cellpose",
    "segment_nuclei_watershed",
    "segment_cytoplasm",
    "segment_pcm",
    "set_notebook_context",
    "shrink_to_markers",
    "shrink_to_markers_robust",
    "stardist3d_from_2d",
    "truncate_cell",
    "voxel_volume",
    "watershed_nuclei",
    "reload_helpers",
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


def open_image_file(input_file: str):
    """Open a microscopy file with AICSImage, falling back to a nd2reader-based
    wrapper for legacy ND2 files that raise ``ValueError: Invalid ChunkMap signature``.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file (.nd2, .czi, .tif, …).

    Returns
    -------
    meta : AICSImage or _ND2ReaderFallback
        Object with the same interface used by the notebook cells
        (``shape``, ``physical_pixel_sizes``, ``get_image_data``,
        ``get_image_dask_data``).
    """
    from aicsimageio import AICSImage
    try:
        return AICSImage(input_file)
    except ValueError as exc:
        if 'ChunkMap' not in str(exc) or not input_file.lower().endswith('.nd2'):
            raise
        print(
            f"[open_image_file] AICSImage raised '{exc}'.\n"
            "Falling back to nd2reader for legacy ND2 file — note: the full "
            "image will be loaded into memory even when big_image=True."
        )
        return _ND2ReaderFallback(input_file)


def load_image_with_roi(input_file: str, roi_coords, big_image=True):
    """Open a microscopy file and load the requested ROI in ZYXC order."""
    meta = open_image_file(input_file)
    image, _ = extract_roi_from_metadata(meta, roi_coords, big_image=big_image)
    print(image.shape)
    return meta, image


def load_image_and_metadata(input_file, roi_coords, big_image=True):
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
    """
    ROI_print = list(roi_coords)
    meta, img = load_image_with_roi(input_file, roi_coords, big_image=big_image)
    r_X = meta.physical_pixel_sizes.X
    r_Y = meta.physical_pixel_sizes.Y
    r_Z = meta.physical_pixel_sizes.Z
    file_meta = read_file_metadata(input_file, meta)
    print(f"Voxel size: [{r_X}, {r_Y}, {r_Z}] um")
    print(f"Date: {file_meta['date']}")
    print(f"Channels: {file_meta['channels']}")
    return meta, img, r_X, r_Y, r_Z, file_meta, ROI_print


def initialize_dataset(input_file, roi_coords, big_image=True,
                       nuclei_diameter=10.0, cell_diameter=30.0,
                       scale_factor=1.0, zoom_factors=None):
    """Load image, read metadata, and compute derived parameters.

    Wraps ``load_image_and_metadata`` and adds expansion-factor and
    zoom-factor computation so the notebook cell is a single call.

    Parameters
    ----------
    input_file : str
        Path to the microscopy file.
    roi_coords : list of 6 ints
        [x0, x1, y0, y1, z0, z1].
    big_image : bool
        If True, use lazy/dask loading with ROI.
    nuclei_diameter, cell_diameter : float
        Approximate diameters in micrometers.
    scale_factor : float
        Extra manual resolution scaling (default 1.0).
    zoom_factors : list of 3 floats or None
        Base [Z, Y, X] zoom factors (default [1, 1, 1]).

    Returns
    -------
    meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
    cyto_factor, PCM_factor, zoom_factors
    """
    if zoom_factors is None:
        zoom_factors = [1.0, 1.0, 1.0]

    meta, img, r_X, r_Y, r_Z, file_meta, ROI_print = load_image_and_metadata(
        input_file, roi_coords, big_image=big_image,
    )

    cyto_factor = int(np.round(cell_diameter / nuclei_diameter))
    PCM_factor = int(cyto_factor * 1.1)
    if PCM_factor == cyto_factor:
        PCM_factor += 1

    zoom_factors = [x * scale_factor for x in zoom_factors]

    return (meta, img, r_X, r_Y, r_Z, file_meta, ROI_print,
            cyto_factor, PCM_factor, zoom_factors)


def prepare_and_preview(img, meta, ROI, big_image,
                        nuclei_diameter, cell_diameter,
                        stain_dict, file_meta,
                        napari_module, progress=None):
    """Prepare image stack, build stain table, and open napari preview.

    Combines ``prepare_image_stack``, ``build_stain_dataframe``, and
    ``view_original_channels`` into a single call.

    Returns
    -------
    im_final_stack : dict
    nuclei_radius, cell_radius, nuclei_volume, cell_volume : float
    stain_df : DataFrame
    viewer : napari.Viewer
    """
    im_final_stack, nuclei_radius, cell_radius, nuclei_volume, cell_volume = (
        prepare_image_stack(img, meta, ROI, big_image, nuclei_diameter, cell_diameter)
    )
    stain_df = build_stain_dataframe(stain_dict, file_meta)
    viewer = view_original_channels(im_final_stack, stain_df, napari_module, progress=progress)
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
        "z_split_aggressive": False,
    }
    config.update(overrides)
    return config


def hist_plot(im_in, stain_complete_df, thresh=0, legend=False):
    """Plot histogram and CDF for each channel."""
    fig, axs = plt.subplots(1, im_in.shape[3], figsize=(15, 2))
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
        axs[c].set_title(stain_complete_df.index[c])
        axs[c].set_yscale("log")
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


def remove_small_islands(binary_matrix, area_threshold):
    """Remove small connected components from a binary mask."""
    labeled_array, num_features = ndi.label(binary_matrix)
    for component_id in range(1, num_features + 1):
        component = labeled_array == component_id
        if component.sum() < area_threshold:
            binary_matrix[component] = 0
    return binary_matrix


def stardist3d_from_2d(
    img_3d,
    model_name="2D_versatile_fluo",
    nucleus_radius=5,
    voxel_size=(1.0, 0.5, 0.5),
    norm=True,
):
    """Apply StarDist2D slice by slice, then split merged objects in 3D."""
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    if img_3d.ndim != 3:
        raise ValueError("Input must be 3D with shape (Z, Y, X)")

    z_spacing, y_spacing, x_spacing = voxel_size

    print(f"Running StarDist2D on {img_3d.shape[0]} z-slices...")
    model = StarDist2D.from_pretrained(model_name)

    labels_3d = np.zeros_like(img_3d, dtype=np.int32)
    current_label = 1

    for z_index in range(img_3d.shape[0]):
        img = img_3d[z_index]
        if norm:
            img = normalize(img, 1, 99.8, axis=None)

        labels_2d, _ = model.predict_instances(img)
        labels_2d = np.where(labels_2d > 0, labels_2d + current_label, 0)
        labels_3d[z_index] = labels_2d
        current_label = labels_2d.max() + 1

    labels_3d = skimage.measure.label(labels_3d > 0, connectivity=1)

    print("Computing distance transform with anisotropic voxel spacing...")
    distance = ndi.distance_transform_edt(labels_3d > 0, sampling=voxel_size)

    footprint = np.ones(
        (
            max(1, int(round(z_spacing / y_spacing))),
            max(1, int(round(nucleus_radius))),
            max(1, int(round(nucleus_radius))),
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

    print("Running 3D watershed to split connected nuclei...")
    labels_split = watershed(-distance, markers, mask=labels_3d > 0)

    print(f"Done. Found {labels_split.max()} nuclei.")
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
    stain_table=None,
    nucleus_color="blue",
    cytoplasm_color="green",
    pcm_color="magenta",
    pad=20,
    out_dir="merged_png",
):
    """Build and save a merged RGB overview around a single nucleus."""
    del nucleus_color, cytoplasm_color, pcm_color

    if stain_table is None:
        try:
            stain_table = _context("stain_complete_df")
        except RuntimeError:
            stain_table = None

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

        if (
            stain_table is not None
            and condition in stain_table.index
            and "Cont_min" in stain_table.columns
        ):
            try:
                clim = (
                    stain_table.loc[condition, "Cont_min"],
                    stain_table.loc[condition, "Cont_max"],
                )
                gamma_value = (
                    stain_table.loc[condition, "Gamma"]
                    if "Gamma" in stain_table.columns
                    else 1.0
                )
                img_display = napari_contrast_gamma_uint8(
                    img_small.astype(np.float32),
                    (float(clim[0]), float(clim[1])),
                    float(gamma_value),
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
    neighbors = defaultdict(set)
    offsets = _build_neighbor_offsets(connectivity)
    z_size, y_size, x_size = marker_labels.shape

    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                current_label = marker_labels[z_index, y_index, x_index]
                if current_label == 0:
                    continue
                for dz, dy, dx in offsets:
                    nz = z_index + dz
                    ny = y_index + dy
                    nx = x_index + dx
                    if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
                        neighbor_label = marker_labels[nz, ny, nx]
                        if neighbor_label > 0 and neighbor_label != current_label:
                            neighbors[current_label].add(neighbor_label)
    return neighbors


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

        if size_label <= size_ratio_thresh * size_biggest or size_label < 20.0:
            new_label[label] = biggest_neighbor

    relabeled = marker_labels.copy()
    for label, target in new_label.items():
        if label != target:
            relabeled[marker_labels == label] = target

    final_labels, _ = ndi.label(relabeled > 0, structure=structure)
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

        if size_label <= size_ratio_thresh * size_biggest or size_label < min_cell_size:
            new_label[label] = biggest_neighbor

    relabeled = marker_labels.copy()
    for label, target in new_label.items():
        if label != target:
            relabeled[marker_labels == label] = target

    final_labels, _ = ndi.label(
        relabeled > 0,
        structure=ndi.generate_binary_structure(3, connectivity),
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


def compute_nuclei_cytoplasm_stats(seg_stack, r_xyz, zooms):
    """Compute centroid positions and volumes for nuclei and cytoplasm."""
    max_label = int(np.max(seg_stack["Nuclei"]))
    nucleus_positions = []
    nucleus_sizes = []
    cytoplasm_positions = []
    cytoplasm_sizes = []

    for label_id in range(1, max_label + 1):
        z_nuc, y_nuc, x_nuc = np.where(seg_stack["Nuclei"] == label_id)
        if x_nuc.size == 0:
            nucleus_positions.append((0.0, 0.0, 0.0))
            nucleus_sizes.append(0.0)
        else:
            nucleus_positions.append(
                (
                    np.mean(x_nuc) * r_xyz[0],
                    np.mean(y_nuc) * r_xyz[1],
                    np.mean(z_nuc) * r_xyz[2],
                )
            )
            nucleus_sizes.append(x_nuc.size * r_xyz[0] * r_xyz[1] * r_xyz[2])

        z_cyto, y_cyto, x_cyto = np.where(seg_stack["Cytoplasm"] == label_id)
        if x_cyto.size == 0:
            cytoplasm_positions.append((0.0, 0.0, 0.0))
            cytoplasm_sizes.append(0.0)
        else:
            cytoplasm_positions.append(
                (
                    np.mean(x_cyto) * r_xyz[0],
                    np.mean(y_cyto) * r_xyz[1],
                    np.mean(z_cyto) * r_xyz[2],
                )
            )
            cytoplasm_sizes.append(x_cyto.size * r_xyz[0] * r_xyz[1] * r_xyz[2])

    return nucleus_positions, nucleus_sizes, cytoplasm_positions, cytoplasm_sizes


def compute_marker_stats_for_marker(marker_idx, seg_stack, filtered_img, r_xyz, zooms):
    """Compute marker measurements per nucleus for one marker channel."""
    stain_complete_df = _context("stain_complete_df")
    stain_df = _context("stain_df")
    condition = stain_complete_df.index[marker_idx]
    seg_key = stain_df.index[marker_idx]

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

    max_label = int(np.max(seg_stack["Nuclei"]))
    for label_id in range(1, max_label + 1):
        nucleus_mask = seg_stack["Nuclei"] == label_id
        cytoplasm_mask = seg_stack["Cytoplasm"] == label_id
        pcm_mask = seg_stack["PCM"] == label_id
        marker_mask = (marker_img > 0) & ((nucleus_mask + cytoplasm_mask + pcm_mask) > 0)
        if not np.any(marker_mask):
            continue

        shared_labels.append(label_id)
        voxels = np.where(marker_mask)
        marker_sizes.append(voxels[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
        if channel_idx is not None:
            values = filtered_img[voxels[0], voxels[1], voxels[2], channel_idx]
            avg_marker.append(float(np.mean(values)) if values.size > 0 else 0.0)
            std_marker.append(float(np.std(values)) if values.size > 0 else 0.0)
        else:
            avg_marker.append(0.0)
            std_marker.append(0.0)

        if marker_img_cyto is not None:
            cytoplasm_marker_mask = (marker_img_cyto > 0) & cytoplasm_mask
            voxels_cyto = np.where(cytoplasm_marker_mask)
            marker_cyto_sizes.append(voxels_cyto[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
            if channel_idx is not None and voxels_cyto[0].size > 0:
                values_cyto = filtered_img[voxels_cyto[0], voxels_cyto[1], voxels_cyto[2], channel_idx]
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
            pcm_marker_mask = (marker_img_pcm > 0) & pcm_mask
            voxels_pcm = np.where(pcm_marker_mask)
            marker_pcm_sizes.append(voxels_pcm[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
            if channel_idx is not None and voxels_pcm[0].size > 0:
                values_pcm = filtered_img[voxels_pcm[0], voxels_pcm[1], voxels_pcm[2], channel_idx]
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


def compute_full_marker_stats_for_marker(marker_idx, seg_final, seg_stack, filtered_img, r_xyz, zooms):
    """Compute full marker measurements from the filtered image channel."""
    stain_complete_df = _context("stain_complete_df")
    condition = stain_complete_df.index[marker_idx]
    channel_idx = None
    for idx, name in enumerate(stain_complete_df.index):
        if name == condition:
            channel_idx = idx
            break

    marker_img = seg_final["Filtered image"][:, :, :, marker_idx]
    if marker_img is None:
        return [], [], [], [], [], [], [], [], [], []

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

    max_label = int(np.max(seg_stack["Nuclei"]))
    for label_id in range(1, max_label + 1):
        nucleus_mask = seg_stack["Nuclei"] == label_id
        cytoplasm_mask = seg_stack["Cytoplasm"] == label_id
        pcm_mask = seg_stack["PCM"] == label_id
        marker_mask = ((nucleus_mask + cytoplasm_mask + pcm_mask) > 0) & (marker_img > 0)

        shared_labels.append(label_id)
        voxels = np.where(marker_mask)
        marker_sizes.append(voxels[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
        if channel_idx is not None:
            values = filtered_img[voxels[0], voxels[1], voxels[2], channel_idx]
            avg_marker.append(float(np.mean(values)) if values.size > 0 else 0.0)
            std_marker.append(float(np.std(values)) if values.size > 0 else 0.0)
        else:
            avg_marker.append(0.0)
            std_marker.append(0.0)

        cytoplasm_marker_mask = (marker_img > 0) & cytoplasm_mask
        voxels_cyto = np.where(cytoplasm_marker_mask)
        marker_cyto_sizes.append(voxels_cyto[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
        if channel_idx is not None and voxels_cyto[0].size > 0:
            values_cyto = filtered_img[voxels_cyto[0], voxels_cyto[1], voxels_cyto[2], channel_idx]
            avg_cyto_marker.append(float(np.mean(values_cyto)))
            std_cyto_marker.append(float(np.std(values_cyto)))
        else:
            avg_cyto_marker.append(0.0)
            std_cyto_marker.append(0.0)

        pcm_marker_mask = (marker_img > 0) & pcm_mask
        voxels_pcm = np.where(pcm_marker_mask)
        marker_pcm_sizes.append(voxels_pcm[0].size * r_xyz[0] * r_xyz[1] * r_xyz[2])
        if channel_idx is not None and voxels_pcm[0].size > 0:
            values_pcm = filtered_img[voxels_pcm[0], voxels_pcm[1], voxels_pcm[2], channel_idx]
            avg_pcm_marker.append(float(np.mean(values_pcm)))
            std_pcm_marker.append(float(np.std(values_pcm)))
        else:
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
):
    """Create one quantification row in the shared notebook output format."""
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


def prepare_stain_settings(
    im_in,
    stain_df,
    name_setup,
    use_setup=True,
    settings=None,
    napari_module=None,
    progress=None,
):
    """Load or interactively collect contrast/gamma settings for each channel."""
    stain_df = stain_df.reset_index(drop=False)
    stain_initial_df = stain_df.copy()
    stain_initial_df.set_index(["Condition", "Marker", "Laser"], inplace=True)
    stain_initial_df[["Cont_min", "Cont_max", "Gamma"]] = [0, 255, 1]
    stain_complete_df = stain_initial_df.copy()

    setup_path = f"{name_setup}_setup.csv"
    setup_exists = os.path.exists(setup_path)

    if use_setup and setup_exists:
        stain_setup_df = pd.read_csv(setup_path)
        stain_setup_df.set_index(["Condition", "Marker", "Laser"], inplace=True)
        for idx in _progress_iter(stain_complete_df.index, progress, desc="Step 07A - Load Setup Rows"):
            if idx in stain_setup_df.index:
                stain_complete_df.loc[idx] = stain_setup_df.loc[idx]
                stain_complete_df["Color"] = stain_initial_df["Color"]
            else:
                use_setup = False
                break

    if not use_setup or not setup_exists:
        if settings is None:
            from napari.settings import get_settings
            settings = get_settings()
        if napari_module is None:
            import napari as napari_module

        stain_complete_df = stain_initial_df.copy()
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
            im_channel = im_in[:, :, :, _]
            imin = float(im_channel.min())
            imax = float(im_channel.max())
            if imax <= imin:
                im_view = np.zeros_like(im_channel, dtype=np.uint8)
            else:
                im_view = ((im_channel - imin) / (imax - imin) * 255).clip(0, 255).astype(np.uint8)
            viewer.add_image(
                im_view,
                name=f"{idx[0]} ({idx[1]})",
                colormap=stain_initial_df.loc[idx]["Color"],
                blending="additive",
            )

        napari_module.run()
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

        if setup_exists:
            stain_setup_df = pd.read_csv(setup_path)
            stain_setup_df.set_index(["Condition", "Marker", "Laser"], inplace=True)
            for idx in _progress_iter(stain_complete_df.index, progress, desc="Step 07D - Write Setup Rows"):
                stain_setup_df.loc[idx] = stain_complete_df.loc[idx]
        else:
            stain_setup_df = stain_complete_df.copy()

        stain_csv_setup_df = stain_setup_df.reset_index().sort_values(by="Condition")
        stain_csv_setup_df = stain_csv_setup_df[["Condition", "Marker", "Laser", "Cont_min", "Cont_max", "Gamma"]]
        stain_csv_setup_df.to_csv(setup_path, index=False)

    stain_df = stain_df.set_index("Condition")
    stain_complete_df = stain_complete_df.reset_index().set_index("Condition")
    stain_complete_df = stain_complete_df.loc[stain_df.index]
    stain_complete_df = stain_complete_df[["Marker", "Laser", "Color", "Cont_min", "Cont_max", "Gamma"]]
    original_stain_complete_df = stain_complete_df.copy()
    return stain_df, stain_complete_df, original_stain_complete_df


def export_channel_histograms(
    im_final_stack,
    stain_complete_df,
    output_path,
    processing_params=None,
    progress=None,
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

    Returns
    -------
    output_path : str or Path
    """
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

    return output_path


def build_labels_dict(
    im_segmentation_stack,
    filtered_img,
    stain_complete_df,
    stain_df,
    r_xyz,
    zooms,
    multilabel=False,
    progress=None,
):
    """Build the compact per-marker quantification dictionary used by the notebook."""
    labels_dict = {}
    nuc_positions, nuc_sizes, cyto_positions, cyto_sizes = compute_nuclei_cytoplasm_stats(
        im_segmentation_stack,
        r_xyz,
        zooms,
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
        ) = compute_marker_stats_for_marker(c, im_segmentation_stack, filtered_img, r_xyz, zooms)

        labels_dict[marker] = _make_labels_record(
            condition,
            stain_complete_df.loc[condition, "Laser"],
            stain_complete_df.loc[condition, "Color"],
            len(shared_labels),
            shared_labels=sorted(shared_labels),
            nucleus_positions=(nuc_positions[i - 1] for i in shared_labels),
            cytoplasm_positions=(cyto_positions[i - 1] for i in shared_labels),
            nucleus_sizes=(nuc_sizes[i - 1] for i in shared_labels),
            cytoplasm_sizes=(cyto_sizes[i - 1] for i in shared_labels),
            marker_sizes=m_sizes,
            avg_marker=m_avg,
            std_marker=m_std,
            marker_cyto_sizes=m_cyto_sizes,
            avg_cyto_marker=m_cyto_avg,
            std_cyto_marker=m_cyto_std,
            marker_pcm_sizes=m_pcm_sizes,
            avg_pcm_marker=m_pcm_avg,
            std_pcm_marker=m_pcm_std,
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
    progress=None,
):
    """Build the full per-marker quantification dictionary for exports and meshes."""
    labels_full_dict = {}
    nuc_positions, nuc_sizes, cyto_positions, cyto_sizes = compute_nuclei_cytoplasm_stats(
        im_segmentation_stack,
        r_xyz,
        zooms,
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
        ) = compute_full_marker_stats_for_marker(c, im_final_stack, im_segmentation_stack, filtered_img, r_xyz, zooms)

        labels_full_dict[marker] = _make_labels_record(
            condition,
            stain_complete_df.iloc[c]["Laser"],
            stain_complete_df.iloc[c]["Color"],
            len(full_labels),
            shared_labels=sorted(full_labels),
            nucleus_positions=(nuc_positions[i - 1] for i in full_labels),
            cytoplasm_positions=(cyto_positions[i - 1] for i in full_labels),
            nucleus_sizes=(nuc_sizes[i - 1] for i in full_labels),
            cytoplasm_sizes=(cyto_sizes[i - 1] for i in full_labels),
            marker_sizes=m_full_sizes,
            avg_marker=m_full_avg,
            std_marker=m_full_std,
            marker_cyto_sizes=m_full_cyto_sizes,
            avg_cyto_marker=m_full_cyto_avg,
            std_cyto_marker=m_full_cyto_std,
            marker_pcm_sizes=m_full_pcm_sizes,
            avg_pcm_marker=m_full_pcm_avg,
            std_pcm_marker=m_full_pcm_std,
        )

    return labels_full_dict


def build_labels_df(
    im_segmentation_stack,
    filtered_img,
    stain_complete_df,
    stain_df,
    r_xyz,
    zooms,
    multilabel=False,
    progress=None,
):
    """Build the compact quantification dictionary, convert it to DataFrames, and return both."""
    labels_dict = build_labels_dict(
        im_segmentation_stack,
        filtered_img,
        stain_complete_df=stain_complete_df,
        stain_df=stain_df,
        r_xyz=r_xyz,
        zooms=zooms,
        multilabel=multilabel,
        progress=progress,
    )
    return labels_dict_to_dataframe(labels_dict, truncate=True, progress=progress)


def build_full_labels_df(
    im_segmentation_stack,
    im_final_stack,
    filtered_img,
    stain_complete_df,
    r_xyz,
    zooms,
    progress=None,
):
    """Build the full quantification dictionary and convert it to a DataFrame."""
    labels_full_dict = build_full_labels_dict(
        im_segmentation_stack,
        im_final_stack,
        filtered_img,
        stain_complete_df=stain_complete_df,
        r_xyz=r_xyz,
        zooms=zooms,
        progress=progress,
    )
    return labels_dict_to_dataframe(labels_full_dict)


def labels_dict_to_dataframe(labels_dict, truncate=False, progress=None):
    """Convert a quantification dictionary to the standard notebook DataFrame."""
    labels_df = pd.DataFrame.from_dict(labels_dict, orient="index", columns=LABELS_TABLE_COLUMNS)
    labels_df.index.name = "Combination"

    if not truncate:
        return labels_df

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
    ]
    for column in _progress_iter(truncate_columns, progress, desc="Step 23D - Truncate Display Columns"):
        truncated_df[column] = truncated_df[column].apply(lambda value: truncate_cell(value))

    return labels_df, truncated_df


def print_population_summary(labels_df, stain_complete_df, stain_df, progress=None):
    """Print the compact summary block used in the analysis section."""
    nuclei_rows = labels_df[labels_df["Condition"] == "NUCLEI"]
    total_cells = float(nuclei_rows.iloc[0]["Number"]) if not nuclei_rows.empty else float(labels_df.iloc[0]["Number"])

    # --- Channel overview ---
    print("CHANNELS:")
    for i, cond in enumerate(stain_complete_df.index):
        row = stain_complete_df.loc[cond]
        print(f"  [{i}] {cond:<20} {row['Marker']:<16} ({row['Laser']})")

    print("_" * 80)
    print("TOT CELLS =", int(total_cells))
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
    def _size_stats_line(sizes_tuple, label, unit="um\u00b3"):
        arr = np.array(sizes_tuple, dtype=float)
        if arr.size == 0:
            return
        print(
            f"{label}:  mean = {np.mean(arr):.2f} {unit}"
            f"  |  std = {np.std(arr):.2f} {unit}"
            f"  |  median = {np.median(arr):.2f} {unit}"
            f"  |  [min = {np.min(arr):.2f},  max = {np.max(arr):.2f}]"
        )

    if not nuclei_rows.empty:
        _size_stats_line(nuclei_rows.iloc[0]["Nuclei size [um3]"], "NUCLEI SIZE")
    cyto_rows = labels_df[labels_df["Condition"] == "CYTOPLASM"]
    if "CYTOPLASM" in stain_df.index and not cyto_rows.empty:
        _size_stats_line(cyto_rows.iloc[0]["Cytoplasm size [um3]"], "CYTOPLASM SIZE")

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

        # Nucleus compartment
        nuc_s = np.array(row["Nuclei size [um3]"], dtype=float)
        avg_i = np.array(row["Avg. marker intensity"], dtype=float)
        std_i = np.array(row["STD marker intensity"], dtype=float)
        if nuc_s.size > 0:
            nuc_line = f"   Nucleus size:        {np.mean(nuc_s):.2f} ± {np.std(nuc_s):.2f} um\u00b3"
            if avg_i.size > 0:
                nuc_line += f"   |  intensity: {np.mean(avg_i):.2f} ± {np.mean(std_i):.2f} a.u."
            print(nuc_line)
        elif avg_i.size > 0:
            print(f"   Nucleus intensity:    {np.mean(avg_i):.2f} ± {np.mean(std_i):.2f} a.u.")

        # Cytoplasm compartment
        if "CYTOPLASM" in stain_df.index:
            cyto_s = np.array(row["Cytoplasm size [um3]"], dtype=float)
            cyto_avg = np.array(row["Avg. marker intensity cytoplasm"], dtype=float)
            cyto_std = np.array(row["STD marker intensity cytoplasm"], dtype=float)
            if cyto_s.size > 0:
                cyto_line = f"   Cytoplasm size:      {np.mean(cyto_s):.2f} ± {np.std(cyto_s):.2f} um\u00b3"
                if cyto_avg.size > 0:
                    cyto_line += f"   |  intensity: {np.mean(cyto_avg):.2f} ± {np.mean(cyto_std):.2f} a.u."
                print(cyto_line)
            elif cyto_avg.size > 0:
                print(f"   Cytoplasm intensity:  {np.mean(cyto_avg):.2f} ± {np.mean(cyto_std):.2f} a.u.")

        # Marker/aggregate size and intensity
        msize = np.array(row["Marker size [um3]"], dtype=float)
        avg_mi = np.array(row["Avg. marker intensity"], dtype=float)
        std_mi = np.array(row["STD marker intensity"], dtype=float)
        if msize.size > 0:
            marker_line = f"   Marker size:          {np.mean(msize):.2f} ± {np.std(msize):.2f} um\u00b3"
            if avg_mi.size > 0:
                marker_line += f"   |  intensity: {np.mean(avg_mi):.2f} ± {np.mean(std_mi):.2f} a.u."
            print(marker_line)

    print("_" * 80)


def build_histogram_report(
    im_segmentation_stack,
    im_final_stack,
    filtered_img,
    stain_df,
    stain_complete_df,
    input_file,
    pad=20,
    thumb_size=None,
    progress=None,
):
    """Collect histogram data, plot KDEs, and generate the per-nucleus PDF report."""
    from reportlab.lib.units import inch
    from pathlib import Path as _Path

    if thumb_size is None:
        thumb_size = (2.0 * inch, 2.0 * inch)

    hist_data, intensity_ranges = collect_histogram_data(
        im_segmentation_stack,
        filtered_img,
        stain_df=stain_df,
        stain_complete_df=stain_complete_df,
        progress=progress,
    )

    fig, axes, x_grid = plot_nucleus_kdes(
        hist_data,
        stain_complete_df=stain_complete_df,
        progress=progress,
    )

    set_notebook_context(
        seg_stack=im_segmentation_stack,
        filtered_img=filtered_img,
        hist_data=hist_data,
        x_grid=x_grid,
        stain_complete_df=stain_complete_df,
        stain_df=stain_df,
    )

    output_pdf = str(_Path(input_file).stem) + "_nuclei_marker.pdf"
    create_row_pdf(
        output_pdf=output_pdf,
        pad=pad,
        thumb_size=thumb_size,
    )

    return hist_data, intensity_ranges, fig, axes, x_grid


def collect_histogram_data(im_segmentation_stack, filtered_img, stain_df, stain_complete_df, progress=None):
    """Collect per-nucleus intensity values for all non-nuclear markers."""
    hist_data = {}
    intensity_ranges = {}

    for c in _progress_iter(range(filtered_img.shape[3]), progress, desc="Step 26A - Collect Histogram Data"):
        if stain_df.index[c] in ("NUCLEI", "CYTOPLASM"):
            continue

        condition = stain_complete_df.index[c]
        marker_img = filtered_img[:, :, :, c]
        intensity_ranges[condition] = (float(marker_img.min()), float(marker_img.max()))
        max_n = int(np.max(im_segmentation_stack["Nuclei"]))

        for nucleus_id in _progress_iter(range(1, max_n + 1), progress, desc=f"Step 26B - {condition} Nuclei"):
            hist_data.setdefault(nucleus_id, {})
            hist_data[nucleus_id].setdefault(condition, [])

            nuc_mask = im_segmentation_stack["Nuclei"] == nucleus_id
            cyto_mask = im_segmentation_stack["Cytoplasm"] == nucleus_id
            pcm_mask = im_segmentation_stack["PCM"] == nucleus_id
            mask_marker = (marker_img > 0) & ((nuc_mask + cyto_mask + pcm_mask) > 0)

            if np.any(mask_marker):
                values = marker_img[mask_marker]
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


def plot_spatial_distributions(labels_df, stain_complete_df, stain_df, im_in, r_X, r_Y, r_Z, zoom_factors, progress=None):
    """Plot X/Y/Z spatial distributions for each population."""
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

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
        if np.size(marker) == 1:
            axs[0].plot(xbin_centers, xcount, label=str(condition), color=color)
            axs[1].plot(ybin_centers, ycount, label=str(condition), color=color)
            axs[2].plot(zbin_centers, zcount, label=str(condition), color=color)
        elif np.size(marker) != 1:
            linestyle = (0, (2, max(np.size(marker) - 1, 1)))
            axs[0].plot(xbin_centers, xcount, label=str(condition), linestyle=linestyle, color=color)
            axs[1].plot(ybin_centers, ycount, label=str(condition), linestyle=linestyle, color=color)
            axs[2].plot(zbin_centers, zcount, label=str(condition), linestyle=linestyle, color=color)

    axs[0].set_title("NUCLEI X DISTRIBUTION")
    axs[0].set_xlabel("[μm]")
    axs[0].legend(loc="upper right")
    axs[0].set_facecolor("black")
    axs[1].set_title("NUCLEI Y DISTRIBUTION")
    axs[1].set_xlabel("[μm]")
    axs[1].legend(loc="upper right")
    axs[1].set_facecolor("black")
    axs[2].set_title("NUCLEI Z DISTRIBUTION")
    axs[2].set_xlabel("[μm]")
    axs[2].legend(loc="upper right")
    axs[2].set_facecolor("black")
    return fig, axs


def plot_size_distributions(labels_df, stain_complete_df, stain_df, progress=None):
    """Plot nuclei and cytoplasm size histograms for each population."""
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

    axs[0].set_title("NUCLEI SIZE DISTRIBUTION")
    axs[0].set_xlabel("[μm3]")
    axs[0].legend(loc="upper right")
    axs[1].set_title("CELL SIZE DISTRIBUTION")
    axs[1].set_xlabel("[μm3]")
    axs[1].legend(loc="upper right")
    return fig, axs


def export_quantification_to_excel(output_path, original_stain_complete_df, labels_full_df, progress=None):
    """Write the main Excel report with staining, nuclei, cytoplasm, and recap sheets."""
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        original_stain_complete_df.to_excel(writer, sheet_name="Staining", index=True)

        nuclei_rows = labels_full_df[labels_full_df["Condition"] == "NUCLEI"]
        if not nuclei_rows.empty:
            nuclei_row = nuclei_rows.iloc[0]
            nuclei_dict = {}
            for k in _progress_iter(range(1, int(nuclei_row["Number"])), progress, desc="Step 32A - Write Nuclei Sheet"):
                position = nuclei_row["Mean nuclei positions [um]"][k - 1]
                nuclei_dict[k] = [position[0], position[1], position[2], nuclei_row["Nuclei size [um3]"][k - 1]]
            cell_df = pd.DataFrame.from_dict(
                nuclei_dict,
                orient="index",
                columns=["X position [um]", "Y position [um]", "Z position [um]", "Nuclei size [um3]"],
            )
            cell_df.to_excel(writer, sheet_name="NUCLEI", index=True)

        cyto_rows = labels_full_df[labels_full_df["Condition"] == "CYTOPLASM"]
        if not cyto_rows.empty:
            cytoplasm_row = cyto_rows.iloc[0]
            xlsx_dict = {}
            columns = ["X position [um]", "Y position [um]", "Z position [um]", "Cytoplasm size [um3]"]

            single_marker_rows = []
            for idx, marker in _progress_iter(
                enumerate(labels_full_df.index),
                progress,
                desc="Step 32B - Prepare Cytoplasm Columns",
                total=len(labels_full_df.index),
            ):
                condition = labels_full_df.iloc[idx]["Condition"]
                if condition not in ["NUCLEI", "CYTOPLASM"] and np.size(condition) == 1:
                    single_marker_rows.append((idx, marker))
                    columns.extend(
                        [
                            f"{marker} marker size [um3]",
                            f"{marker} marker size cytoplasm [um3]",
                            f"{marker} marker size PCM [um3]",
                            f"{marker} intensity [-]",
                            f"{marker} STD",
                            f"{marker} intensity cytoplasm [-]",
                            f"{marker} STD",
                            f"{marker} intensity PCM [-]",
                            f"{marker} STD",
                        ]
                    )

            for k in _progress_iter(range(1, int(cytoplasm_row["Number"])), progress, desc="Step 32C - Write Cytoplasm Sheet"):
                position = cytoplasm_row["Mean cytoplasm positions [um]"][k - 1]
                row = [position[0], position[1], position[2], cytoplasm_row["Cytoplasm size [um3]"][k - 1]]
                for idx, marker in single_marker_rows:
                    shared = labels_full_df.iloc[idx]["Shared labels"]
                    if k in shared:
                        shared_idx = list(shared).index(k)
                        row.extend(
                            [
                                labels_full_df.iloc[idx]["Marker size [um3]"][shared_idx],
                                labels_full_df.iloc[idx]["Marker size cytoplasm [um3]"][shared_idx],
                                labels_full_df.iloc[idx]["Marker size PCM [um3]"][shared_idx],
                                labels_full_df.iloc[idx]["Avg. marker intensity"][shared_idx],
                                labels_full_df.iloc[idx]["STD marker intensity"][shared_idx],
                                labels_full_df.iloc[idx]["Avg. marker intensity cytoplasm"][shared_idx],
                                labels_full_df.iloc[idx]["STD marker intensity cytoplasm"][shared_idx],
                                labels_full_df.iloc[idx]["Avg. marker intensity PCM"][shared_idx],
                                labels_full_df.iloc[idx]["STD marker intensity PCM"][shared_idx],
                            ]
                        )
                    else:
                        row.extend([" "] * 9)
                xlsx_dict[k] = row

            cell_df = pd.DataFrame.from_dict(xlsx_dict, orient="index", columns=columns)
            cell_df.to_excel(writer, sheet_name="CYTOPLASM", index=True)

        resume_df = labels_full_df.drop(
            columns=[
                "Shared labels",
                "Mean nuclei positions [um]",
                "Mean cytoplasm positions [um]",
                "Nuclei size [um3]",
                "Cytoplasm size [um3]",
                "Marker size [um3]",
                "Avg. marker intensity",
                "Marker size cytoplasm [um3]",
                "Avg. marker intensity cytoplasm",
                "Marker size PCM [um3]",
                "Avg. marker intensity PCM",
            ]
        ).copy()
        resume_df["Laser"] = [labels_full_df.iloc[t]["Laser"] if np.size(labels_full_df.iloc[t]["Condition"]) == 1 else "" for t in range(len(labels_full_df))]
        resume_df["Color"] = [labels_full_df.iloc[t]["Color"] if np.size(labels_full_df.iloc[t]["Condition"]) == 1 else "" for t in range(len(labels_full_df))]

        nuclei_rows = labels_full_df[labels_full_df["Condition"] == "NUCLEI"]
        total_cells = float(nuclei_rows.iloc[0]["Number"]) if not nuclei_rows.empty else float(labels_full_df.iloc[0]["Number"])
        resume_df["%"] = [
            100.0 * labels_full_df.iloc[t]["Number"] / total_cells if labels_full_df.iloc[t]["Condition"] != "NUCLEI" else ""
            for t in range(len(labels_full_df))
        ]
        resume_df["Mean nuclei size [um3]"] = [np.mean(values) if len(values) > 0 else 0.0 for values in labels_full_df["Nuclei size [um3]"]]
        resume_df["Mean cytoplasm size [um3]"] = [np.mean(values) if len(values) > 0 else 0.0 for values in labels_full_df["Cytoplasm size [um3]"]]
        resume_df["Mean marker size [um3]"] = [
            np.mean(values) if labels_full_df.iloc[t]["Condition"] not in ["NUCLEI", "CYTOPLASM"] and np.size(labels_full_df.iloc[t]["Condition"]) == 1 and len(values) > 0 else ""
            for t, values in enumerate(labels_full_df["Marker size [um3]"])
        ]
        resume_df.to_excel(writer, sheet_name="RECAP", index=True)

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


def create_row_pdf(output_pdf="nuclei_row_pages.pdf", pad=20, thumb_size=None):
    """Create the nuclei report PDF using notebook context previously registered."""
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

    os.makedirs("crop_png", exist_ok=True)
    os.makedirs("merged_png", exist_ok=True)
    os.makedirs("density_png", exist_ok=True)

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
            fname = f"crop_png/n{nucleus_id}_{condition}.png"
            color = stain_complete_df.loc[condition, "Color"] if (
                condition in stain_complete_df.index and "Color" in stain_complete_df.columns
            ) else "gray"
            if color == "WHITE":
                color = "GRAY"
            if condition in stain_complete_df.index and "Cont_min" in stain_complete_df.columns:
                try:
                    clim = (
                        stain_complete_df.loc[condition, "Cont_min"],
                        stain_complete_df.loc[condition, "Cont_max"],
                    )
                    gamma_value = stain_complete_df.loc[condition, "Gamma"] if "Gamma" in stain_complete_df.columns else 1.0
                    save_raw_png(arr, fname, contrast_limits=clim, gamma=gamma_value, color=color)
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
            stain_table=stain_complete_df,
            nucleus_color=nucleus_color,
            cytoplasm_color="green",
            pcm_color="magenta",
            pad=pad,
            out_dir="merged_png",
        )
        if merged_png is None:
            merged_png = f"merged_png/n{nucleus_id}_merged_placeholder.png"
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
        ax.legend(legend_handles, all_conditions, loc="upper right", framealpha=0.9)
        plt.tight_layout()
        density_png = f"density_png/n{nucleus_id}_density.png"
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
        label_row = [Paragraph(cond, label_style) for cond in marker_conditions] + [Paragraph("Merged", label_style)]
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
    from scipy import ndimage as ndi

    out = np.asarray(label_img).copy()
    current_labels = np.unique(out)
    current_labels = current_labels[current_labels > 0]
    next_label = int(out.max()) + 1
    total_splits = 0

    for label_id in current_labels:
        label_mask = out == label_id
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
                out[z][slice_components[z] == c] = new_label

    return out, total_splits


def _split_z_watershed(label_img, min_fragment_vox):
    """Watershed-reinforced z-split: any z with multiple significant 2D
    components seeds a local watershed that carves the full 3D label."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    out = np.asarray(label_img).copy()
    next_label = int(out.max()) + 1
    total_splits = 0
    min_component_area_2d = max(4, min_fragment_vox // 2)

    for _pass in range(3):
        current_labels = np.unique(out)
        current_labels = current_labels[current_labels > 0]
        pass_splits = 0

        for label_id in current_labels:
            label_mask = out == label_id
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
                out[sub_labels == sid] = next_label
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
    from scipy import ndimage as ndi

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

    def _is_z_consistent_pair(labels, a, b):
        mask_a = labels == a
        mask_b = labels == b
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
        for a, b in pairs:
            if _find(a) == _find(b):
                continue
            if _is_z_consistent_pair(out, a, b):
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


def _filter_labels_by_roundness_xy(label_img, min_roundness):
    """Remove 3D labels whose largest XY slice is less circular than threshold."""
    threshold = max(0.0, float(min_roundness))
    if threshold <= 0.0:
        return label_img, 0

    out = np.asarray(label_img).copy()
    max_label = int(np.max(out))
    removed = 0

    for label_id in range(1, max_label + 1):
        mask = out == label_id
        if not np.any(mask):
            continue

        z_counts = np.count_nonzero(mask, axis=(1, 2))
        z_best = int(np.argmax(z_counts))
        slice_mask = mask[z_best]
        area = int(np.count_nonzero(slice_mask))

        if area <= 2:
            out[mask] = 0
            removed += 1
            continue

        contours, _ = cv2.findContours(
            slice_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
        roundness = 0.0 if perimeter <= 0.0 else float((4.0 * np.pi * area) / (perimeter ** 2))

        if roundness < threshold:
            out[mask] = 0
            removed += 1

    return out, removed

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
):
    """
    Segment nuclei using watershed with multi-scale erosion and EDT peak fallback.
    
    Implements a robust watershed-based segmentation algorithm that:
    1. Handles Z-anisotropy by adjusting erosion depths
    2. Uses multi-scale erosion to create multiple seed points for dumbbell/multi-lobed nuclei
    3. Falls back to EDT peak detection when erosion fails to split nuclei
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
    
    Returns
    -------
    im_out : ndarray, shape (Z, Y, X), dtype int32
        Labeled nucleus segmentation (label value per voxel). Label 0 = background.
    
    debug_info : dict
        Debugging information with keys:
        - 'z_anisotropy': Computed Z anisotropy factor
        - 'z_split_weight': Effective Z-weighting applied to erosion
        - 'scales_with_seeds': Number of erosion scales that produced markers
        - 'added_peak_seed_count': Number of seeds added via EDT peak detection
        - 'added_seed_count': Number of fallback seeds added
        - 'erosion_triplets': List of (z, y, x) erosion radii attempted
        - 'boundary_components': Number of boundary-touching components
        - 'restored_isolated_count': Voxels restored after cleanup
        - 'dmin', 'dmax', 'n_scales': Diameter scaling parameters
        - 'min_seed_vox': Minimum seed size threshold in voxels
    
    Notes
    -----
    - Modifies parameter r_zZ by computing effective_r_zZ for Z-anisotropy handling
    - Uses peak_local_max from skimage.feature for EDT-based fallback
    - Applies watershed transform with negative distance map to favor markers
    - Removes small island labels but preserves those touching image boundaries
    
    Example
    -------
    >>> binary = im_threshold > 0  # Binary foreground mask
    >>> nuclei, debug = segment_nuclei_watershed(
    ...     binary,
    ...     r_zX=0.1, r_zY=0.1, r_zZ=0.2,
    ...     nuclei_diameter=30.0,
    ...     nuclei_bridge_shrink_factor=0.28,
    ... )
    >>> print(f"Nuclei found: {int(nuclei.max())}, Peak-based seeds: {debug['added_peak_seed_count']}")
    """
    from skimage import morphology
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed, relabel_sequential

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

    boundary_components = set()
    cc_labels_for_boundary, _ = ndi.label(binary_mask)
    for cc_id in np.unique(cc_labels_for_boundary):
        if cc_id == 0:
            continue
        cc_mask = cc_labels_for_boundary == cc_id
        if np.any(cc_mask & boundary_mask):
            boundary_components.add(int(cc_id))

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

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    next_marker = 0
    erosion_triplets = []
    scales_with_seeds = 0

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

        valid_ids = []
        for seed_id in range(1, num_i + 1):
            seed_mask = mk_i == seed_id
            seed_size = int(np.sum(seed_mask))

            overlaps_boundary = False
            overlapping_cc_ids = np.unique(cc_labels_for_boundary[seed_mask])
            overlapping_cc_ids = overlapping_cc_ids[overlapping_cc_ids > 0]
            if len(overlapping_cc_ids) > 0:
                overlaps_boundary = any(
                    int(cc_id) in boundary_components for cc_id in overlapping_cc_ids
                )

            threshold = min_seed_vox_boundary if overlaps_boundary else min_seed_vox
            if seed_size >= threshold:
                valid_ids.append(seed_id)

        if len(valid_ids) == 0:
            continue

        scales_with_seeds += 1

        for seed_id in valid_ids:
            seed_mask = mk_i == seed_id
            if np.any(markers[seed_mask]):
                continue
            next_marker += 1
            markers[seed_mask] = next_marker

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

            for seed_id in range(1, num_r + 1):
                seed_mask = mk_r == seed_id
                seed_size = int(np.sum(seed_mask))

                overlaps_boundary = False
                overlapping_cc_ids = np.unique(cc_labels_for_boundary[seed_mask])
                overlapping_cc_ids = overlapping_cc_ids[overlapping_cc_ids > 0]
                if len(overlapping_cc_ids) > 0:
                    overlaps_boundary = any(
                        int(cc_id) in boundary_components for cc_id in overlapping_cc_ids
                    )

                threshold = min_seed_vox_boundary if overlaps_boundary else min_seed_vox
                if seed_size < threshold:
                    continue
                if np.any(markers[seed_mask]):
                    continue
                next_marker += 1
                markers[seed_mask] = next_marker

        num = int(markers.max())

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

    for cc_id in range(1, num_cc + 1):
        cc_mask = cc_labels == cc_id
        cc_marker_ids = np.unique(markers[cc_mask])
        cc_marker_ids = cc_marker_ids[cc_marker_ids > 0]
        local_distance = np.where(cc_mask, distance, 0.0)

        # Recovery for dumbbell-like nuclei: if erosion still leaves only
        # one seed, try to split the component using multiple EDT peaks.
        # Uses 2D XY projection to prevent over-splitting across Z.
        if cc_marker_ids.size <= 1 and np.any(cc_mask):
            peak_coords_3d = detect_peaks_xy_with_best_z(
                local_distance,
                peak_min_distance_xy,
                peak_threshold_fraction,
            )

            if len(peak_coords_3d) >= 2:
                if cc_marker_ids.size == 1:
                    markers[(markers == int(cc_marker_ids[0])) & cc_mask] = 0
                for peak_coord in peak_coords_3d:
                    if markers[peak_coord] != 0:
                        continue
                    next_marker += 1
                    markers[peak_coord] = next_marker
                    added_peak_seed_count += 1
                cc_marker_ids = np.unique(markers[cc_mask])
                cc_marker_ids = cc_marker_ids[cc_marker_ids > 0]

        if cc_marker_ids.size == 0:
            seed_pos = np.unravel_index(np.argmax(local_distance), local_distance.shape)
            next_marker += 1
            markers[seed_pos] = next_marker
            added_seed_count += 1

    im_out = watershed(-distance, markers, mask=binary_mask)

    preserve_labels = set()
    for cc_id in range(1, num_cc + 1):
        cc_vals = np.unique(im_out[cc_labels == cc_id])
        cc_vals = cc_vals[cc_vals > 0]
        if cc_vals.size == 1:
            preserve_labels.add(int(cc_vals[0]))
        if cc_id in boundary_components:
            for val in cc_vals:
                preserve_labels.add(int(val))

    full_nucleus_vol_um3 = (4.0 * np.pi * ((nuclei_diameter / 2.0) ** 3)) / 3.0
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

    im_out, removed_by_roundness = _filter_labels_by_roundness_xy(
        im_out,
        nuclei_min_roundness,
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
        'added_seed_count': added_seed_count,
        'erosion_triplets': erosion_triplets,
        'boundary_components': boundary_components,
        'restored_isolated_count': restored_isolated_count,
        'dmin': dmin,
        'dmax': dmax,
        'n_scales': n_scales,
        'min_seed_vox': min_seed_vox,
        'min_cell_vox': min_cell_vox,
        'nuclei_min_roundness': float(max(0.0, nuclei_min_roundness)),
        'removed_by_roundness': removed_by_roundness,
        'z_consistency_splits': z_consistency_splits,
        'z_consistency_merges': z_consistency_merges,
    }

    return im_out, debug_info

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


def apply_median_denoise(image_stack):
    """Apply median filter to each channel independently."""
    from scipy import ndimage as ndi
    from skimage import filters
    return apply_per_channel_filter(image_stack, lambda ch: filters.median(ch))


def apply_gaussian_smoothing(image_stack, sigma=0.5):
    """Apply Gaussian filter to each channel independently."""
    from skimage import filters
    return apply_per_channel_filter(
        image_stack, 
        lambda ch: filters.gaussian(ch, sigma, preserve_range=True)
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


def resample_to_isotropic(image_stack, zoom_factors, meta=None):
    """
    Resample image stack to isotropic voxel spacing.
    
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
        resampled = scipy_zoom(image_stack[:, :, :, c], zoom=zoom_factors, order=1)
        resampled = resampled - np.min(resampled)
        normalized = (resampled - resampled.min()) / (resampled.max() - resampled.min() + 1e-8) * 255
        im_out[:, :, :, c] = np.clip(normalized, 0, 255).astype('uint8')
    
    if meta is not None:
        r_zX = meta.physical_pixel_sizes.X / zoom_factors[2]
        r_zY = meta.physical_pixel_sizes.Y / zoom_factors[1]
        r_zZ = meta.physical_pixel_sizes.Z / zoom_factors[0]
        return im_out, r_zX, r_zY, r_zZ
    
    return im_out


def extract_roi_from_metadata(meta, roi_coords, big_image=True):
    """
    Extract a region of interest from an ND2 file.
    
    Parameters
    ----------
    meta : AICSImage
        Metadata object from bioio.AICSImage.
    
    roi_coords : list of 6 ints
        [x0, x1, y0, y1, z0, z1] coordinates. Use 0 to keep full range.
    
    big_image : bool
        If True, uses dask for lazy loading. If False, loads full image.
    
    Returns
    -------
    image : ndarray, shape (Z, Y, X, C)
        Extracted image in ZYX order.
    
    roi_used : list
        Actual ROI coordinates used (with 0s replaced by full ranges).
    """
    from aicsimageio import AICSImage
    
    x0, x1, y0, y1, z0, z1 = roi_coords
    
    # Replace 0s with full range
    if x1 == 0:
        x1 = meta.shape[4]
    if y1 == 0:
        y1 = meta.shape[3]
    if z1 == 0:
        z1 = meta.shape[2]
    
    if big_image:
        lazy = meta.get_image_dask_data("ZYXC")
        sub = lazy[z0:z1, y0:y1, x0:x1, :]
        image = sub.compute()
    else:
        image = meta.get_image_data("ZYXC", T=0)
    
    return image, [x0, x1, y0, y1, z0, z1]


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


def prepare_image_stack(img, meta, ROI, big_image, nuclei_diameter, cell_diameter):
    """Compute geometry parameters and build the initial image stack.

    Derives nuclei/cell radii and volumes from diameter inputs, applies ROI
    cropping for the non-big-image case, casts to float32, and fixes axes
    order so the result is always (Z, Y, X, C).

    Parameters
    ----------
    img : ndarray
        Image array already loaded into memory (shape Z, Y, X, C or similar).
    meta : AICSImage
        Opened AICSImage object.
    ROI : list of 6 ints
        [x0, x1, y0, y1, z0, z1]. Zero values are replaced by the image extent.
    big_image : bool
        If True the full img is used as-is; if False a sub-region is read
        from meta and cropped according to ROI.
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

    x0, x1, y0, y1, z0, z1 = ROI
    if x1 == 0:
        x1 = img.shape[0]
    if y1 == 0:
        y1 = img.shape[1]
    if z1 == 0:
        z1 = img.shape[2]

    if big_image:
        im_original = img.astype('float32')
        im_original_ROI = im_original.copy()
    else:
        im_original = meta.get_image_data("ZYXC", S=0, T=0).astype('float32')
        im_original_ROI = im_original[z0:z1, y0:y1, x0:x1, :]

    # Fix axes order: ensure (Z, Y, X, C), drop singleton T axis if present
    orig = im_original_ROI
    if len(orig.shape) == 5 and orig.shape[4] == 1:
        orig = orig[..., 0]
    if len(orig.shape) == 4 and orig.shape[-1] > 50:
        chan_axis = next((i for i, s in enumerate(orig.shape) if s < 50), None)
        if chan_axis is not None and chan_axis != 3:
            orig = np.moveaxis(orig, chan_axis, -1)

    print(f"Image stack ready — shape: {orig.shape}")
    return {'Original image': orig}, nuclei_radius, cell_radius, nuclei_volume, cell_volume


def build_stain_dataframe(stain_dict: dict, file_meta: dict) -> "pd.DataFrame":
    """Build and sort the staining metadata DataFrame.

    Normalises all string keys/values in *stain_dict* to uppercase, constructs
    a DataFrame with columns ``['Marker', 'Laser', 'Color']``, then sorts rows
    to match the channel order reported by *file_meta*.

    Parameters
    ----------
    stain_dict : dict
        User-defined mapping of condition name to [marker, laser/channel, color].
    file_meta : dict
        Output of ``read_file_metadata``; must contain a ``'channels'`` list.

    Returns
    -------
    stain_df : pd.DataFrame
        Sorted staining table indexed by condition name.
    """
    norm = {
        k.upper(): [item.upper() if isinstance(item, str) else item for item in v]
        for k, v in stain_dict.items()
    }
    stain_df = pd.DataFrame.from_dict(norm, orient='index', columns=['Marker', 'Laser', 'Color'])
    stain_df.index.name = 'Condition'

    laser_order = file_meta.get("channels") or []
    order_map = {name.strip().upper(): i for i, name in enumerate(laser_order)}
    stain_df['order'] = stain_df['Laser'].map(order_map)
    stain_df = stain_df.sort_values('order').drop(columns='order')

    if 'NUCLEI' not in stain_df.index:
        print('[build_stain_dataframe] Warning: no NUCLEI condition found!')

    return stain_df


def view_original_channels(im_final_stack: dict, stain_df: "pd.DataFrame",
                            napari_module, progress=None) -> object:
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
    progress : callable, optional
        A ``tqdm``-compatible progress wrapper (e.g. ``tqdm``).

    Returns
    -------
    viewer : napari.Viewer
    """
    if progress is None:
        def progress(x, **kw): return x

    im_in = im_final_stack['Original image'].copy()
    _close_all_napari_viewers(napari_module)
    viewer = napari_module.Viewer(title="Original Image Channels")

    for c, c_name in progress(
        enumerate(stain_df['Marker']),
        total=len(stain_df['Marker']),
        desc='Step 06 - Visualize Channels',
        leave=False,
    ):
        im_channel = im_in[:, :, :, c]
        ch_min, ch_max = im_channel.min(), im_channel.max()
        im_8b = ((im_channel - ch_min) / (ch_max - ch_min + 1e-8) * 255).clip(0, 255).astype('uint8')
        viewer.add_image(
            im_8b,
            name=f"{stain_df.index[c]} ({c_name})",
            colormap=stain_df['Color'].iloc[c],
            blending='additive',
        )

    viewer.scale_bar.visible = True
    for layer in viewer.layers:
        layer.units = ('um', 'um', 'um')
    return viewer


def apply_threshold_per_channel(
    image_stack,
    stain_complete_df,
    nuclei_diameter,
    cell_diameter,
    r_zxyz,
    threshold_method='otsu',
    progress=None,
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
    progress : callable, optional
        Progress wrapper (e.g. tqdm).

    Returns
    -------
    im_out : ndarray, shape (Z, Y, X, C)
        Binary thresholded image stack (same dtype as input).
    """
    import SimpleITK as sitk
    from skimage.filters import threshold_sauvola

    _valid_methods = ('otsu', 'median', 'huang')
    if threshold_method not in _valid_methods:
        raise ValueError(f"threshold_method must be one of {_valid_methods}, got '{threshold_method}'")

    if progress is None:
        def progress(x, **kw): return x

    r_zX, r_zY, _ = r_zxyz
    im_out = image_stack.copy()
    nuclei_size = int(nuclei_diameter / (np.mean([r_zX, r_zY])))
    cell_size = int(cell_diameter / (np.mean([r_zX, r_zY])))

    for c in progress(range(image_stack.shape[3]), desc='Step 15 - Threshold Channels'):
        img = sitk.GetImageFromArray(image_stack[:, :, :, c])

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
            arr_tmp = sitk.GetArrayFromImage(img)
            non_zero_tmp = arr_tmp[arr_tmp > 0]
            global_thr_value = float(np.median(non_zero_tmp)) if non_zero_tmp.size > 0 else 0.0
        elif threshold_method == 'huang':
            th_filter = sitk.HuangThresholdImageFilter()
            th_filter.Execute(stretched)
            th_filter.Execute(img)
            global_thr_value = th_filter.GetThreshold()

        if stain_complete_df.index[c] == "NUCLEI":
            window_size = 1 * nuclei_size
        else:
            window_size = 4 * cell_size + 1

        if int(window_size) % 2 == 0:
            window_size += 1

        arr = sitk.GetArrayFromImage(img)

        # Sauvola threshold map
        sauvola_value = threshold_sauvola(arr, window_size=int(window_size))

        # Global statistical background, excluding zeros
        non_zero = arr[arr > 0]
        if non_zero.size > 0:
            hist, bins = np.histogram(non_zero, bins=256, range=(0, non_zero.max()))
            mode_bin = bins[np.argmax(hist)]
            print(mode_bin)
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
        bg_std = bg_vals.std() + 1e-6  # noqa: F841

        bg_mean_z = arr.mean()
        bg_std_z = arr.std() + 1e-6
        statistical_thr = bg_mean_z + 3.0 * bg_std_z

        # Clip Sauvola to avoid over-thresholding large/bright cells
        max_sauvola = bg_mean_z + 2.0 * bg_std_z
        sauvola_clipped = np.minimum(sauvola_value, max_sauvola)

        # Combined threshold map
        final_thr = (
            0.60 * sauvola_clipped +
            0.25 * statistical_thr +
            0.15 * global_thr_value
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
                print(
                    f"  ch {c}: oversaturation detected "
                    f"({saturated_fraction * 100:.1f}% near-max), "
                    f"threshold floor raised to P{sat_pct:.0f}={saturation_floor:.1f}"
                )

        # Gain-based primary mask and rescue
        gain = arr / (bg_mean + 1e-6)
        primary = (arr > final_thr) & (gain > gain_ass)
        rescue = (gain > (gain_ass + 3.0)) & (arr > statistical_thr)
        arrayseg = primary | rescue

        if stain_complete_df.index[c] != 'NUCLEI':
            min_size = np.ceil(0.8 * np.pi * ((nuclei_size / 2) ** 2))
        else:
            min_size = np.ceil(0.4 * np.pi * ((nuclei_size / 2) ** 2))

        im_out[:, :, :, c] = remove_small_islands(arrayseg, min_size)

    return im_out


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

    print(
        f"Cellpose: {int(labels.max())} nuclei segmented "
        f"(model={model_type}, diameter_px={diameter_px:.1f}, "
        f"anisotropy={anisotropy:.2f})"
    )
    return labels.astype(np.int32)


def segment_nuclei(
    im_final_stack,
    stain_df,
    stain_complete_df,
    nuclei_split_config,
    r_zxyz,
    nuclei_diameter,
    trig_stardist=False,
    trig_cellpose=False,
    progress=None,
):
    """
    Segment nuclei from the image using watershed, StarDist, or Cellpose.

    Parameters
    ----------
    im_final_stack : dict
        Must contain 'Threshold image' and 'Filtered image' arrays (Z, Y, X, C).
    stain_df, stain_complete_df : DataFrame
        Staining metadata.
    nuclei_split_config : dict
        Configuration dict returned by get_nuclei_split_config().
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes in micrometers.
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    trig_stardist : bool
        If True, use StarDist.
    trig_cellpose : bool
        If True, use Cellpose 3D (takes priority over StarDist and watershed).
    progress : callable, optional
        Progress wrapper (e.g. tqdm).

    Returns
    -------
    im_segmentation_stack : dict
        Dict with key 'Nuclei' mapping to the integer label array (Z, Y, X).
    """
    from skimage.measure import label as skimage_label
    from skimage import morphology

    if progress is None:
        def progress(x, **kw): return x

    r_zX, r_zY, r_zZ = r_zxyz
    im_segmentation_stack = {}

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
            im_segmentation_stack['Nuclei'] = im_out
            im_segmentation_stack['Cytoplasm'] = np.zeros_like(im_out)
            im_segmentation_stack['PCM'] = np.zeros_like(im_out)
            return im_segmentation_stack

        im_in = im_final_stack['Threshold image'].copy()
        im_thresh = np.zeros(im_in.shape[:3], dtype=bool)
        for c in range(im_in.shape[3]):
            im_thresh = im_thresh | (im_in[:, :, :, c] > 0)

        split_cfg = dict(nuclei_split_config)
        im_out, debug_info = segment_nuclei_watershed(
            binary_mask=im_thresh,
            r_zX=r_zX,
            r_zY=r_zY,
            r_zZ=r_zZ,
            nuclei_diameter=nuclei_diameter,
            **split_cfg,
        )

        erosion_triplets = debug_info['erosion_triplets']
        er_z = [e[0] for e in erosion_triplets] if erosion_triplets else [1]
        er_y = [e[1] for e in erosion_triplets] if erosion_triplets else [1]
        er_x = [e[2] for e in erosion_triplets] if erosion_triplets else [1]

        print(
            f"LD mode: {int(im_out.max())} cells segmented "
            f"(shrink_factor={split_cfg['nuclei_bridge_shrink_factor']}, "
            f"diameter_range=[{debug_info['dmin']:.2f},{debug_info['dmax']:.2f}]x, "
            f"scales={debug_info['n_scales']}, "
            f"scales_with_seeds={debug_info['scales_with_seeds']}, "
            f"peak_seeds={debug_info['added_peak_seed_count']}, "
            f"z_anisotropy={debug_info['z_anisotropy']:.2f}, "
            f"z_weight={debug_info['z_split_weight']:.2f}, "
            f"erosion Z={min(er_z)}-{max(er_z)} "
            f"Y={min(er_y)}-{max(er_y)} "
            f"X={min(er_x)}-{max(er_x)} vox, "
            f"min_seed_vox={debug_info['min_seed_vox']}, "
            f"min_cell_vox={debug_info['min_cell_vox']}, "
            f"boundary_components={len(debug_info['boundary_components'])}, "
            f"added_component_seeds={debug_info['added_seed_count']}, "
            f"restored_isolated_voxels={debug_info['restored_isolated_count']}, "
            f"removed_by_roundness={debug_info['removed_by_roundness']}, "
            f"z_consistency_splits={debug_info['z_consistency_splits']}, "
            f"z_consistency_merges={debug_info['z_consistency_merges']})"
        )

        im_segmentation_stack['Nuclei'] = im_out
        im_segmentation_stack['Cytoplasm'] = np.zeros_like(im_out)
        im_segmentation_stack['PCM'] = np.zeros_like(im_out)
        return im_segmentation_stack

    im_in = im_final_stack['Threshold image'].copy()
    split_cfg = dict(nuclei_split_config)

    for c in progress(range(im_in.shape[3]), desc='Step 17 - Segment Nuclei'):
        if stain_complete_df.index[c] != 'NUCLEI':
            continue

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
            )
            im_mask = transl > 0
            im_mask = morphology.erosion(
                im_mask, footprint=np.ones((2, 2, 2))
            ).astype(im_mask.dtype)
            im_out, _ = skimage_label((transl * im_mask) > 0, return_num=True)

        else:
            binary_mask = im_in[:, :, :, c].astype(bool)
            im_out, debug_info = segment_nuclei_watershed(
                binary_mask=binary_mask,
                r_zX=r_zX,
                r_zY=r_zY,
                r_zZ=r_zZ,
                nuclei_diameter=nuclei_diameter,
                **split_cfg,
            )

            erosion_triplets = debug_info['erosion_triplets']
            er_z = [e[0] for e in erosion_triplets] if erosion_triplets else [1]
            er_y = [e[1] for e in erosion_triplets] if erosion_triplets else [1]
            er_x = [e[2] for e in erosion_triplets] if erosion_triplets else [1]

            print(
                f"Nuclei found: {int(im_out.max())} "
                f"(shrink_factor={split_cfg['nuclei_bridge_shrink_factor']}, "
                f"diameter_range=[{debug_info['dmin']:.2f},{debug_info['dmax']:.2f}]x, "
                f"scales={debug_info['n_scales']}, "
                f"scales_with_seeds={debug_info['scales_with_seeds']}, "
                f"peak_seeds={debug_info['added_peak_seed_count']}, "
                f"z_anisotropy={debug_info['z_anisotropy']:.2f}, "
                f"z_weight={debug_info['z_split_weight']:.2f}, "
                f"erosion Z={min(er_z)}-{max(er_z)} "
                f"Y={min(er_y)}-{max(er_y)} "
                f"X={min(er_x)}-{max(er_x)} vox, "
                f"min_seed_vox={debug_info['min_seed_vox']}, "
                f"min_cell_vox={debug_info['min_cell_vox']}, "
                f"boundary_components={len(debug_info['boundary_components'])}, "
                f"added_component_seeds={debug_info['added_seed_count']}, "
                f"restored_isolated_voxels={debug_info['restored_isolated_count']}, "
                f"removed_by_roundness={debug_info['removed_by_roundness']}, "
                f"z_consistency_splits={debug_info['z_consistency_splits']}, "
                f"z_consistency_merges={debug_info['z_consistency_merges']})"
            )

        im_segmentation_stack['Nuclei'] = im_out

    return im_segmentation_stack


def segment_cytoplasm(
    im_final_stack,
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    cyto_markers,
    cyto_factor,
    nuclei_diameter,
    r_zxyz,
    progress=None,
):
    """
    Segment cytoplasm from the thresholded image.

    If a CYTOPLASM channel is present it is segmented with watershed.
    Otherwise cytoplasm is grown from nuclei labels using cyto_markers (if any)
    or a simple label-grow.

    Parameters
    ----------
    im_final_stack : dict
        Must contain 'Threshold image'.
    im_segmentation_stack : dict
        Must contain 'Nuclei'.
    stain_df, stain_complete_df : DataFrame
        Staining metadata.
    cyto_markers : list of str
        Marker labels that contribute to cytoplasm expansion.
    cyto_factor : float
        Growth factor for label expansion when no explicit CYTOPLASM channel.
    nuclei_diameter : float
        Approximate nucleus diameter in micrometers.
    r_zxyz : tuple of float (r_zX, r_zY, r_zZ)
        Isotropic voxel sizes.
    progress : callable, optional
        Progress wrapper.

    Returns
    -------
    im_segmentation_stack : dict
        Updated dict with 'Cytoplasm' key added.
    stain_complete_df : DataFrame
        Updated dataframe (CYTOPLASM row added if it was absent).
    """
    from skimage.measure import label as skimage_label

    if progress is None:
        def progress(x, **kw): return x

    r_zX, r_zY, r_zZ = r_zxyz

    if 'Nuclei' not in im_segmentation_stack:
        raise RuntimeError(
            "Run nuclei segmentation first so im_segmentation_stack contains 'Nuclei'."
        )

    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)
            or len(cyto_markers) > 0 or cyto_factor > 1):
        return im_segmentation_stack, stain_complete_df

    im_in = im_final_stack['Threshold image'].copy()
    im_out = np.zeros_like(im_in[:, :, :, 0], dtype=np.int32)

    has_cyto_channel = 'CYTOPLASM' in stain_df.index

    if has_cyto_channel:
        for c in progress(range(im_in.shape[3]), desc='Step 18A - Segment Cytoplasm'):
            if stain_df.index[c] == 'CYTOPLASM':
                from scipy import ndimage as _ndi
                cyto_binary = im_in[:, :, :, c] > 0
                nuc_mask = im_segmentation_stack['Nuclei'] > 0
                combined_mask = cyto_binary | nuc_mask
                distance = _ndi.distance_transform_edt(
                    combined_mask, sampling=[r_zZ, r_zY, r_zX]
                )
                im_out = watershed(
                    -distance, im_segmentation_stack['Nuclei'],
                    mask=combined_mask,
                )
                print(f"Cytoplasm segmented from CYTOPLASM channel + nuclei region (watershed)")
                break
    else:
        if len(cyto_markers) == 0:
            im_out = grow_labels(im_segmentation_stack['Nuclei'], cyto_factor)
            print(f"Cytoplasm grown from nuclei labels (factor={cyto_factor})")
        else:
            im_out = im_segmentation_stack['Nuclei'] > 0
            for c in progress(range(im_in.shape[3]), desc='Step 18B - Apply Cyto Markers'):
                idx = stain_complete_df.index[c]
                marker = stain_complete_df.loc[idx, 'Marker']
                if marker in cyto_markers:
                    im_out = im_out + im_in[:, :, :, c]
            binary_mask = (im_out > 0).copy()
            im_out = grow_markers_within_islands_limited(
                im_segmentation_stack['Nuclei'], binary_mask, max_distance=10.0
            )
            print(f"Cytoplasm expanded using markers: {cyto_markers}")
        stain_complete_df = stain_complete_df.copy()
        stain_complete_df.loc['CYTOPLASM'] = ['', '', '', '', '', '']

    nuclei_labels = im_segmentation_stack['Nuclei']
    max_label = int(nuclei_labels.max())
    filled = 0
    for label_id in range(1, max_label + 1):
        nuc_mask = nuclei_labels == label_id
        if not np.any(nuc_mask):
            continue
        cyto_mask = im_out == label_id
        cyto_beyond_nuc = cyto_mask & ~nuc_mask
        if np.any(cyto_beyond_nuc):
            continue
        im_out[cyto_mask] = 0
        single = np.zeros_like(nuclei_labels, dtype=np.int32)
        single[nuc_mask] = label_id
        grown = grow_labels(single, cyto_factor)
        fill_mask = (grown == label_id) & (im_out == 0)
        im_out[fill_mask] = label_id
        filled += 1
    if filled > 0:
        print(f"Cytoplasm gap-filled for {filled} nuclei by label-grow (factor={cyto_factor})")

    im_segmentation_stack = dict(im_segmentation_stack)
    im_segmentation_stack['Cytoplasm'] = im_out.copy()
    return im_segmentation_stack, stain_complete_df


def segment_pcm(
    im_segmentation_stack,
    stain_df,
    cyto_markers,
    cyto_factor,
    PCM_factor,
):
    """
    Build the pericellular matrix (PCM) label volume by expanding cytoplasm
    or nuclei labels and subtracting the cytoplasm.

    Parameters
    ----------
    im_segmentation_stack : dict
        Must contain 'Nuclei' and 'Cytoplasm'.
    stain_df : DataFrame
        Staining metadata.
    cyto_markers : list of str
        Marker labels used for cytoplasm expansion.
    cyto_factor : float
        Growth factor used for cytoplasm.
    PCM_factor : float
        Total growth factor for PCM (relative to nuclei).

    Returns
    -------
    im_segmentation_stack : dict
        Updated dict with 'PCM' key added.
    """
    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)):
        return im_segmentation_stack

    has_cyto_channel = 'CYTOPLASM' in stain_df.index

    if has_cyto_channel or len(cyto_markers) > 0:
        P_factor = int(PCM_factor - cyto_factor)
        im_out = grow_labels(im_segmentation_stack['Cytoplasm'], P_factor)
    else:
        im_out = grow_labels(im_segmentation_stack['Nuclei'], PCM_factor)

    im_out = im_out - im_segmentation_stack['Cytoplasm']

    im_segmentation_stack = dict(im_segmentation_stack)
    im_segmentation_stack['PCM'] = im_out.copy()
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
    if progress is None:
        def progress(x, **kw): return x

    if not (('NUCLEI' in stain_df.index) or ('CYTOPLASM' in stain_df.index)):
        # LD mode: label each channel by masking the Nuclei label array with
        # the per-channel threshold so LIVE and DEAD cells are separately identified.
        im_in = im_final_stack['Threshold image'].copy()
        im_segmentation_stack = dict(im_segmentation_stack)
        for c in progress(range(im_in.shape[3]), desc='Step 20 - Assign Labels To Channels'):
            cond = stain_df.index[c]
            im_segmentation_stack[cond] = (im_in[:, :, :, c] > 0) * im_segmentation_stack['Nuclei']
        return im_segmentation_stack

    im_in = im_final_stack['Threshold image'].copy()
    im_segmentation_stack = dict(im_segmentation_stack)

    for c in progress(range(im_in.shape[3]), desc='Step 20 - Assign Labels To Channels'):
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
    denoised, corrected, filtered, equalized).
    Viewer 1 (only when NUCLEI or CYTOPLASM is present) shows label overlays for
    nuclei, cytoplasm, PCM, and marker channels.

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
    if progress is None:
        def progress(x, **kw): return x

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
    for stage_key, stage_prefix, scale in progress(stages, desc='Step 22A - Add Layers To Viewer 0'):
        for c in range(im_thr.shape[3]):
            idx = stain_complete_df.index[c]
            marker = stain_complete_df.loc[idx, 'Marker']
            color = stain_complete_df['Color'].iloc[c]
            viewer_0.add_image(
                im_final_stack[stage_key][:, :, :, c],
                name=f'{stage_prefix} {idx} ({marker})', colormap=color,
                blending='additive', scale=scale,
            )
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
            color = stain_df['Color'].iloc[c]
            viewer_1.add_image(
                eq_img[:, :, :, c],
                name=f'EQ {idx} ({marker})', colormap=color,
                blending='additive', scale=scale_zoom,
            )
        for c in progress(range(len(stain_complete_df.index)), desc='Step 22B - Add Labels To Viewer 1'):
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
                viewer_1.add_labels(
                    im_segmentation_stack[idx].astype(np.int32),
                    name=f'{idx} ({marker})', blending='additive', scale=scale_zoom,
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
    size : int
        Side length of the cubic export volume in voxels (default 90).
    """
    import pyvista as pv
    from pathlib import Path as _Path

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

    half = size // 2
    Zmax, Ymax, Xmax = nuc_vol.shape
    z0d, z1d = cz - half, cz + half
    y0d, y1d = cy - half, cy + half
    x0d, x1d = cx - half, cx + half

    z0c, z1c = max(0, z0d), min(Zmax, z1d)
    y0c, y1c = max(0, y0d), min(Ymax, y1d)
    x0c, x1c = max(0, x0d), min(Xmax, x1d)

    zp0, zp1 = z0c - z0d, z0c - z0d + (z1c - z0c)
    yp0, yp1 = y0c - y0d, y0c - y0d + (y1c - y0c)
    xp0, xp1 = x0c - x0d, x0c - x0d + (x1c - x0c)

    def _crop3d(vol):
        out = np.zeros((size, size, size), dtype=vol.dtype)
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
    grid.dimensions = (size, size, size)
    grid.origin = (float(x0d), float(y0d), float(z0d))
    grid.spacing = (1.0, 1.0, 1.0)
    grid.point_data['Nuclei_label'] = nuclei_crop.ravel()
    grid.point_data['Cytoplasm_label'] = cytoplasm_crop.ravel()
    for ch_name, crop in marker_crops.items():
        grid.point_data[ch_name] = crop.ravel()

    out_path = str(_Path(input_file).stem + f"_nuc{nuc_label}_3Dcrop.vtk")
    grid.save(out_path)
    print(f"Saved: {out_path}  ({size}^3 voxels, {2 + len(marker_crops)} channels)")


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
    """
    import pyvista as pv
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    from pathlib import Path as _Path
    from IPython.display import clear_output

    r_X, r_Y, r_Z = r_xyz
    nuc_max = int(np.max(im_segmentation_stack['Nuclei']))
    cyto_max = int(np.max(im_segmentation_stack['Cytoplasm']))
    pcm_max = int(np.max(im_segmentation_stack['PCM']))

    blocks_nuclei = pv.MultiBlock()
    blocks_cyto = pv.MultiBlock()
    blocks_PCM = pv.MultiBlock()

    nuc_vol = np.zeros((nuc_max + 1,))
    nuc_coord = np.zeros((nuc_max + 1, 3))
    cyto_vol = np.zeros((cyto_max + 1,))
    cyto_coord = np.zeros((cyto_max + 1, 3))
    PCM_vol = np.zeros((pcm_max + 1,))
    PCM_coord = np.zeros((pcm_max + 1, 3))

    iter_ = (
        progress(range(1, nuc_max + 1), desc='Step 30 - Build VTK Volumes')
        if progress else range(1, nuc_max + 1)
    )
    k = 0
    for j in iter_:
        clear_output(wait=True)
        print(f'NUCLEI {j} / {nuc_max}')

        # --- nuclei ---
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['Nuclei'] == j))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, "part_nuclei_mesh.stl")

        mesh_nuclei = pv.read("part_nuclei_mesh.stl")
        if mesh_nuclei.volume > 0.0:
            mesh_nuclei.decimate(target_reduction=0.8, inplace=True)
            nuc_vol[k] = mesh_nuclei.volume
            nuc_coord[k] = mesh_nuclei.center

            mesh_nuclei.cell_data['ID'] = np.ones(mesh_nuclei.n_cells) * (k + 1)
            mesh_nuclei.cell_data['Nuclei volume (um3)'] = (
                np.ones(mesh_nuclei.n_cells) * nuc_vol[k] * r_X * r_Y * r_Z / np.prod(zoom_factors)
            )
            mesh_nuclei.cell_data['Z nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][0] * r_Z / zoom_factors[0]
            mesh_nuclei.cell_data['Y nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][1] * r_Y / zoom_factors[1]
            mesh_nuclei.cell_data['X nuclei (um)'] = np.ones(mesh_nuclei.n_cells) * nuc_coord[k][2] * r_X / zoom_factors[2]
            blocks_nuclei.append(mesh_nuclei)
            k += 1

        # --- cytoplasm ---
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['Cytoplasm'] == j))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, "part_cyto_mesh.stl")
        mesh_cyto = pv.read("part_cyto_mesh.stl")

        # --- PCM ---
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack['PCM'] == j))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, "part_PCM_mesh.stl")
        mesh_PCM = pv.read("part_PCM_mesh.stl")

        if mesh_cyto.volume > 0.0:
            mesh_cyto.decimate(target_reduction=0.8, inplace=True)
            mesh_PCM.decimate(target_reduction=0.8, inplace=True)

            cyto_vol[k] = mesh_cyto.volume
            cyto_coord[k] = mesh_cyto.center
            PCM_vol[k] = mesh_PCM.volume
            PCM_coord[k] = mesh_PCM.center

            voxel_scale = r_X * r_Y * r_Z / np.prod(zoom_factors)

            mesh_cyto.cell_data['ID'] = np.ones(mesh_cyto.n_cells) * (k + 1)
            mesh_PCM.cell_data['ID'] = np.ones(mesh_PCM.n_cells) * (k + 1)
            mesh_cyto.cell_data['Cellular volume (um3)'] = np.ones(mesh_cyto.n_cells) * cyto_vol[k] * voxel_scale
            mesh_PCM.cell_data['PCM volume (um3)'] = np.ones(mesh_PCM.n_cells) * PCM_vol[k] * voxel_scale
            mesh_cyto.cell_data['Z cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][0] * r_Z / zoom_factors[0]
            mesh_cyto.cell_data['Y cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][1] * r_Y / zoom_factors[1]
            mesh_cyto.cell_data['X cell (um)'] = np.ones(mesh_cyto.n_cells) * cyto_coord[k][2] * r_X / zoom_factors[2]
            mesh_PCM.cell_data['Z PCM (um)'] = np.ones(mesh_PCM.n_cells) * PCM_coord[k][0] * r_Z / zoom_factors[0]
            mesh_PCM.cell_data['Y PCM (um)'] = np.ones(mesh_PCM.n_cells) * PCM_coord[k][1] * r_Y / zoom_factors[1]
            mesh_PCM.cell_data['X PCM (um)'] = np.ones(mesh_PCM.n_cells) * PCM_coord[k][2] * r_X / zoom_factors[2]

            for i, marker in enumerate(labels_full_df.index):
                row = labels_full_df.iloc[i]
                cond = row['Condition']
                if cond in ('NUCLEI', 'CYTOPLASM') or np.size(marker) != 1:
                    continue
                shared = list(row['Shared labels'])
                if j in shared:
                    idx = shared.index(j)
                    vol_um3 = row['Marker size [um3]'][idx]
                    cyto_combined = (cyto_vol[k] + PCM_vol[k]) * voxel_scale
                    mesh_cyto.cell_data[marker + ' volume (um3)'] = np.ones(mesh_cyto.n_cells) * vol_um3
                    mesh_PCM.cell_data[marker + ' volume (um3)'] = np.ones(mesh_PCM.n_cells) * vol_um3
                    mesh_cyto.cell_data[marker + ' volume cytoplasm (um3)'] = np.ones(mesh_cyto.n_cells) * row['Marker size cytoplasm [um3]'][idx]
                    mesh_PCM.cell_data[marker + ' volume PCM (um3)'] = np.ones(mesh_PCM.n_cells) * row['Marker size PCM [um3]'][idx]
                    mesh_cyto.cell_data[marker + ' rel. vol. (-)'] = np.ones(mesh_cyto.n_cells) * (vol_um3 / cyto_combined)
                    mesh_PCM.cell_data[marker + ' rel. vol. (-)'] = np.ones(mesh_PCM.n_cells) * (vol_um3 / cyto_combined)
                    mesh_cyto.cell_data[marker + ' rel. vol. cytoplasm (-)'] = np.ones(mesh_cyto.n_cells) * (row['Marker size cytoplasm [um3]'][idx] / (cyto_vol[k] * voxel_scale))
                    mesh_PCM.cell_data[marker + ' rel. vol. PCM (-)'] = np.ones(mesh_PCM.n_cells) * (row['Marker size PCM [um3]'][idx] / (PCM_vol[k] * voxel_scale))
                    mesh_cyto.cell_data[marker + ' avg. intensity (-)'] = np.ones(mesh_cyto.n_cells) * row['Avg. marker intensity'][idx]
                    mesh_PCM.cell_data[marker + ' avg. intensity (-)'] = np.ones(mesh_PCM.n_cells) * row['Avg. marker intensity'][idx]
                    mesh_cyto.cell_data[marker + ' avg. cytoplasm int. (-)'] = np.ones(mesh_cyto.n_cells) * row['Avg. marker intensity cytoplasm'][idx]
                    mesh_PCM.cell_data[marker + ' avg. PCM int. (-)'] = np.ones(mesh_PCM.n_cells) * row['Avg. marker intensity PCM'][idx]
                else:
                    mesh_cyto.cell_data[marker + ' expression (um3)'] = np.zeros(mesh_cyto.n_cells)
                    mesh_cyto.cell_data[marker + ' rel. expr. (-)'] = np.zeros(mesh_cyto.n_cells)

            blocks_cyto.append(mesh_cyto)
            blocks_PCM.append(mesh_PCM)

    stem = str(_Path(input_file).stem)
    blocks_nuclei.extract_geometry().save(stem + '_NUCLEI_labelled.vtk')
    blocks_cyto.extract_geometry().save(stem + '_CYTOPLASM_labelled.vtk')
    blocks_PCM.extract_geometry().save(stem + '_PCM_labelled.vtk')


def export_marker_stl(
    im_segmentation_stack,
    stain_df,
    stain_complete_df,
    input_file,
    progress=None,
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
    progress : callable or None
        tqdm-compatible wrapper.
    """
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    from pathlib import Path as _Path

    iter_ = (
        progress(
            enumerate(stain_complete_df.index),
            total=len(stain_complete_df.index),
            desc='Step 31 - Export Marker STL',
        )
        if progress else enumerate(stain_complete_df.index)
    )
    for c, _marker in iter_:
        if stain_complete_df.index[c] in ('NUCLEI', 'CYTOPLASM', 'PCM'):
            continue
        row = stain_complete_df.iloc[c]
        simpleVolume = mrn.simpleVolumeFrom3Darray(np.float32(im_segmentation_stack[stain_df.index[c]] > 0))
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
        _g2m_settings = mr.GridToMeshSettings()
        _g2m_settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
        _g2m_settings.isoValue = 0.5
        mesh_stl = mr.gridToMesh(floatGrid, _g2m_settings)
        mr.saveMesh(mesh_stl, str(_Path(input_file).stem) + "_" + row['Marker'] + "_mesh.stl")


def export_fea_mesh(
    im_segmentation_stack,
    input_file,
    progress=None,
):
    """Build a FEA tetrahedral mesh from the nuclei segmentation and write an Abaqus .inp file.

    This function combines three steps:
    1. Tetrahedralize the nuclei surface mesh with TetGen.
    2. Assign tetrahedral elements to individual nucleus labels.
    3. Write a final ``_FEA.inp`` Abaqus file with per-nucleus element sets.

    Parameters
    ----------
    im_segmentation_stack : dict
        Must contain 'Nuclei'.
    input_file : str or Path
        Source image path (used to derive output filenames).
    progress : callable or None
        tqdm-compatible wrapper.
    """
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    import meshio
    import tetgen
    import statistics as st
    from pathlib import Path as _Path

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
    tet.write('FE_segmentation_full.vtk', binary=False)

    meshel = meshio.read('FE_segmentation_full.vtk')
    meshel.write('FE_segmentation.inp')

    # --- Step 2: assign elements to nucleus labels ---
    cell_elements = {c: [] for c in range(1, nuc_max + 1)}

    iter_ = (
        progress(enumerate(elems), total=len(elems), desc='Step 33B - Assign Elements To Cells')
        if progress else enumerate(elems)
    )
    for ce, x in iter_:
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

    # --- Step 3: write element sets and finalise .inp ---
    with open("FE_segmentation.inp", "a") as f:
        for c in (
            progress(range(1, nuc_max + 1), desc='Step 33C - Write Element Sets')
            if progress else range(1, nuc_max + 1)
        ):
            f.write(f"*Elset, elset=cell{c}\n")
            j = 1
            for t in range(1, len(cell_elements[c])):
                f.write(str(cell_elements[c][t]) + ",")
                j += 1
                if j > 16:
                    f.write("\n")
                    j = 1
            f.write("\n")

    with open("FE_segmentation.inp", "r") as f:
        lines = f.readlines()

    out_path = str(_Path(input_file).stem) + "_FEA.inp"
    with open(out_path, "w") as f:
        for line in (
            progress(lines, desc='Step 33D - Write Final INP')
            if progress else lines
        ):
            if line == "*NODE\n":
                f.write("*PART, name=Part-1\n")
            f.write(line)
        f.write("*END PART\n")

