from collections import defaultdict
import os

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image as PILImage
from scipy import ndimage as ndi
from skimage.draw import line as draw_line
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from skimage.segmentation import watershed


__all__ = [
    "assign_labels",
    "compute_full_marker_stats_for_marker",
    "compute_marker_stats_for_marker",
    "compute_nuclei_cytoplasm_stats",
    "contr_limit",
    "contr_stretch",
    "create_row_pdf",
    "crop_nucleus_with_padding",
    "detect_peaks_xy_with_best_z",
    "double_plateau_hist_equalization_nd",
    "gamma_trans",
    "get_stain_name",
    "grow_labels",
    "grow_markers_within_islands_limited",
    "hist_plot",
    "ImageProcessing",
    "make_anisotropic_footprint",
    "merge_small_touching_labels",
    "merge_touching_labels",
    "napari_gamma",
    "napari_contrast_gamma_uint8",
    "remove_small_island_labels",
    "remove_small_islands",
    "save_merged_figure",
    "save_raw_png",
    "save_single_channel_png",
    "set_notebook_context",
    "shrink_to_markers",
    "shrink_to_markers_robust",
    "stardist3d_from_2d",
    "truncate_cell",
    "voxel_volume",
    "watershed_nuclei",
]


def set_notebook_context(**kwargs):
    """Store notebook variables that some helper functions read later."""
    globals().update(kwargs)


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
                    np.mean(x_nuc * r_xyz[0] / zooms[2]),
                    np.mean(y_nuc * r_xyz[1] / zooms[1]),
                    np.mean(z_nuc * r_xyz[2] / zooms[0]),
                )
            )
            nucleus_sizes.append(x_nuc.size * r_xyz[0] * r_xyz[1] * r_xyz[2] / np.prod(zooms))

        z_cyto, y_cyto, x_cyto = np.where(seg_stack["Cytoplasm"] == label_id)
        if x_cyto.size == 0:
            cytoplasm_positions.append((0.0, 0.0, 0.0))
            cytoplasm_sizes.append(0.0)
        else:
            cytoplasm_positions.append(
                (
                    np.mean(x_cyto * r_xyz[0] / zooms[2]),
                    np.mean(y_cyto * r_xyz[1] / zooms[1]),
                    np.mean(z_cyto * r_xyz[2] / zooms[0]),
                )
            )
            cytoplasm_sizes.append(x_cyto.size * r_xyz[0] * r_xyz[1] * r_xyz[2] / np.prod(zooms))

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
        marker_sizes.append(voxels[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
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
            marker_cyto_sizes.append(voxels_cyto[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
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
            marker_pcm_sizes.append(voxels_pcm[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
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
        marker_sizes.append(voxels[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
        if channel_idx is not None:
            values = filtered_img[voxels[0], voxels[1], voxels[2], channel_idx]
            avg_marker.append(float(np.mean(values)) if values.size > 0 else 0.0)
            std_marker.append(float(np.std(values)) if values.size > 0 else 0.0)
        else:
            avg_marker.append(0.0)
            std_marker.append(0.0)

        cytoplasm_marker_mask = (marker_img > 0) & cytoplasm_mask
        voxels_cyto = np.where(cytoplasm_marker_mask)
        marker_cyto_sizes.append(voxels_cyto[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
        if channel_idx is not None and voxels_cyto[0].size > 0:
            values_cyto = filtered_img[voxels_cyto[0], voxels_cyto[1], voxels_cyto[2], channel_idx]
            avg_cyto_marker.append(float(np.mean(values_cyto)))
            std_cyto_marker.append(float(np.std(values_cyto)))
        else:
            avg_cyto_marker.append(0.0)
            std_cyto_marker.append(0.0)

        pcm_marker_mask = (marker_img > 0) & pcm_mask
        voxels_pcm = np.where(pcm_marker_mask)
        marker_pcm_sizes.append(voxels_pcm[0].size * voxel_volume(r_xyz[0], r_xyz[1], r_xyz[2], zooms))
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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
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

    if thumb_size is None:
        thumb_size = (2.2 * inch, 2.2 * inch)

    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []
    nuclei = sorted(hist_data.keys())
    if not nuclei:
        raise ValueError("hist_data is empty.")

    all_conditions = sorted({condition for nucleus_data in hist_data.values() for condition in nucleus_data.keys()})
    marker_conditions = [condition for condition in all_conditions if condition.lower() != "nuclei"]
    marker_conditions = (marker_conditions + marker_conditions)[:3]
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

        data = [
            [Image(density_png, width=6.0 * inch, height=2.0 * inch)],
            [Image(path, width=thumb_size[0], height=thumb_size[1]) for path in channel_pngs] + [Image(merged_png, width=thumb_size[0], height=thumb_size[1])],
        ]
        table = Table(data, colWidths=[1.5 * inch] * 4, rowHeights=[2.0 * inch, thumb_size[1]])
        table.setStyle(TableStyle([
            ("SPAN", (0, 0), (3, 0)),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
