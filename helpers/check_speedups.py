"""Check that the faster per-label steps (bounding-box crops) give exactly
the results of the previous version, and how much faster they are.

The previous version of ``helpers/notebook_helpers.py`` is read from git
(``REF``, default HEAD) and both versions are run on the same synthetic 3D
image of touching nuclei. Run from the repository root, e.g. in a notebook
cell::

    %run helpers/check_speedups.py

or ``python helpers/check_speedups.py [git-ref]``. Each line says SAME or
DIFFERENT, with the time of the old and the new version.
"""

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
REF = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "HEAD"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_old(ref):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:helpers/notebook_helpers.py"],
                         capture_output=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "notebook_helpers_old.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("notebook_helpers_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(shape=(36, 192, 192), n=110, seed=0):
    """Touching, partly merged nuclei (~16 voxels across) with intensity dips."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    intensity = np.zeros(shape, np.float32)
    for _ in range(n):
        c = rng.uniform([6, 10, 10], [shape[0] - 6, shape[1] - 10, shape[2] - 10])
        r = rng.uniform(6.0, 9.0)
        d2 = ((zz - c[0]) * 2.0) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        intensity = np.maximum(intensity, rng.uniform(120, 230) * np.exp(-d2 / (2 * (0.6 * r) ** 2)))
    intensity += rng.normal(0, 6, shape).astype(np.float32)
    intensity = np.clip(intensity, 0, 255)
    mask = ndi.gaussian_filter(intensity, 1.0) > 60
    return mask, ndi.gaussian_filter(intensity, 0.5), intensity


def _same(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, pd.DataFrame):
        return a.equals(b)
    if isinstance(a, (np.ndarray, np.generic)) or isinstance(b, (np.ndarray, np.generic)):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) \
            if np.asarray(a).dtype.kind == "f" else np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


results = []


def compare(name, old_fn, new_fn):
    t0 = time.perf_counter()
    out_old = old_fn()
    t1 = time.perf_counter()
    out_new = new_fn()
    t2 = time.perf_counter()
    same = _same(out_old, out_new)
    results.append((name, same, t1 - t0, t2 - t1))
    print(f"{'SAME     ' if same else 'DIFFERENT'} {name:<48} old {t1 - t0:7.2f} s   "
          f"new {t2 - t1:7.2f} s")
    return out_new


def main():
    import importlib
    import helpers.notebook_helpers as new
    new = importlib.reload(new)  # the file as it is now, not a copy loaded earlier
    old = _load_old(REF)
    print(f"Old version: {REF}; new version: working tree.\n")

    mask, filt, raw = _synthetic()
    spacing = dict(r_zX=0.5, r_zY=0.5, r_zZ=1.0)
    cfg = new.get_nuclei_split_config(
        "balanced", nuclei_bridge_shrink_factor=0.22, nuclei_seed_min_fraction=0.03,
        nuclei_min_roundness=0.55, nuclei_gradient_smooth_sigma=1.5)

    labels, _ = compare(
        "segment_nuclei_watershed (whole step)",
        lambda: old.segment_nuclei_watershed(mask, **spacing, nuclei_diameter=8.0, intensity_img=filt,
                                             intensity_img_raw=raw, **cfg),
        lambda: new.segment_nuclei_watershed(mask, **spacing, nuclei_diameter=8.0, intensity_img=filt,
                                             intensity_img_raw=raw, **cfg))
    print(f"   ({int(labels.max())} labels)")

    # The steps on their own, on a first-pass labelling with merged clusters.
    first, _ = ndi.label(mask)
    sp = (1.0, 0.5, 0.5)
    exp_vox = (4 / 3 * np.pi * 4.0 ** 3) / (0.5 * 0.5 * 1.0)
    compare("_split_nonround_clusters_by_intensity_peaks",
            lambda: old._split_nonround_clusters_by_intensity_peaks(first, raw, sp, 0.55, exp_vox, 3, 0.25,
                                                                    smooth_sigma=1.5),
            lambda: new._split_nonround_clusters_by_intensity_peaks(first, raw, sp, 0.55, exp_vox, 3, 0.25,
                                                                    smooth_sigma=1.5))
    compare("_split_nonround_clusters_by_rethreshold",
            lambda: old._split_nonround_clusters_by_rethreshold(first, raw, sp, 0.55, exp_vox, smooth_sigma=1.5),
            lambda: new._split_nonround_clusters_by_rethreshold(first, raw, sp, 0.55, exp_vox, smooth_sigma=1.5))
    compare("_filter_labels_by_roundness_xy",
            lambda: old._filter_labels_by_roundness_xy(labels, 0.55, 1.5 * exp_vox),
            lambda: new._filter_labels_by_roundness_xy(labels, 0.55, 1.5 * exp_vox))
    compare("_split_labels_by_z_consistency (watershed)",
            lambda: old._split_labels_by_z_consistency(first, 8, aggressive=True),
            lambda: new._split_labels_by_z_consistency(first, 8, aggressive=True))
    compare("_split_labels_by_z_consistency (topology)",
            lambda: old._split_labels_by_z_consistency(first, 8, aggressive=False),
            lambda: new._split_labels_by_z_consistency(first, 8, aggressive=False))
    compare("_merge_labels_by_z_consistency",
            lambda: old._merge_labels_by_z_consistency(labels),
            lambda: new._merge_labels_by_z_consistency(labels))

    # Quantification (Cells 21-24 of the LD notebook).
    cells = new.expand_labels_um(labels, 2.0, sp).astype(np.int32)
    seg = {"Nuclei": labels.astype(np.int32), "Cytoplasm": cells, "PCM": np.zeros_like(cells),
           "LIVE": (raw > 90).astype(np.int32)}
    filtered = np.stack([filt, raw], axis=-1)
    stain = pd.DataFrame({"Marker": ["Hoechst", "Calcein"], "Laser": ["405", "488"],
                          "Color": ["BLUE", "GREEN"], "Cont_min": [0, 0], "Cont_max": [255, 255],
                          "Gamma": [1.0, 1.0]}, index=["NUCLEI", "LIVE"])
    compare("compute_nuclei_cytoplasm_stats",
            lambda: old.compute_nuclei_cytoplasm_stats(seg, (0.5, 0.5, 1.0), None),
            lambda: new.compute_nuclei_cytoplasm_stats(seg, (0.5, 0.5, 1.0), None))
    compare("_compute_marker_stats_core",
            lambda: old._compute_marker_stats_core(1, seg, filtered, (0.5, 0.5, 1.0), stain, "LIVE", True),
            lambda: new._compute_marker_stats_core(1, seg, filtered, (0.5, 0.5, 1.0), stain, "LIVE", True))
    compare("compute_percell_marker_intensity_df",
            lambda: old.compute_percell_marker_intensity_df(seg, {"Filtered image": filtered}, stain),
            lambda: new.compute_percell_marker_intensity_df(seg, {"Filtered image": filtered}, stain))
    compare("collect_histogram_data",
            lambda: old.collect_histogram_data(seg, filtered, stain, stain),
            lambda: new.collect_histogram_data(seg, filtered, stain, stain))

    n_diff = sum(not same for _, same, _, _ in results)
    t_old = sum(r[2] for r in results)
    t_new = sum(r[3] for r in results)
    print(f"\n{len(results) - n_diff}/{len(results)} identical. Total: old {t_old:.1f} s, "
          f"new {t_new:.1f} s ({t_old / max(t_new, 1e-9):.1f}x faster).")
    if n_diff:
        print("Some results DIFFER: please report the lines marked DIFFERENT.")


if __name__ == "__main__":  # also with %run in a notebook
    main()
