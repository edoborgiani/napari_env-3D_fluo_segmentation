"""Optimal (validated) settings of the nuclei segmentation, for the user notebooks.

The user notebooks ask only what depends on the experiment (file, ROI, stain
table, cell size, method, exports). Every setting of the segmentation itself
was tuned against manual ground truths and is read here from
``settings/optimal_settings_v<version>.json`` (``SETTINGS_DIR``, the
``settings`` folder at the top of the repository).

``version='best'`` (the default of `load_optimal_settings`) loads the
version with the highest nuclei mean F1 on the ground truths (then F1@0.5;
`best_version`): v1.6.41 at present. It is used by `NucleoSeg3D_v1.6.41.ipynb`
and the LIVE/DEAD notebook (`helpers.optimal_settings_ld`).
`NucleoSeg3D_v1.6.4.ipynb` asks for ``'1.6.4'``, so it keeps its results.

- **v1.6.4**: tuned on four ground truths;
- **v1.6.41**: re-tuned on six ground truths, each scored as a new image.
  Compared with v1.6.4:

  - the calibrated NUCLEI contrast uses gamma ``nuclei_gamma`` 1.15
    instead of 1.0;
  - the resampling keeps the last Z slice (``resample_edge_mode``
    'nearest', Cell 10);
  - the cell bodies (Cell 18) have their own settings (``cyto_*``,
    `cytoplasm_settings`), tuned on Cellpose cell bodies of the manual nuclei;
  - settings that follow each image's channel histograms
    (``histogram_rules``, `adapt_to_histograms`, Cell 8; fitted and checked
    leave-one-image-out): the nuclei edge fraction from the NUCLEI
    signal-to-noise ratio.

In both versions:

- the NUCLEI contrast of every image is set from its own histogram
  (calibrated rule, `calibrated_contrast`);
- the nucleus size, `scale_factor` and the um sizes are measured / adapted
  in each image (`adapt_settings_to_image`);
- everything else (smoothing, threshold, watershed split configuration,
  size refinement, Z elongation, haze removal, flare trim, saddle / tall
  split, rescue of missed nuclei, edges) is the validated value.

Any of them can still be changed by name with ``custom_settings`` (Cell 4).
"""

import json
from pathlib import Path

VERSIONS = ("1.6.4", "1.6.41")
LATEST = VERSIONS[-1]
# the settings files: <repository>/settings/
SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings"


def _validation_score(version):
    """(mean F1, F1@0.5, number of ground truths) of one settings file."""
    record = json.loads((SETTINGS_DIR / f"optimal_settings_v{version}.json")
                        .read_text(encoding="utf-8"))
    v = (record.get("validation") or {}).get("all") or {}
    return (float(v.get("mean_f1") or 0.0), float(v.get("f1_50") or 0.0),
            len(record.get("ground_truths") or []))


def best_version():
    """The version whose validated settings score best on the manual ground
    truths: highest nuclei mean F1 (IoU 0.5-0.75, accurate outlines), then
    F1@0.5 (nuclei found), then the number of ground truths. Settings files
    missing from ``settings/`` are skipped."""
    present = [v for v in VERSIONS if (SETTINGS_DIR / f"optimal_settings_v{v}.json").exists()]
    if not present:
        raise FileNotFoundError(f"No settings file in {SETTINGS_DIR}")
    return max(present, key=_validation_score)


def resolve_version(version="best"):
    """``version``, with ``'best'`` replaced by `best_version`."""
    if version == "best":
        return best_version()
    if version not in VERSIONS:
        raise ValueError(f"version must be 'best' or one of {VERSIONS}, not {version!r}")
    return version


def settings_file(version="best"):
    """The settings file of one version (``settings/optimal_settings_v<version>.json``;
    ``'best'`` = `best_version`)."""
    return SETTINGS_DIR / f"optimal_settings_v{resolve_version(version)}.json"


# the settings file of the v1.6.4 notebook (kept for older code)
SETTINGS_FILE = settings_file("1.6.4")

# segment_nuclei keywords taken from the settings (besides the user's own)
_SEGMENT_KEYS = (
    "size_reference", "iterative_split", "oversize_factor", "max_split_iterations",
    "merge_undersized", "undersize_factor", "split_gamma", "split_gamma_step",
    "min_piece_fraction", "z_elongation", "nuclei_expand_um", "nuclei_flare_trim_factor",
    "nuclei_saddle_dip", "nuclei_edge_fraction", "nuclei_edge_margin_um",
    "nuclei_tall_volume_factor", "nuclei_tall_elongation_factor", "nuclei_rescue_contrast",
    "nuclei_rescue_grow_fraction")
# run_batch_folder(nuclei_options=...) keys
_BATCH_OPTION_KEYS = (
    "z_atten_strength", "z_atten_conditions", "z_bg_radius_um", "z_bg_percentile",
    "global_threshold_weight", "sauvola_weight", "z_elongation", "nuclei_expand_um",
    "nuclei_flare_trim_factor", "nuclei_saddle_dip", "nuclei_edge_fraction",
    "nuclei_edge_margin_um", "nuclei_tall_volume_factor", "nuclei_tall_elongation_factor",
    "nuclei_rescue_contrast", "nuclei_rescue_grow_fraction", "nuclei_contrast",
    "nuclei_gamma", "resample_edge_mode")


def load_optimal_settings(custom_settings=None, path=None, verbose=True, version="best"):
    """The validated settings of ``version`` as a dict (``opt``), with
    ``custom_settings`` (``{name: value}``) applied on top.
    ``version``: ``'best'`` (default) = the best-scoring settings file
    (`best_version`), or a version of `VERSIONS` (e.g. ``'1.6.4'`` to keep
    the results of an older notebook).
    ``nuclei_split_config_overrides`` in ``custom_settings`` is merged into the
    validated one. Adds ``nuclei_split_config`` (for `segment_nuclei`) and
    ``_version`` (the version loaded). ``path`` (a settings file) overrides
    ``version``."""
    from helpers.notebook_helpers import get_nuclei_split_config
    if verbose and not path and version == "best":
        best = best_version()
        others = ", ".join(f"v{v} {_validation_score(v)[0]:.3f}" for v in VERSIONS
                           if v != best and (SETTINGS_DIR / f"optimal_settings_v{v}.json").exists())
        print(f"version='best': v{best}, the highest nuclei mean F1 on the ground truths"
              + (f" (others: {others})." if others else "."))
    path = Path(path) if path else settings_file(version)
    record = json.loads(path.read_text(encoding="utf-8"))
    opt = dict(record["settings"])
    opt.setdefault("nuclei_gamma", 1.0)  # v1.6.4 file: gamma 1.0
    # v1.6.4 file: the resampling leaves the last Z slice (and row / column) empty
    opt.setdefault("resample_edge_mode", "constant")
    custom = dict(custom_settings or {})
    unknown = set(custom) - set(opt)
    if unknown:
        raise ValueError(f"Unknown setting(s) in custom_settings: {sorted(unknown)}. "
                         f"Known: {sorted(opt)}")
    if "nuclei_split_config_overrides" in custom:
        custom["nuclei_split_config_overrides"] = {
            **opt["nuclei_split_config_overrides"], **custom["nuclei_split_config_overrides"]}
    opt.update(custom)
    # v1.6.41: the settings the user set are not changed by the histogram rules
    opt["_custom_keys"] = sorted(custom)
    if opt["nuclei_contrast"] not in ("calibrated", None):
        raise ValueError("nuclei_contrast must be 'calibrated' or None (None = as the other "
                         "channels: setup file, napari or automatic_contrast)")
    opt["nuclei_split_config"] = get_nuclei_split_config(
        profile="balanced", **opt["nuclei_split_config_overrides"])
    opt["_version"] = record.get("version")
    if verbose:
        v = record.get("validation", {})
        print(f"Optimal settings v{record.get('version')} ({path.name}), validated on "
              f"{len(record.get('ground_truths', []))} manual ground truths "
              f"(mean F1 {v.get('all', {}).get('mean_f1')}, F1@0.5 "
              f"{v.get('all', {}).get('f1_50')}).")
        if custom:
            print("Changed by custom_settings: " + ", ".join(
                f"{k} = {custom[k]!r}" for k in sorted(custom)))
    return opt


def set_nuclei_contrast(im_original, stain_complete_df, original_stain_complete_df, opt):
    """Cell 8: the NUCLEI contrast / gamma from the image's own histogram
    (``opt['nuclei_contrast'] == 'calibrated'``), in both stain tables."""
    if opt["nuclei_contrast"] != "calibrated" or "NUCLEI" not in stain_complete_df.index:
        return None
    from helpers.notebook_helpers import calibrated_contrast
    contrast = calibrated_contrast(im_original[..., stain_complete_df.index.get_loc("NUCLEI")],
                                   gamma=opt.get("nuclei_gamma", 1.0))
    for col, value in contrast.items():
        stain_complete_df.loc["NUCLEI", col] = value
        original_stain_complete_df.loc["NUCLEI", col] = value
    print(f"NUCLEI contrast from the image's own histogram: Cont_min {contrast['Cont_min']}, "
          f"Cont_max {contrast['Cont_max']}, Gamma {contrast['Gamma']:g} (the segmentation "
          "settings were validated with this rule).")
    return contrast


def adapt_to_histograms(im_original, stain_complete_df, opt):
    """Cell 8 (v1.6.41): the settings that follow the channel histograms
    (``opt['histogram_rules']``, ``value = clip(a + b x feature, lo, hi)``,
    see `helpers.histogram_adapt`), when ``opt['adapt_to_histograms']``.
    Settings given in ``custom_settings`` are kept. Updates ``opt`` in place;
    returns ``[(setting, old, new, feature)]``."""
    opt["_histogram_changes"] = []
    if not opt.get("adapt_to_histograms") or not opt.get("histogram_rules"):
        return []
    from helpers.histogram_adapt import adapt_settings, features_by_condition
    feats = features_by_condition(im_original, stain_complete_df)
    changes = adapt_settings(opt, opt["histogram_rules"], feats, keep=opt.get("_custom_keys", ()))
    opt["_histogram_features"] = feats
    opt["_histogram_changes"] = changes
    nuc = feats.get("NUCLEI")
    if nuc:
        print(f"NUCLEI histogram: background {nuc['background']:.1f}, noise {nuc['noise']:.2f}, "
              f"signal-to-noise {nuc['snr']:.1f} (0-255 scale).")
    for name, old, new, src in changes:
        print(f"Adapted to the histogram ({src}): {name} {old} -> {new}")
    kept = [k for k in opt["histogram_rules"] if k.partition(".")[0] in opt.get("_custom_keys", ())]
    if kept:
        print("Kept as set in custom_settings: " + ", ".join(kept))
    return changes


def adapt_to_image(im_original, stain_complete_df, voxel_size_zyx, nuclei_diameter,
                   cell_diameter, opt, is_2d=False):
    """Cell 8: measure the typical nucleus in the NUCLEI channel and adapt
    ``nuclei_diameter``, ``opt['scale_factor']`` and the um sizes to it
    (``opt['auto_adapt_settings']``). Updates ``opt`` in place. Returns
    ``(nuclei_diameter, zooms, cyto_factor, changes)``."""
    changes = []
    if not opt["auto_adapt_settings"]:
        print("Settings not adapted to the image (auto_adapt_settings = False).")
    elif is_2d:
        # The nucleus can't be measured in 2D; only the resolution is adapted,
        # to the given nuclei_diameter.
        from helpers.notebook_helpers import adapt_scale_2d
        opt["scale_factor"], changes = adapt_scale_2d(
            nuclei_diameter, opt["scale_factor"], voxel_size_zyx[-1])
    elif "NUCLEI" not in stain_complete_df.index:
        print("No NUCLEI channel: settings not adapted.")
    else:
        from helpers.notebook_helpers import adapt_settings_to_image
        adapted, changes = adapt_settings_to_image(
            im_original, stain_complete_df, tuple(voxel_size_zyx),
            {"nuclei_diameter": nuclei_diameter, "scale_factor": opt["scale_factor"],
             "z_bg_radius_um": opt["z_bg_radius_um"],
             "nuclei_edge_margin_um": opt["nuclei_edge_margin_um"]})
        nuclei_diameter = adapted["nuclei_diameter"]
        for k in ("scale_factor", "z_bg_radius_um", "nuclei_edge_margin_um"):
            opt[k] = adapted[k]
    zooms = [z * opt["scale_factor"] for z in opt["zoom_factors"]]
    if is_2d:
        zooms[0] = 1.0  # as initialize_dataset: never resample a single plane along Z
    cyto_factor = int(round(cell_diameter / nuclei_diameter))
    return nuclei_diameter, zooms, cyto_factor, changes


def correct_depth(im_final_stack, stain_complete_df, opt, voxel_z=None):
    """Cell 9: compensate the signal lost with depth when
    ``opt['z_atten_correction']`` (off in the validated settings)."""
    if not opt["z_atten_correction"] or im_final_stack['Original image'].shape[0] <= 2:
        return im_final_stack
    from helpers.notebook_helpers import correct_z_attenuation, plot_z_attenuation
    conds = list(stain_complete_df.index) if opt["z_atten_conditions"] == "all" else \
        [c for c in opt["z_atten_conditions"] if c in stain_complete_df.index]
    channels = [stain_complete_df.index.get_loc(c) for c in conds]
    im_final_stack['Original image'], profiles = correct_z_attenuation(
        im_final_stack['Original image'], model=opt["z_atten_model"],
        strength=opt["z_atten_strength"], channels=channels, inplace=True)
    plot_z_attenuation(profiles, names=conds, voxel_z=voxel_z)
    return im_final_stack


def remove_haze(im_final_stack, stain_complete_df, opt, voxel_size_zyx):
    """Cell 11: remove the out-of-focus haze of the NUCLEI channel slice by
    slice (``opt['z_bg_radius_um']``)."""
    if opt["z_bg_radius_um"] <= 0 or im_final_stack['Denoised image'].shape[0] <= 1 \
            or "NUCLEI" not in stain_complete_df.index:
        return im_final_stack
    from helpers.notebook_helpers import correct_z_background
    im_final_stack['Denoised image'] = correct_z_background(
        im_final_stack['Denoised image'], opt["z_bg_radius_um"], tuple(voxel_size_zyx),
        percentile=opt["z_bg_percentile"], channels=[stain_complete_df.index.get_loc("NUCLEI")])
    print(f"NUCLEI haze removed slice by slice (radius {opt['z_bg_radius_um']:.3g} um).")
    return im_final_stack


def processing_settings(opt):
    """Cells 13-15: ``sigma``, ``num_plateaus``, ``plateau_factor``,
    ``threshold_method`` and the threshold weights."""
    return {k: opt[k] for k in ("sigma", "num_plateaus", "plateau_factor", "threshold_method",
                                "global_threshold_weight", "sauvola_weight")}


def segmentation_settings(opt):
    """Cell 17: keyword arguments of `segment_nuclei`."""
    return {"nuclei_split_config": opt["nuclei_split_config"],
            **{k: opt[k] for k in _SEGMENT_KEYS}}


def cytoplasm_settings(opt):
    """Cell 18C: keyword arguments of `segment_cytoplasm`. In v1.6.4 only
    ``split_gamma`` (shared with the nuclei split); v1.6.41 adds the cell-body
    settings (``cyto_*`` keys of the settings file)."""
    out = {"split_gamma": opt["split_gamma"]}
    if "cyto_grow_um" not in opt:
        return out
    thr = opt.get("cyto_threshold")
    out.update(
        split_by_intensity_gradient=opt["cyto_split_by_intensity_gradient"],
        cyto_threshold=dict(thr) if thr else None,
        membrane_close_fraction=opt["cyto_membrane_close_fraction"],
        membrane_fill_area_factor=opt["cyto_membrane_fill_area_factor"],
        mask_close_um=opt["cyto_mask_close_um"],
        fill_holes_in_cells=opt["cyto_fill_holes"],
        grow_um=opt["cyto_grow_um"],
    )
    return out


def batch_settings(opt):
    """Cell 6: keyword arguments of `run_batch_folder` (every image gets its
    own NUCLEI contrast and size adaptation)."""
    options = {k: opt[k] for k in _BATCH_OPTION_KEYS}
    options["z_atten_model"] = opt["z_atten_model"] if opt["z_atten_correction"] else "off"
    options["auto_adapt"] = opt["auto_adapt_settings"]
    if "histogram_rules" in opt:  # v1.6.41: per image, the user's own settings kept
        keep = set(opt.get("_custom_keys", ()))
        rules = {k: r for k, r in (opt["histogram_rules"] or {}).items()
                 if k.partition(".")[0] not in keep}
        options["histogram_rules"] = rules if opt.get("adapt_to_histograms") and rules else None
    extra = {}
    if "cyto_grow_um" in opt:  # v1.6.41: the cell-body settings
        from helpers.notebook_helpers import get_cyto_split_config
        cyto = {k: v for k, v in cytoplasm_settings(opt).items() if k != "split_gamma"}
        extra["cyto_split_config"] = {**get_cyto_split_config(profile="balanced"), **cyto}
    return {**extra, "scale_factor": opt["scale_factor"], "zoom_factors": opt["zoom_factors"],
            "nuclei_split_config": opt["nuclei_split_config"],
            **{k: opt[k] for k in ("size_reference", "iterative_split", "oversize_factor",
                                   "max_split_iterations", "merge_undersized",
                                   "undersize_factor", "split_gamma", "split_gamma_step",
                                   "min_piece_fraction", "sigma", "num_plateaus",
                                   "plateau_factor", "threshold_method")},
            "nuclei_options": options}


def settings_for_report(opt, adapt_changes=()):
    """Cell 16: the settings written to the histogram Excel file."""
    skip = {"nuclei_split_config", "nuclei_split_config_overrides", "histogram_rules"}
    report = {f"Optimal setting {k}": (str(v) if isinstance(v, (list, dict)) else v)
              for k, v in opt.items() if k not in skip and not k.startswith("_")}
    report["Settings version"] = opt.get("_version")
    report["Watershed split config"] = str(opt["nuclei_split_config_overrides"])
    report["Settings adapted to the image"] = (
        "; ".join(f"{n} {o} -> {v}" for n, o, v, _ in adapt_changes) or
        ("no change" if opt["auto_adapt_settings"] else "off"))
    if "histogram_rules" in opt:  # v1.6.41
        report["Histogram rules"] = str(opt["histogram_rules"])
        report["Settings adapted to the histograms"] = (
            "; ".join(f"{n} {o} -> {v} ({s})" for n, o, v, s in opt.get("_histogram_changes", []))
            or ("no change" if opt.get("adapt_to_histograms") else "off"))
        for cond, f in (opt.get("_histogram_features") or {}).items():
            report[f"Histogram {cond}"] = ", ".join(f"{k} {v}" for k, v in f.items())
    return report
