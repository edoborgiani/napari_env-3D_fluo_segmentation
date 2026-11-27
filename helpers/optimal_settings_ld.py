"""Validated settings for the LIVE/DEAD notebook (NucleoSeg3D_LD_v1.2.2.ipynb).

The LIVE/DEAD notebook uses the validated settings of the nuclei notebook,
read from ``settings/optimal_settings_v<version>.json``
(`helpers.optimal_settings.load_optimal_settings`, with ``custom_settings``
on top). ``version='best'`` (default) takes the best-scoring version
(`helpers.optimal_settings.best_version`, v1.6.41 at present); ``'1.6.4'``
gives the settings the notebook used before. In the nuclei notebook the
per-image steps act on the NUCLEI channel; here the nuclei are segmented
from the merged **nuclear** markers (e.g. DEAD = EthD-1 / PI, a Hoechst
counterstain), so:

- the contrast of every nuclear marker is set from its own histogram
  (calibrated rule, with gamma ``nuclei_gamma``: 1.15 in v1.6.41);
- the NUCLEI histogram rules (v1.6.41: the nuclei edge fraction from the
  signal-to-noise ratio) read the merged nuclear markers
  (`adapt_to_histograms`);
- the typical nucleus is measured in the nuclear markers and
  ``nuclei_diameter``, ``scale_factor`` and the um sizes adapted to it;
- the haze removal (and the depth correction, off in the validated
  settings) acts on the nuclear markers;
- the nuclei get the validated segmentation steps (watershed split
  configuration, split gamma, size refinement, Z elongation, flare trim,
  saddle / tall split, edges); the rescue of missed nuclei is off
  (`LD_OVERRIDES`);
- the cell bodies (cytoplasmic markers, e.g. LIVE = calcein) get the
  validated smoothing, threshold, watershed split configuration, split gamma
  and size refinement. The steps after the watershed were validated on
  nuclei only and are not applied to them, and neither are the v1.6.41
  cell-body settings (``cyto_*``, tuned on the cytoplasm of the nuclei
  notebook).

The LIVE/DEAD call itself (``min_positive_fraction``, ``ld_rule``) is an
experiment choice, set in Cell 3.
"""

from helpers.optimal_settings import (  # noqa: F401  (re-exported for the notebook)
    SETTINGS_FILE, _SEGMENT_KEYS, _BATCH_OPTION_KEYS, load_optimal_settings,
    settings_for_report, best_version)

# Validated values that do not fit the LIVE/DEAD analysis. The rescue of
# missed nuclei adds nuclei the threshold left out; in the LIVE/DEAD notebook
# positivity is read from the threshold masks, so a rescued nucleus is
# positive for no marker (no LIVE/DEAD call), and it would undo the
# bright-signal filter (`ld_binary_threshold`). custom_settings still
# override these.
LD_OVERRIDES = {"nuclei_rescue_contrast": 0.0}


def load_ld_settings(custom_settings=None, verbose=True, version="best"):
    """The validated settings of ``version`` (``'best'`` = the best-scoring
    settings file, or e.g. ``'1.6.4'``) with the LIVE/DEAD changes of
    ``LD_OVERRIDES``, then ``custom_settings`` on top.

    A ``scale_factor`` given in ``custom_settings`` is used as given
    (``opt['scale_factor_fixed']``): the adaptation to the image (Cell 8,
    batch mode) still measures the nucleus but no longer changes the
    resolution. Without it the resolution is adapted as before, so that a
    nucleus spans 12-19 voxels."""
    custom = dict(custom_settings or {})
    opt = load_optimal_settings({**LD_OVERRIDES, **custom}, verbose=verbose, version=version)
    # only what the user set is kept by the histogram rules (not LD_OVERRIDES)
    opt["_custom_keys"] = sorted(custom)
    opt["scale_factor_fixed"] = "scale_factor" in custom
    if verbose:
        changed = {k: v for k, v in LD_OVERRIDES.items() if k not in custom}
        if changed:
            print("LIVE/DEAD: " + ", ".join(f"{k} = {v!r}" for k, v in changed.items())
                  + " (rescue of missed nuclei off: positivity is read from the "
                  "threshold masks).")
        if opt["scale_factor_fixed"]:
            print(f"scale_factor = {opt['scale_factor']:g} from custom_settings: used as given "
                  "(not adapted to the image).")
    return opt


# Size refinement / splitting settings shared by the nuclei and the cell bodies.
_SHARED_SEGMENT_KEYS = (
    "size_reference", "iterative_split", "oversize_factor", "max_split_iterations",
    "merge_undersized", "undersize_factor", "split_gamma", "split_gamma_step",
    "min_piece_fraction")


def set_nuclear_contrast(im_original, stain_complete_df, original_stain_complete_df, opt,
                         nuclear_conditions):
    """Cell 8: contrast / gamma of every nuclear marker from its own histogram
    (``opt['nuclei_contrast'] == 'calibrated'``, gamma ``opt['nuclei_gamma']``),
    in both stain tables."""
    if opt["nuclei_contrast"] != "calibrated" or not nuclear_conditions:
        return None
    from helpers.notebook_helpers import ld_calibrated_nuclear_contrast
    return ld_calibrated_nuclear_contrast(im_original, stain_complete_df,
                                          original_stain_complete_df, nuclear_conditions,
                                          gamma=opt.get("nuclei_gamma", 1.0))


def adapt_to_histograms(im_original, stain_complete_df, opt, nuclear_conditions):
    """Cell 8 (v1.6.41 settings): the NUCLEI histogram rules
    (``opt['histogram_rules']``), read on the merged nuclear markers, when
    ``opt['adapt_to_histograms']``. Settings given in ``custom_settings``
    are kept. Updates ``opt`` in place; returns ``[(setting, old, new, feature)]``.
    Nothing changes with the v1.6.4 settings (no rules)."""
    opt["_histogram_changes"] = []
    if not opt.get("adapt_to_histograms") or not opt.get("histogram_rules"):
        return []
    from helpers.notebook_helpers import ld_adapt_to_histograms
    changes, feats = ld_adapt_to_histograms(
        opt, opt["histogram_rules"], im_original, stain_complete_df, nuclear_conditions,
        keep=opt.get("_custom_keys", ()))
    opt["_histogram_features"] = feats
    opt["_histogram_changes"] = changes
    nuc = feats.get("NUCLEI")
    if nuc:
        print(f"Nuclear markers (merged) histogram: background {nuc['background']:.1f}, "
              f"noise {nuc['noise']:.2f}, signal-to-noise {nuc['snr']:.1f} (0-255 scale).")
    for name, old, new, src in changes:
        print(f"Adapted to the histogram ({src.replace('NUCLEI', 'nuclear markers')}): "
              f"{name} {old} -> {new}")
    return changes


def adapt_to_image(im_original, stain_complete_df, voxel_size_zyx, nuclei_diameter, opt,
                   nuclear_conditions, is_2d=False):
    """Cell 8: measure the typical nucleus in the nuclear markers and adapt
    ``nuclei_diameter``, ``opt['scale_factor']`` and the um sizes to it
    (``opt['auto_adapt_settings']``). Updates ``opt`` in place. Returns
    ``(nuclei_diameter, zooms, changes)``."""
    changes = []
    fixed_scale = bool(opt.get("scale_factor_fixed"))
    if not opt["auto_adapt_settings"]:
        print("Settings not adapted to the image (auto_adapt_settings = False).")
    elif is_2d:
        # The nucleus can't be measured in 2D; only the resolution is adapted,
        # to the given nuclei_diameter.
        if fixed_scale:
            print(f"2D image: scale_factor {opt['scale_factor']:g} from custom_settings kept.")
        else:
            from helpers.notebook_helpers import adapt_scale_2d
            opt["scale_factor"], changes = adapt_scale_2d(
                nuclei_diameter, opt["scale_factor"], voxel_size_zyx[-1])
    elif not nuclear_conditions:
        print("No nuclear marker: settings not adapted to the image.")
    else:
        from helpers.notebook_helpers import ld_adapt_settings_to_image
        adapted, changes = ld_adapt_settings_to_image(
            im_original, stain_complete_df, tuple(voxel_size_zyx), nuclear_conditions,
            {"nuclei_diameter": nuclei_diameter, "scale_factor": opt["scale_factor"],
             "z_bg_radius_um": opt["z_bg_radius_um"],
             "nuclei_edge_margin_um": opt["nuclei_edge_margin_um"]},
            adapt_scale=not fixed_scale)
        if fixed_scale:
            print(f"scale_factor {opt['scale_factor']:g} from custom_settings kept (a nucleus "
                  f"spans {adapted['nuclei_diameter'] * opt['scale_factor'] / voxel_size_zyx[-1]:.1f} "
                  "voxels; the validated settings were tuned at 12-19).")
        nuclei_diameter = adapted["nuclei_diameter"]
        for k in ("scale_factor", "z_bg_radius_um", "nuclei_edge_margin_um"):
            opt[k] = adapted[k]
    zooms = [z * opt["scale_factor"] for z in opt["zoom_factors"]]
    if is_2d:
        zooms[0] = 1.0  # as initialize_dataset: never resample a single plane along Z
    return nuclei_diameter, zooms, changes


def correct_depth(im_final_stack, stain_complete_df, opt, nuclear_conditions, voxel_z=None):
    """Cell 9: compensate the signal lost with depth when
    ``opt['z_atten_correction']`` (off in the validated settings);
    'NUCLEI' in ``opt['z_atten_conditions']`` = every nuclear marker."""
    if not opt["z_atten_correction"] or im_final_stack['Original image'].shape[0] <= 2:
        return im_final_stack
    from helpers.notebook_helpers import ld_correct_depth, plot_z_attenuation
    im_final_stack['Original image'], profiles, conds = ld_correct_depth(
        im_final_stack['Original image'], stain_complete_df, nuclear_conditions,
        opt["z_atten_conditions"], model=opt["z_atten_model"], strength=opt["z_atten_strength"])
    if profiles is not None:
        plot_z_attenuation(profiles, names=conds, voxel_z=voxel_z)
    return im_final_stack


def remove_haze(im_final_stack, stain_complete_df, opt, voxel_size_zyx, nuclear_conditions):
    """Cell 11: remove the out-of-focus haze of the nuclear markers slice by
    slice (``opt['z_bg_radius_um']``)."""
    from helpers.notebook_helpers import ld_remove_nuclear_haze
    im_final_stack['Denoised image'] = ld_remove_nuclear_haze(
        im_final_stack['Denoised image'], stain_complete_df, nuclear_conditions,
        opt["z_bg_radius_um"], tuple(voxel_size_zyx), percentile=opt["z_bg_percentile"])
    return im_final_stack


def nuclei_settings(opt):
    """Cell 17: keyword arguments of `segment_ld_nuclei` (every validated
    NUCLEI segmentation setting)."""
    return {"nuclei_split_config": opt["nuclei_split_config"],
            **{k: opt[k] for k in _SEGMENT_KEYS}}


def cell_body_settings(opt):
    """Cell 18: keyword arguments of `segment_ld_cells` (watershed split
    configuration, split gamma and size refinement)."""
    return {"nuclei_split_config": opt["nuclei_split_config"],
            **{k: opt[k] for k in _SHARED_SEGMENT_KEYS}}


def ld_batch_settings(opt):
    """Cell 6: keyword arguments of `run_ld_batch_folder` (every image gets
    its own nuclear-marker contrast, histogram rules and size adaptation)."""
    options = {k: opt[k] for k in _BATCH_OPTION_KEYS}
    options["z_atten_model"] = opt["z_atten_model"] if opt["z_atten_correction"] else "off"
    options["auto_adapt"] = opt["auto_adapt_settings"]
    options["adapt_scale"] = not opt.get("scale_factor_fixed", False)
    if opt.get("histogram_rules"):  # v1.6.41 settings: per image, the user's own settings kept
        keep = set(opt.get("_custom_keys", ()))
        rules = {k: r for k, r in opt["histogram_rules"].items()
                 if k.partition(".")[0] not in keep}
        options["histogram_rules"] = rules if opt.get("adapt_to_histograms") and rules else None
    return {"scale_factor": opt["scale_factor"], "zoom_factors": opt["zoom_factors"],
            "nuclei_split_config": opt["nuclei_split_config"],
            **{k: opt[k] for k in _SHARED_SEGMENT_KEYS},
            **{k: opt[k] for k in ("sigma", "num_plateaus", "plateau_factor",
                                   "threshold_method")},
            "nuclei_options": options}
