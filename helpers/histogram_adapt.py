"""v1.6.41: settings adapted to the conditions read off each channel's histogram.

Every channel of the image is described by a few numbers read from its own
intensity histogram, on the same 0-255 min-max scale the stain table's
``Cont_min`` / ``Cont_max`` refer to (`channel_histogram_features`):

- ``background``: the median (most voxels of a fluorescence stack are
  background), ``noise``: 1.4826 x the median absolute deviation around it;
- ``p99`` / ``p99_9`` / ``p99_95``: the bright tail (nuclei, stained cells);
- ``snr``: (p99.95 - background) / noise, how far the brightest structures
  stand out of the noise;
- ``signal_range``: (p99.95 - background) / 255, how much of the scale the
  signal uses (low = a few hot voxels squeezed the min-max stretch);
- ``bright_fraction``: fraction of voxels above background + half the
  signal, roughly the volume fraction of the stained objects;
- ``tail_ratio``: (p99 - background) / (p99.95 - background), the shape of
  the bright tail (near 1 = many voxels as bright as the brightest; low = a
  few very bright objects over many dim ones);
- ``saturated_fraction``: fraction of voxels at the channel's maximum.

The rules (``rules`` block of ``optimal_settings_v<version>.json``) turn
these features into settings: ``value = clip(a + b x feature, lo, hi)``,
fitted by ``validation/fit_adaptive_rules.py`` on the ground truths and kept
only where a rule fitted without an image predicted a better value for it
than the one shared value (leave one image out). `adapt_to_histograms`
applies them in Cell 8, before the nucleus size is measured.
"""

import numpy as np

FEATURES = ("background", "noise", "p99", "p99_9", "p99_95", "snr", "signal_range",
            "bright_fraction", "tail_ratio", "saturated_fraction")


def channel_histogram_features(channel):
    """Histogram features of one channel (Z, Y, X) or (Y, X); see the module
    docstring. The stack is subsampled above 4 M voxels (as the calibrated
    contrast rule). The values are on the 0-255 min-max scale of the stain
    table but not rounded to 8 bits (the noise of a 16-bit stack is often
    below one 8-bit grey level)."""
    raw = np.asarray(channel)
    sub = raw[::2, ::2, ::2] if raw.ndim == 3 and raw.size > 4_000_000 else raw
    lo, hi = float(raw.min()), float(raw.max())
    s = ((sub.astype(np.float32).ravel() - lo) / max(hi - lo, 1e-9)) * 255.0
    bg = float(np.median(s))
    noise = 1.4826 * float(np.median(np.abs(s - bg)))
    p99, p99_9, p99_95 = (float(v) for v in np.percentile(s, [99, 99.9, 99.95]))
    signal = max(p99_95 - bg, 1e-6)
    return {
        "background": round(bg, 3),
        "noise": round(noise, 3),
        "p99": round(p99, 3), "p99_9": round(p99_9, 3), "p99_95": round(p99_95, 3),
        "snr": round(signal / max(noise, 1e-3), 3),
        "signal_range": round(signal / 255.0, 4),
        "bright_fraction": round(float(np.mean(s > bg + 0.5 * signal)), 5),
        "tail_ratio": round(float(np.clip((p99 - bg) / signal, 0, 1)), 4),
        "saturated_fraction": round(float(np.mean(sub >= hi)), 6),
    }


def histogram_table(image, stain_complete_df):
    """One row of `channel_histogram_features` per channel of ``image``
    (Z, Y, X, C; channels in the order of ``stain_complete_df``)."""
    import pandas as pd
    rows = []
    for c, idx in enumerate(stain_complete_df.index):
        cond = idx[0] if isinstance(idx, tuple) else idx
        rows.append({"Condition": cond, **channel_histogram_features(np.asarray(image)[..., c])})
    return pd.DataFrame(rows).set_index("Condition")


def apply_rule(rule, features):
    """``clip(a + b x features[rule['feature']], lo, hi)`` (rounded to
    ``rule['round']`` decimals)."""
    x = float(features[rule["feature"]])
    if rule.get("log"):
        x = float(np.log10(max(x, 1e-9)))
    v = rule["a"] + rule["b"] * x
    v = float(np.clip(v, rule["lo"], rule["hi"]))
    return round(v, int(rule.get("round", 3)))


def adapted_values(rules, features_by_condition):
    """``{setting: value}`` of every rule whose channel is present.
    ``rules``: ``{setting: {channel, feature, a, b, lo, hi, ...}}``;
    ``features_by_condition``: ``{condition: features}``. The rule's
    ``channel`` is a condition name, or ``'cyto'`` for the mean of the
    features of the cytoplasm / membrane channels."""
    out = {}
    for setting, rule in (rules or {}).items():
        ch = rule.get("channel", "NUCLEI")
        if ch == "cyto":
            feats = [f for c, f in features_by_condition.items() if c != "NUCLEI"]
            if not feats:
                continue
            f = {k: float(np.mean([x[k] for x in feats])) for k in feats[0]}
        elif ch in features_by_condition:
            f = features_by_condition[ch]
        else:
            continue
        out[setting] = apply_rule(rule, f)
    return out


def get_setting(settings, name):
    """``settings[name]``; ``'a.b'`` reads ``settings['a']['b']``."""
    head, _, key = name.partition(".")
    if key:
        return (settings.get(head) or {}).get(key)
    return settings.get(name)


def set_setting(settings, name, value):
    """``settings[name] = value``; ``'a.b'`` sets ``settings['a']['b']`` (in a
    copy of ``settings['a']``, so a dict shared with other images is not
    changed)."""
    head, _, key = name.partition(".")
    if key:
        settings[head] = {**(settings.get(head) or {}), key: value}
    else:
        settings[name] = value


def adapt_settings(settings, rules, features_by_condition, keep=()):
    """Apply the ``rules`` to ``settings`` (in place). Settings named in
    ``keep`` (set by the user) and settings missing from ``settings`` are
    left as they are. Returns ``[(name, old, new, rule description)]``."""
    changes = []
    for name, value in adapted_values(rules, features_by_condition).items():
        head = name.partition(".")[0]
        if name in keep or head in keep or head not in settings:
            continue
        old = get_setting(settings, name)
        if old is None:
            continue
        set_setting(settings, name, value)
        r = rules[name]
        x = r["feature"] if not r.get("log") else f"log10({r['feature']})"
        changes.append((name, old, value, f"{r.get('channel', 'NUCLEI')} {x}"))
    return changes


def features_by_condition(image, stain_complete_df):
    """``{condition: features}`` of every channel (`histogram_table`)."""
    table = histogram_table(image, stain_complete_df)
    return {c: row.to_dict() for c, row in table.iterrows()}
