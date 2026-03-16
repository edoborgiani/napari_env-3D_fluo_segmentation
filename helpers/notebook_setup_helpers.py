"""Shared setup helpers for segmentation notebooks."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any


DEFAULT_PIP_PACKAGES = [
    "aicsimageio[nd2]",
    "nd2reader",
    "xlsxwriter",
    "reportlab",
]


def install_required_packages(extra_packages: list[str] | None = None) -> None:
    """Install required notebook packages with the active Python executable."""
    packages = list(DEFAULT_PIP_PACKAGES)
    if extra_packages:
        packages.extend(extra_packages)

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("Package installation complete.")


def load_common_imports(
    profile: str = "nuclei",
    enable_napari_interactive: bool = True,
) -> dict[str, Any]:
    """Import libraries for a notebook profile and return them as globals-ready dict."""
    imported: dict[str, Any] = {}

    imported["Path"] = importlib.import_module("pathlib").Path
    imported["os"] = importlib.import_module("os")
    imported["st"] = importlib.import_module("statistics")
    imported["cv2"] = importlib.import_module("cv2")
    imported["napari"] = importlib.import_module("napari")
    imported["pd"] = importlib.import_module("pandas")
    imported["pv"] = importlib.import_module("pyvista") if profile == "nuclei" else None
    imported["np"] = importlib.import_module("numpy")
    imported["plt"] = importlib.import_module("matplotlib.pyplot")
    imported["mcolors"] = importlib.import_module("matplotlib.colors") if profile == "nuclei" else None
    imported["sitk"] = importlib.import_module("SimpleITK") if profile == "nuclei" else None
    imported["skimage"] = importlib.import_module("skimage")

    imported["ndi"] = importlib.import_module("scipy.ndimage")
    imported["gaussian_kde"] = importlib.import_module("scipy.stats").gaussian_kde if profile == "nuclei" else None
    imported["label"] = importlib.import_module("scipy.ndimage").label
    imported["zoom"] = importlib.import_module("scipy.ndimage").zoom
    imported["binary_dilation"] = (
        importlib.import_module("scipy.ndimage").binary_dilation if profile == "nuclei" else None
    )
    imported["generate_binary_structure"] = (
        importlib.import_module("scipy.ndimage").generate_binary_structure if profile == "nuclei" else None
    )

    imported["filters"] = importlib.import_module("skimage.filters")
    imported["morphology"] = importlib.import_module("skimage.morphology")
    imported["peak_local_max"] = importlib.import_module("skimage.feature").peak_local_max if profile == "nuclei" else None
    imported["watershed"] = importlib.import_module("skimage.segmentation").watershed
    imported["relabel_sequential"] = (
        importlib.import_module("skimage.segmentation").relabel_sequential if profile == "nuclei" else None
    )
    imported["ball"] = importlib.import_module("skimage.morphology").ball if profile == "nuclei" else None
    imported["threshold_otsu"] = importlib.import_module("skimage.filters").threshold_otsu if profile == "nuclei" else None
    imported["threshold_niblack"] = (
        importlib.import_module("skimage.filters").threshold_niblack if profile == "nuclei" else None
    )
    imported["threshold_sauvola"] = importlib.import_module("skimage.filters").threshold_sauvola
    imported["regionprops"] = importlib.import_module("skimage.measure").regionprops if profile == "nuclei" else None

    imported["combinations"] = importlib.import_module("itertools").combinations if profile == "nuclei" else None
    imported["defaultdict"] = importlib.import_module("collections").defaultdict if profile == "nuclei" else None
    imported["AICSImage"] = importlib.import_module("aicsimageio").AICSImage
    imported["ND2Reader"] = importlib.import_module("nd2reader").ND2Reader
    imported["Colormap"] = importlib.import_module("vispy.color").Colormap if profile == "nuclei" else None
    imported["to_rgb"] = importlib.import_module("matplotlib.colors").to_rgb if profile == "nuclei" else None
    imported["normalize"] = importlib.import_module("csbdeep.utils").normalize if profile == "nuclei" else None
    imported["StarDist2D"] = importlib.import_module("stardist.models").StarDist2D if profile == "nuclei" else None
    imported["mr"] = importlib.import_module("meshlib.mrmeshpy") if profile == "nuclei" else None
    imported["mrn"] = importlib.import_module("meshlib.mrmeshnumpy") if profile == "nuclei" else None
    imported["clear_output"] = importlib.import_module("IPython.display").clear_output if profile == "nuclei" else None
    imported["meshio"] = importlib.import_module("meshio") if profile == "nuclei" else None
    imported["tetgen"] = importlib.import_module("tetgen") if profile == "nuclei" else None
    imported["xlsxwriter"] = importlib.import_module("xlsxwriter") if profile == "nuclei" else None
    imported["PILImage"] = importlib.import_module("PIL.Image").Image if profile == "nuclei" else None

    if profile == "nuclei":
        reportlab_platypus = importlib.import_module("reportlab.platypus")
        imported["RLImage"] = reportlab_platypus.Image
        imported["SimpleDocTemplate"] = reportlab_platypus.SimpleDocTemplate
        imported["Image"] = reportlab_platypus.Image
        imported["Paragraph"] = reportlab_platypus.Paragraph
        imported["Spacer"] = reportlab_platypus.Spacer
        imported["Table"] = reportlab_platypus.Table
        imported["TableStyle"] = reportlab_platypus.TableStyle
        imported["PageBreak"] = reportlab_platypus.PageBreak

        reportlab_pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        imported["A4"] = reportlab_pagesizes.A4
        imported["inch"] = importlib.import_module("reportlab.lib.units").inch
        imported["getSampleStyleSheet"] = importlib.import_module("reportlab.lib.styles").getSampleStyleSheet
        imported["colors"] = importlib.import_module("reportlab.lib.colors")
    else:
        imported["RLImage"] = None
        imported["SimpleDocTemplate"] = None
        imported["Image"] = None
        imported["Paragraph"] = None
        imported["Spacer"] = None
        imported["Table"] = None
        imported["TableStyle"] = None
        imported["PageBreak"] = None
        imported["A4"] = None
        imported["inch"] = None
        imported["getSampleStyleSheet"] = None
        imported["colors"] = None

    get_settings = importlib.import_module("napari.settings").get_settings
    imported["get_settings"] = get_settings
    settings = get_settings()
    imported["settings"] = settings

    if enable_napari_interactive:
        settings.application.ipy_interactive = True

    return imported
