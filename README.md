# 3D Fluorescence Segmentation with Napari

This repository provides Jupyter notebooks and shared Python helpers for 3D segmentation and quantification of nuclei and fluorescence structures in microscopy images, using the Napari ecosystem. It targets researchers in bioimage analysis who need robust, reproducible workflows for volumetric fluorescence data.

> **Disclaimer:** This repository is freely available for use. The associated pipeline is currently being prepared for publication — please contact [Edoardo Borgiani](https://github.com/edoborgiani) for more information on how to use the pipeline or for collaboration enquiries.

## Features
- **Two active workflows**: nuclei segmentation (`Fluo_3D_nuc_seg_v1.6`, latest) and Live/Dead segmentation (`Fluo_3D_LD_seg_v1.2`, latest). Earlier versions are kept locally in `old_v/` (not tracked in version control).
- **Shared helper library** (`helpers/`): processing, quantification, visualization, and report-export functions shared across notebooks.
- **Profile-aware imports** (`helpers/notebook_setup_helpers.py`): `load_nuclei_notebook_setup()` and `load_ld_notebook_setup()` load only the dependencies each workflow needs, avoiding unnecessary overhead.
- **3D Image Processing**: normalization, resampling to isotropic voxel size, denoising, thresholding, and watershed / StarDist / Cellpose 3D segmentation.
- **Interactive ROI selection** (nuclei workflow): set `interactive_roi = True` to drag a rectangle in a napari window instead of typing pixel coordinates.
- **LD union labeling**: when no dedicated NUCLEI channel is present, `segment_nuclei()` automatically falls back to merging all threshold channels via bitwise OR and running connected-component labeling to identify individual cells; the same watershed / Cellpose / StarDist method choice as the nuclei workflow applies to the merged mask.
- **Napari Integration**: interactive visualization and manual correction at each processing step.
- **Quantification & Export**: per-cell marker statistics, spatial distributions, Excel reports, 3D mesh export (VTK/STL/INP), and — nuclei workflow only — per-nucleus KDE distribution plots and a PDF report.

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.6.ipynb      # Nuclei segmentation — latest recommended version
├── Fluo_3D_LD_seg_v1.2.ipynb       # Live/Dead segmentation — latest recommended version
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── helpers/
    ├── __init__.py
    ├── notebook_helpers.py         # Core processing, segmentation, and export functions
    └── notebook_setup_helpers.py   # Package installation and profile-aware import loader
```

> **Note:** The `old_v/` folder (containing earlier notebook versions) and Python `__pycache__` directories are excluded from version control and exist only locally.

## Getting Started

### Prerequisites

- **Python 3.10 or 3.11.** `tensorflow>=2.16,<2.17` and the `numpy==1.26.4` / `scipy==1.14.1` pins in `requirements.txt` do not support Python 3.13+, and 3.12 support is inconsistent across the pinned versions — 3.10/3.11 is the tested range (the `tetgen` pin in `requirements.txt` and the Apple Silicon `conda` example below both key off Python 3.10). Check your version with `python --version` (Windows/macOS) or `python3 --version` (macOS/Linux).
- **Git**, to clone the repository.

> **Tip — faster installs:** `requirements.txt` pulls in several large, dependency-heavy packages (napari, TensorFlow, PyTorch-based Cellpose, VTK/PyVista, SimpleITK). Plain `pip` can take a long time to resolve and download all of them. Installing [`uv`](https://github.com/astral-sh/uv) first and using it in place of `pip install` (same `requirements.txt`, no other changes needed) resolves and installs the same packages dramatically faster:
> ```
> pip install uv
> uv pip install -r requirements.txt
> ```
> The plain `pip install -r requirements.txt` commands below still work exactly the same if you'd rather not add `uv`.

### Windows

> **Important:** Run all commands below in **PowerShell**, not Command Prompt (`cmd.exe`). Look for "Windows PowerShell" or "PowerShell" in the Start menu — the icon is a dark blue console with a `>_` prompt (Command Prompt uses a plain black icon). Terminal in VS Code and Windows Terminal also default to PowerShell. The commands use PowerShell-only syntax (e.g. `.venv\Scripts\Activate.ps1`) and will fail or behave differently in `cmd.exe`.

1. **Clone the repository**
   ```powershell
   git clone https://github.com/edoborgiani/napari_env-3D_fluo_segmentation.git
   cd napari_env-3D_fluo_segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell reports that `python` is not recognized, use the [Python Launcher](https://docs.python.org/3/using/windows.html#launcher) instead — it ships with the official python.org installer even when `python` isn't on `PATH`: `py -3.11 -m venv .venv`.

   > If activation fails with a message about running scripts being disabled on this system, PowerShell's execution policy is blocking it. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (in the same PowerShell window), confirm with `Y`, then re-run the activation command above.

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```powershell
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

---

### macOS

1. **Clone the repository**
   ```bash
   git clone https://github.com/edoborgiani/napari_env-3D_fluo_segmentation.git
   cd napari_env-3D_fluo_segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

> **Note (Apple Silicon — M1/M2/M3):** If step 3 fails with build errors for packages like `tetgen` or `meshlib`, your Mac's ARM architecture is likely the cause. In that case, skip steps 2–3 above and use [Miniforge](https://github.com/conda-forge/miniforge) to create a conda environment instead:
> ```bash
> conda create -n napari-fluo python=3.10
> conda activate napari-fluo
> pip install -r requirements.txt
> ```
> Then proceed directly to step 4.

---

### Linux

1. **Clone the repository**
   ```bash
   git clone https://github.com/edoborgiani/napari_env-3D_fluo_segmentation.git
   cd napari_env-3D_fluo_segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Debian/Ubuntu, the system `python3` package often omits the `venv` module, which makes this step fail with `ensurepip is not available`. Install it first with `sudo apt-get install python3-venv` (or `python3.10-venv` / `python3.11-venv` for a non-default version), then re-run the command above.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

> **Note (Headless servers only):** If you are running on a remote Linux server without a physical display (e.g. an HPC cluster accessed via SSH), Napari's Qt backend will fail to open. Run the following commands **before step 4** to start a virtual framebuffer:
> ```bash
> sudo apt-get install libxcb-xinerama0 xvfb
> export DISPLAY=:99
> Xvfb :99 -screen 0 1024x768x24 &
> ```
> This is not needed on a standard desktop Linux installation.

## Usage

- Follow the notebook cells in order: setup → image loading → preprocessing → segmentation → quantification → export.
- The first cell calls `notebook_setup_helpers.load_nuclei_notebook_setup()` (or `load_ld_notebook_setup()` in the LD notebook) to import all required libraries in one step; the second cell calls `notebook_helpers.reload_helpers()` so edits to the helper file take effect without restarting the kernel.
- Use Napari for interactive visualization and manual corrections at any step.
- All shared processing logic lives in `helpers/notebook_helpers.py` — customize functions there rather than duplicating code across notebooks.

## Detailed Workflow: `Fluo_3D_nuc_seg_v1.6.ipynb`

### 1. Environment Setup
Cell 1 loads all required imports in one step via `load_nuclei_notebook_setup()`. Cell 2 calls `reload_helpers()` to reload `helpers/notebook_helpers.py` without restarting the kernel.

### 2. Inputs & Setup
- Set `input_file` to your `.nd2` or `.tif` file path, plus `ROI`, `name_setup`, `nuclei_diameter`/`cell_diameter`, `scale_factor`, the segmentation method flags (`trig_cellpose`, `trig_stardist`, `trig_cellpose_cyto`), and `nuclei_split_config`.
- Set `interactive_roi = True` to pick the ROI visually instead of typing coordinates — `select_roi_interactively()` opens a napari window with a draggable rectangle over the full image (X/Y only; Z stays as set in `ROI`).
- `initialize_dataset()` loads the image — automatically choosing lazy (chunked, dask-based) or eager reading depending on the file's size, with no setting to configure — reads physical pixel sizes from metadata, and computes derived parameters for correct spatial scaling.

### 3. Define Sample & Staining Information
- Configure `stain_dict` to map channel names — which must match the metadata printed after loading — to biological markers and display colors.
- `prepare_and_preview()` builds the image stack and the `stain_df` working table, and opens a napari viewer for channel inspection.

### 4. Setup & Per-Channel Contrast/Gamma
- `prepare_stain_settings()` loads or creates a CSV of per-channel contrast/gamma settings — reused automatically if a matching file exists for `name_setup`, otherwise set interactively in napari.

### 5. Image Preprocessing
- **Normalization**: channels normalized to [0, 255] via `run_normalize()`.
- **Resampling**: isotropic voxel resampling via `run_resample()`.
- **Denoising**: median filtering via `run_denoise()`.
- **Contrast/Gamma & Smoothing**: per-channel contrast/gamma (`run_contrast_gamma()`) and Gaussian smoothing (`run_smooth()`, tunable `sigma`).
- **Histogram equalization**: `run_equalize()`, tunable via `num_plateaus` / `plateau_factor`.
- **Histogram export**: per-channel histograms and a Parameters sheet saved to Excel via `export_channel_histograms()`.

### 6. Thresholding
- `run_threshold()` combines a selectable global method (Otsu / median / Huang via `threshold_method`), local Sauvola thresholding, and a statistical-background component into a combined binary mask (the blend weights between the three components are fixed internally, not user-tunable); the resulting histogram marks where the global and combined thresholds landed.

### 7. Segmentation
- **Nuclei**: 3D watershed (default, tunable via `nuclei_split_config`), StarDist2D slice-by-slice with 3D merging (`trig_stardist=True`), or Cellpose 3D (`trig_cellpose=True`) — all via `segment_nuclei()`.
- **Cytoplasm / PCM**: `segment_pcm()` grows nuclei labels into cytoplasm/PCM regions when a CYTOPLASM channel or cyto markers are defined, or shapes the cell body directly with Cellpose 3D (`trig_cellpose_cyto=True`) — independent of which method found the nuclei, so e.g. StarDist nuclei + Cellpose cell shape is a valid combination. When two touching cells' cytoplasm merges into one region, `split_by_intensity_gradient=True` (default) splits it where the marker signal fades rather than only at the geometric midpoint — tune `gradient_smooth_sigma` / `distance_weight` / `intensity_weight` / `gradient_weight` if needed.
- **Label assignment**: `assign_channel_labels()` maps segmented structures to marker channels.
- **Aggregate detection**: `detect_aggregates()` flags large multi-cell aggregates separately.

### 8. Visualization
- `view_processing_results()` opens napari overlays for raw, denoised, thresholded, and labelled images at each stage.

### 9. Quantification
- `build_labels_df()` computes per-object marker overlap, intensity, volume, and centroid position (X, Y, Z).
- `print_population_summary()` prints a population-level summary to the notebook.
- `build_full_labels_df()` builds the full quantification table at original (non-zoomed) resolution.
- `build_histogram_report()` generates per-nucleus histogram data, KDE distribution plots, and a PDF report (channel-labelled image rows rendered at their correct aspect ratio).
- `plot_spatial_distributions()` and `plot_size_distributions()` plot the segmented population's spatial and size distributions.

### 10. Export
- **Excel**: full quantification tables, stain settings, and processing parameters via `export_quantification_to_excel()`.
- **3D meshes**: VTK volumes (`build_vtk_volumes()`) and per-marker STL meshes (`export_marker_stl()`) for nuclei, cytoplasm, PCM, and markers, for visualization in ParaView or similar. Set `nuc_3D_export=True` to additionally export a single-nucleus VTK sub-volume.
- **FEA**: optional `.inp` file generated via tetrahedralization (`export_fea_mesh()`, using `tetgen`).

---

## Detailed Workflow: `Fluo_3D_LD_seg_v1.2.ipynb`

The Live/Dead notebook follows the same helper-based structure as the nuclei notebook, adapted for two-channel viability assays (e.g. Calcein-AM / EthD).

| Aspect | Nuclei notebook | LD notebook |
|---|---|---|
| Profile | `"nuclei"` | `"ld"` (lighter imports) |
| Segmentation | Watershed / StarDist / Cellpose 3D on the NUCLEI channel | Same method choice, applied to the union of all threshold channels |
| Cytoplasm / PCM | Dedicated channels + grow / Cellpose steps | Not applicable |
| NUCLEI row in `stain_complete_df` | Populated by `segment_nuclei()` | Added as empty placeholder after segmentation |
| Per-nucleus PDF report | Yes, during Quantification (`build_histogram_report()`) | Not generated |
| Export | Excel / VTK / STL / FEA | Same pipeline |

### 1. Environment Setup
Cell 1 loads all required imports in one step via `load_ld_notebook_setup()`, which applies a lighter import profile than the nuclei workflow. Cell 2 calls `reload_helpers()` to reload `helpers/notebook_helpers.py` without restarting the kernel.

### 2. Load Image Data
- Set `input_file` to your `.nd2` or `.tif` file path.
- `initialize_dataset()` loads the image — automatically choosing lazy (chunked, dask-based) or eager reading depending on the file's size, with no setting to configure — reads physical pixel sizes from metadata, and computes derived parameters for correct spatial scaling.

### 3. Define Sample & Staining Information
- Configure `stain_dict` with `LIVE` / `DEAD` channel entries — do **not** add a NUCLEI entry, since all channels are merged for segmentation.
- Set `nuclei_diameter`, `cell_diameter`, `multilabel`, and `nuclei_split_config` (via `get_nuclei_split_config()`).
- `prepare_and_preview()` builds the image stack and the `stain_df` working table, and opens a napari viewer for channel inspection.

### 4. ROI & Scaling
- Adjust `ROI` and `scale_factor` to crop or downsample for faster iteration. (Interactive ROI selection is currently a nuclei-workflow-only feature.)

### 5. Setup & Per-Channel Contrast/Gamma
- `prepare_stain_settings()` loads or creates a CSV of per-channel contrast/gamma settings — reused automatically if a matching file exists for `name_setup`, otherwise set interactively in napari.

### 6. Image Preprocessing
- **Normalization**: channels normalized to [0, 255] via `run_normalize()`.
- **Resampling**: isotropic voxel resampling via `run_resample()`.
- **Denoising**: median filtering via `run_denoise()`.
- **Contrast/Gamma & Smoothing**: per-channel contrast/gamma (`run_contrast_gamma()`) and Gaussian smoothing (`run_smooth()`, tunable `sigma`).
- **Histogram equalization**: `run_equalize()`, tunable via `num_plateaus` / `plateau_factor`.
- **Histogram export**: per-channel histograms and a Parameters sheet saved to Excel via `export_channel_histograms()`.

### 7. Thresholding
- `run_threshold()` combines a selectable global method (Otsu / median / Huang via `threshold_method`), local Sauvola thresholding, and a statistical-background component into a combined binary mask (the blend weights between the three components are fixed internally, not user-tunable).

### 8. Segmentation
- **Cells**: `segment_nuclei()` — watershed (default), Cellpose 3D (`trig_cellpose=True`), or StarDist (`trig_stardist=True`) — applied to the union of all threshold channels merged via bitwise OR, with connected-component labeling identifying individual cells.
- A NUCLEI placeholder row is added to `stain_complete_df` after segmentation so downstream helpers work correctly.
- `assign_channel_labels()` maps LIVE / DEAD intensity into the segmented objects.

### 9. Visualization
- `view_processing_results()` opens napari overlays for raw, denoised, thresholded, and labelled images at each stage.

### 10. Quantification
- `build_labels_df()` computes per-object marker overlap, intensity, volume, and centroid position (X, Y, Z).
- `print_population_summary()` prints LIVE/DEAD counts and percentages.
- `build_full_labels_df()` builds the full quantification table at original (non-zoomed) resolution.
- `plot_spatial_distributions()` and `plot_size_distributions()` plot the LIVE/DEAD population's spatial and size distributions. Unlike the nuclei notebook, this workflow does not generate a per-nucleus PDF report.

### 11. Export
- **Excel**: full quantification tables via `export_quantification_to_excel()`.
- **3D meshes**: VTK volumes (`build_vtk_volumes()`) and per-marker STL meshes (`export_marker_stl()`) for segmented cells and markers, for visualization in ParaView or similar.
- **FEA**: optional `.inp` file generated via tetrahedralization (`export_fea_mesh()`, using `tetgen`).

---

## Requirements
See `requirements.txt` for the full list. Key dependencies:
- `napari[all]`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`
- `aicsimageio[nd2]`, `nd2reader`
- `tensorflow`, `csbdeep`, `stardist`, `cellpose` — segmentation models
- `pyvista`, `SimpleITK`
- `meshio`, `tetgen`, `meshlib`
- `xlsxwriter`, `reportlab`, `Pillow`

## Contributing
Contributions are welcome. Please open issues or pull requests for bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Napari team and contributors
- scikit-image, numpy, scipy, and matplotlib communities

---
For questions or support, please open an issue on GitHub.
