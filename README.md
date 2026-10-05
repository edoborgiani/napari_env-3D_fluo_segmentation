# 3D Fluorescence Segmentation with Napari

This repository provides Jupyter notebooks and shared Python helpers for 3D segmentation and quantification of nuclei and fluorescence structures in microscopy images, using the Napari ecosystem. It targets researchers in bioimage analysis who need robust, reproducible workflows for volumetric fluorescence data.

> **Disclaimer:** This repository is freely available for use. The associated pipeline is currently being prepared for publication — please contact [Edoardo Borgiani](https://github.com/edoborgiani) for more information on how to use the pipeline or for collaboration enquiries.

## Features
- **Two active workflows**: nuclei segmentation (`Fluo_3D_nuc_seg_v1.6.2`, latest) and Live/Dead segmentation (`Fluo_3D_LD_seg_v1.2`, latest). The previous nuclei version (`Fluo_3D_nuc_seg_v1.6.1`) is still in the repository; older versions are kept locally in `old_v/` (not tracked in version control).
- **Interactive stain table** (nuclei workflow): after loading, pick each condition's channel and display color from drop-down menus, type the marker names and tick the intracellular markers — no hand-written `stain_dict`. The confirmed table is saved and reused on the next run.
- **Batch mode** (nuclei workflow): set `input_file` to a folder to process every image in it with the same settings; outputs go to `<folder>_output/<image name>/`. Images whose channel list differs from the first one are flagged before the batch starts.
- **Automatic nuclei splitting**: touching nuclei are split without any profile to choose — a built-in first pass (including splitting a nucleus that forks into separate islands above and below its centre), followed by a size-based refinement that re-splits oversized labels with increasingly aggressive settings and merges small fragments into their neighbour.
- **Shared helper library** (`helpers/`): processing, quantification, visualization, and report-export functions shared across notebooks.
- **Profile-aware imports** (`helpers/notebook_setup_helpers.py`): `load_nuclei_notebook_setup()` and `load_ld_notebook_setup()` load only the dependencies each workflow needs.
- **3D image processing**: normalization, resampling to isotropic voxel size, denoising, thresholding, and watershed / StarDist / Cellpose 3D segmentation.
- **Interactive ROI selection** (nuclei workflow): set `interactive_roi = True` to drag a rectangle in a napari window instead of typing pixel coordinates.
- **Automatic contrast**: set `automatic_contrast = True` to skip the interactive napari contrast/gamma viewer and pick contrast limits per channel from the histogram.
- **LD union labeling**: when no dedicated NUCLEI channel is present, `segment_nuclei()` merges all threshold channels and segments the union with the same watershed / Cellpose / StarDist choice.
- **Napari integration**: interactive visualization at each processing step.
- **Quantification & export**: per-cell marker statistics (for positive cells and for all cells), spatial and size distributions, Excel reports, 3D mesh export (VTK/STL/INP), and — nuclei workflow only — per-nucleus KDE plots and a PDF report.

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.6.2.ipynb    # Nuclei segmentation — latest recommended version
├── Fluo_3D_nuc_seg_v1.6.1.ipynb    # Nuclei segmentation — previous version
├── Fluo_3D_LD_seg_v1.2.ipynb       # Live/Dead segmentation — latest recommended version
├── requirements.txt                # Python dependencies
├── overrides.txt                   # uv --override file (see Prerequisites)
├── README.md                       # This file
└── helpers/
    ├── __init__.py
    ├── notebook_helpers.py         # Core processing, segmentation, and export functions
    └── notebook_setup_helpers.py   # Package installation and profile-aware import loader
```

> **Note:** The `old_v/` folder (containing earlier notebook versions) and Python `__pycache__` directories are excluded from version control and exist only locally.

## Getting Started

### Prerequisites

- **Python 3.10, 3.11, 3.12, 3.13, or 3.14.** `requirements.txt` pins `numpy`/`scipy`/`tensorflow`/`lxml` per Python version via markers, so one file covers all five (3.10/3.11 is the most tested range; 3.12–3.14 have been checked against current PyPI wheel availability but are less exercised in practice; **3.14 is the newest supported — avoid 3.15+**). **Python 3.14 limitation:** TensorFlow publishes no 3.14 wheels, so TensorFlow, `csbdeep` and `stardist` are not installed there and the StarDist option (`trig_stardist`) is unavailable; Cellpose and the classical methods work normally. Use Python 3.10–3.13 if you need StarDist. Likewise, `aicspylibczi` (the Zeiss `.czi` reader) has no 3.14 wheels, so it is skipped there and **`.czi` files can't be opened on 3.14** — use Python 3.10–3.13 for `.czi`, or export the images to OME-TIFF. Check your version with `python --version` (Windows/macOS) or `python3 --version` (macOS/Linux). Don't have it? Download from [python.org/downloads](https://www.python.org/downloads/).
- **Git**, to clone the repository. Don't have it? Download from [git-scm.com/downloads](https://git-scm.com/downloads).
- **[`uv`](https://github.com/astral-sh/uv)**, used in every "Install dependencies" step below instead of plain `pip`. Two independent reasons:
  - **Speed:** `requirements.txt` pulls in several large, dependency-heavy packages (napari, TensorFlow, PyTorch-based Cellpose, VTK/PyVista, SimpleITK). Plain `pip` can take a long time to resolve and download all of them; `uv` resolves and installs the same packages dramatically faster.
  - **Correctness on Python 3.13:** `aicsimageio==4.14.0` unconditionally pins `lxml<5`, which has no Python 3.13 wheels. `requirements.txt` raises that floor for 3.13, but only `uv`'s `--override` flag can force it past aicsimageio's declared cap (plain `pip` has no equivalent — it would try to build old `lxml` from source and fail without a full C++ toolchain).

  Install it once with `pip install uv`. On Python 3.10/3.11/3.12 the `--override` flag used below is a no-op (nothing to override), so the exact same command works unchanged on every supported Python version.

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

   If PowerShell reports that `python` is not recognized, use the [Python Launcher](https://docs.python.org/3/using/windows.html#launcher) instead — it ships with the official python.org installer even when `python` isn't on `PATH`: `py -3.11 -m venv .venv` (or `py -3.13 -m venv .venv` / `py -3.14 -m venv .venv`).

   > If activation fails with a message about running scripts being disabled on this system, PowerShell's execution policy is blocking it. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (in the same PowerShell window), confirm with `Y`, then re-run the activation command above.

3. **Install dependencies**
   ```powershell
   pip install uv
   uv pip install -r requirements.txt --override overrides.txt
   ```

4. **Launch Jupyter**
   ```powershell
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.2.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

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
   pip install uv
   uv pip install -r requirements.txt --override overrides.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.2.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

> **Note (Apple Silicon — M1/M2/M3):** If step 3 fails with build errors for packages like `tetgen` or `meshlib`, your Mac's ARM architecture is likely the cause. In that case, skip steps 2–3 above and use [Miniforge](https://github.com/conda-forge/miniforge) to create a conda environment instead:
> ```bash
> conda create -n napari-fluo python=3.10
> conda activate napari-fluo
> pip install uv
> uv pip install -r requirements.txt --override overrides.txt
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
   pip install uv
   uv pip install -r requirements.txt --override overrides.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.6.2.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.ipynb` for Live/Dead segmentation.

> **Note (Headless servers only):** If you are running on a remote Linux server without a physical display (e.g. an HPC cluster accessed via SSH), Napari's Qt backend will fail to open. Run the following commands **before step 4** to start a virtual framebuffer:
> ```bash
> sudo apt-get install libxcb-xinerama0 xvfb
> export DISPLAY=:99
> Xvfb :99 -screen 0 1024x768x24 &
> ```
> This is not needed on a standard desktop Linux installation.

## Usage

- Run the notebook cells in order. Each code cell has a short description right above it (**Cell N** (title)), and the notebook is divided into six sections: imports → inputs and setup → preparation → processing and segmentation → quantification → export.
- The first cell calls `load_nuclei_notebook_setup()` (or `load_ld_notebook_setup()` in the LD notebook) to import all required libraries in one step; the second calls `reload_helpers()` so edits to the helper file take effect without restarting the kernel.
- Use Napari for interactive visualization at any step.
- All shared processing logic lives in `helpers/notebook_helpers.py` — customize functions there rather than duplicating code across notebooks.

## Detailed Workflow: `Fluo_3D_nuc_seg_v1.6.2.ipynb`

### 1. Import Libraries and Helpers (Cells 1-2)
Cell 1 loads all required imports via `load_nuclei_notebook_setup()`. Cell 2 calls `reload_helpers()` to reload `helpers/notebook_helpers.py` without restarting the kernel.

### 2. Inputs and Setup (Cells 3-5)
- **Cell 3 — Settings.** The only cell to edit for a new dataset:
  - `input_file`: an image file (`.nd2`, `.tif`/`.tiff`/`.ome.tif`, `.czi`, `.lif` or `.oib`; for multi-scene `.czi`/`.lif` only the first scene is read) — or a **folder** for batch mode.
  - `ROI`, `interactive_roi`, `name_setup`, `use_setup`, `automatic_contrast`.
  - `nuclei_diameter`, `cell_diameter`, `scale_factor`, `zoom_factors`.
  - Segmentation method: `trig_cellpose`, `trig_stardist` (both `False` = watershed) and `trig_cellpose_cyto`.
  - `multilabel`, `aggregate_grow_factor`.
  - Nuclei size refinement: `size_reference`, `iterative_split`, `oversize_factor`, `max_split_iterations`, `merge_undersized`, `undersize_factor`.
- **Cell 4 — Load image and stain table.**
  - Loads the image with `initialize_dataset()`, automatically choosing lazy (chunked) or eager reading from the file's size. With `interactive_roi = True`, `select_roi_interactively()` first opens a napari window with a draggable ROI rectangle (X/Y only).
  - Prints the channel list (`print_channel_list()`). In batch mode nothing is loaded: the metadata of every image is read and any image whose channels differ from the first one is flagged.
  - Shows the **interactive stain table** (`edit_stain_dict()`, requires `ipywidgets`, installed with `jupyter`). One row per condition: NUCLEI and CYTOPLASM are fixed, other stains can be added or removed. Choose each condition's channel (`no` = not used) and color (blue, red, green, white, yellow, magenta), type the marker name, tick **Intracellular?** for markers that define the cytoplasm (`cyto_markers`), and **Membrane?** for membrane stains (`membrane_markers`, also included in `cyto_markers`): from v1.6.2 their rings are filled to get the cytoplasm, and the bright membrane is used as the border between touching cells. The two ticks exclude each other. The table is checked live (e.g. a channel used twice, a missing marker name). **Confirm** saves it to `<name_setup>_stain_dict.json`; with `use_setup = True` it is reloaded, already confirmed, on the next run.
- **Cell 5 — Read the stain table and run the batch.** Reads the confirmed table into `stain_dict` / `cyto_markers`. If it isn't confirmed yet, execution stops with a message (click **Confirm**, then run again from Cell 5). In batch mode, `run_batch_folder()` then processes every image in the folder with the whole pipeline (no viewers or plots) and stops; the image-processing parameters for the batch are set in this cell.

### 3. Preparation and Preview (Cells 6-8)
- `prepare_and_preview()` builds the image stack and the `stain_df` table (channels not used in the stain table are dropped) and opens a napari viewer for channel inspection.
- `prepare_stain_settings()` loads or creates per-channel contrast/gamma settings — reused from `<name_setup>_setup.csv` if it exists, otherwise set interactively in napari, or picked automatically from the histogram if `automatic_contrast = True`.
- Cell 8 shows the complete stain settings table for review.

### 4. Image Processing and Segmentation (Cells 9-22)
- **Preprocessing:** normalization (`run_normalize()`), isotropic resampling (`run_resample()`), median denoising (`run_denoise()`), contrast/gamma (`run_contrast_gamma()`), Gaussian smoothing (`run_smooth()`, tunable `sigma`) and histogram equalization (`run_equalize()`, tunable `num_plateaus` / `plateau_factor`).
- **Thresholding (Cell 15):**
  - Marker channels: `run_threshold()` combines a selectable global method (`threshold_method`: Otsu / median / Huang), local Sauvola thresholding and a statistical-background component (blend weights fixed internally). The histogram marks the global and final thresholds per channel. These masks decide marker positivity.
  - Cytoplasm (v1.6.2): `segment_cytoplasm()` uses the **union of the stains** thresholded by `run_threshold()` - the CYTOPLASM channel and/or the intracellular markers, plus the membrane markers with their rings closed and filled (enclosed areas up to ~1.5 cell cross-sections, `build_cytoplasm_mask()`) - and assigns exactly these voxels to the nuclei, without any growth (a cell with no stained cytoplasm keeps just its nucleus). Cells are only grown from the nuclei when there is no CYTOPLASM channel and no ticked marker at all.
- **Histogram export:** `export_channel_histograms()` saves every stage's histograms and a Parameters sheet to Excel.
- **Nuclei:** `segment_nuclei()` with watershed (default), StarDist (`trig_stardist`) or Cellpose 3D (`trig_cellpose`). Splitting is automatic — there is no split profile to choose:
  - the watershed first pass uses a fixed built-in configuration: it also splits touching nuclei where the nuclear signal dips between them, and always splits a nucleus whose z-slices show separate islands (e.g. two lobes above and below its centre);
  - then, for every method, a size-based refinement sets a reference nucleus size (`size_reference`), merges fragments smaller than `undersize_factor` × reference into a touching neighbour (only if the result is still one plausible nucleus with no dark seam at the contact), and re-splits labels bigger than `oversize_factor` × reference with increasingly aggressive settings ('aggressive' → 'very aggressive' → 'extreme'). Labels that can't be fixed are kept, never deleted.
- **Cytoplasm / PCM:** `segment_cytoplasm()` builds the cytoplasm from the CYTOPLASM channel (optionally shaped with Cellpose 3D, `trig_cellpose_cyto = True`), otherwise from the intracellular markers, otherwise by growing the nuclei; touching cells are split where the marker signal fades. `segment_pcm()` adds the pericellular matrix shell.
- **Label assignment and aggregates:** `assign_channel_labels()` links cells to marker channels; `detect_aggregates()` finds cell aggregates.
- **Visualization:** `view_processing_results()` opens napari viewers with every stage and the segmentation, including "Thresh islands" layers to check the threshold masks.

### 5. Quantification and Analysis (Cells 23-29)
Marker intensity is reported both for **positive cells** (cells that passed that marker's threshold) and for **all cells**.
- `compute_percell_marker_intensity_df()` and `build_labels_df()`: per-cell marker intensity, overlap, volume and position.
- `print_population_summary()`: counts and percentages per condition.
- `build_full_labels_df()`: the full table at original (non-resampled) resolution for export.
- `build_histogram_report()`: per-nucleus KDE plots and a PDF report.
- `plot_spatial_distributions()`, `plot_size_distributions()` and `plot_marker_intensity_clouds()` (flow-cytometry-style intensity vs cytoplasm size plot, saved as PNG).

### 6. Export (Cells 30-34)
- **3D meshes:** VTK volumes (`build_vtk_volumes()`) and per-marker STL meshes (`export_marker_stl()`) for ParaView or similar. Set `nuc_3D_export = True` in Cell 32 to also export a single cell as a VTK sub-volume.
- **Excel:** full quantification tables, stain settings and processing parameters (`export_quantification_to_excel()`).
- **FEA:** Abaqus `.inp` mesh via tetrahedralization (`export_fea_mesh()`, using `tetgen`).

---

## Detailed Workflow: `Fluo_3D_LD_seg_v1.2.ipynb`

The Live/Dead notebook follows the same helper-based structure as the nuclei notebook, adapted for two-channel viability assays (e.g. Calcein-AM / EthD).

| Aspect | Nuclei notebook | LD notebook |
|---|---|---|
| Profile | `"nuclei"` | `"ld"` (lighter imports) |
| Stain definition | Interactive stain table | `stain_dict` typed in the notebook |
| Segmentation | Watershed / StarDist / Cellpose 3D on the NUCLEI channel | Same method choice, applied to the union of all threshold channels |
| Cytoplasm / PCM | Dedicated channels + grow / Cellpose steps | Not applicable |
| NUCLEI row in `stain_complete_df` | Populated by `segment_nuclei()` | Added as empty placeholder after segmentation |
| Per-nucleus PDF report | Yes, during Quantification (`build_histogram_report()`) | Not generated |
| Export | Excel / VTK / STL / FEA | Same pipeline |

### 1. Environment Setup
Cell 1 loads all required imports in one step via `load_ld_notebook_setup()`, which applies a lighter import profile than the nuclei workflow. Cell 2 calls `reload_helpers()` to reload `helpers/notebook_helpers.py` without restarting the kernel.

### 2. Load Image Data
- Set `input_file` to your image file path (`.nd2`, `.tif`/`.tiff`/`.ome.tif`, `.czi`, `.lif` or `.oib`).
- `initialize_dataset()` loads the image — automatically choosing lazy (chunked, dask-based) or eager reading depending on the file's size, with no setting to configure — reads physical pixel sizes from metadata, and computes derived parameters for correct spatial scaling.

### 3. Define Sample & Staining Information
- Configure `stain_dict` with `LIVE` / `DEAD` channel entries — do **not** add a NUCLEI entry, since all channels are merged for segmentation.
- Set `nuclei_diameter`, `cell_diameter`, `multilabel`, and `nuclei_split_config` (via `get_nuclei_split_config()`).
- `prepare_and_preview()` builds the image stack and the `stain_df` working table, and opens a napari viewer for channel inspection.

### 4. ROI & Scaling
- Adjust `ROI` and `scale_factor` to crop or downsample for faster iteration. (Interactive ROI selection is currently a nuclei-workflow-only feature.)

### 5. Setup & Per-Channel Contrast/Gamma
- `prepare_stain_settings()` loads or creates a CSV of per-channel contrast/gamma settings — reused automatically if a matching file exists for `name_setup`, otherwise set interactively in napari, or picked automatically per channel from the histogram (peak+1 to the 99th percentile, no viewer) if `automatic_contrast = True`.

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
- `jupyter` (includes `ipywidgets`, used by the interactive stain table)
- `aicsimageio[nd2]`, `nd2reader`, `aicspylibczi` + `fsspec`, `readlif`, `oiffile` — image readers (ND2, TIFF/OME-TIFF, CZI, LIF, OIB)
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
