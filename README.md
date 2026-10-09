# 3D Fluorescence Segmentation with Napari

This repository provides Jupyter notebooks and shared Python helpers for the segmentation and quantification of nuclei and fluorescence structures in 3D (and 2D) microscopy images, using the Napari ecosystem. It targets researchers in bioimage analysis who need robust, reproducible workflows for volumetric fluorescence data.

> **Disclaimer:** This repository is freely available for use. The associated pipeline is currently being prepared for publication — please contact [Edoardo Borgiani](https://github.com/edoborgiani) for more information on how to use the pipeline or for collaboration enquiries.

## Features
- **Two active workflows**: nuclei segmentation (`Fluo_3D_nuc_seg_v1.6.3`, latest) and Live/Dead segmentation with a binary LIVE/DEAD call per cell (`Fluo_3D_LD_seg_v1.2.1`, latest). The previous versions (`Fluo_3D_nuc_seg_v1.6.2`, `Fluo_3D_nuc_seg_v1.6.1`, `Fluo_3D_LD_seg_v1.2`) are still in the repository; older versions are kept locally in `old_v/` (not tracked in version control).
- **2D and 3D images**: a file without a Z stack (or a ROI with a single Z slice) is detected automatically and processed as a 2D image — segmented in the image plane, with every size reported as an area (µm²) instead of a volume; the 3D-only exports are skipped. Plain TIFFs (no OME metadata) are read by their dimensions: 3 = a 2D image with channels, 4 = a 3D stack with channels.
- **Interactive stain table**: after loading, pick each condition's channel and display color from drop-down menus and type the marker names — no hand-written `stain_dict`. The nuclei workflow asks which markers form the cytoplasm (intracellular / membrane), the LD workflow whether each marker is **nuclear** or **cytoplasmic**. The confirmed table is saved and reused on the next run, and also names and colors the channels of the interactive ROI preview.
- **Batch mode**: set `input_file` to a folder to process every image in it with the same settings; outputs go to `<folder>_output/<image name>/`. Images whose channel list differs from the first one are flagged before the batch starts. 2D and 3D images can be mixed.
- **Automatic splitting of touching nuclei and cells**: a built-in first pass (including splitting a nucleus that forks into separate islands above and below its centre), then a size-based refinement that re-splits oversized labels with increasingly aggressive settings and merges small fragments into their neighbour. From nuclei v1.6.3 / LD v1.2.1:
  - **split gamma**: a gamma > 1 on the intensity used to split makes the dim seams between touching objects darker faster than the signal, so the split zones are more distinct; it grows at every re-split level;
  - **second merge round of split pieces**: slivers of a few pixels/voxels left by a split are merged into the closest big piece of the same split.
- **Shared helper library** (`helpers/`): processing, quantification, visualization, and report-export functions shared across notebooks.
- **Profile-aware imports** (`helpers/notebook_setup_helpers.py`): `load_nuclei_notebook_setup()` and `load_ld_notebook_setup()` load the dependencies each workflow needs.
- **Image processing**: normalization, resampling to isotropic voxel size, denoising, thresholding, and watershed / StarDist / Cellpose segmentation (3D, or 2D on a 2D image).
- **Interactive ROI selection**: set `interactive_roi = True` to drag a rectangle in a napari window instead of typing pixel coordinates.
- **Automatic contrast**: set `automatic_contrast = True` to skip the interactive napari contrast/gamma viewer and pick contrast limits per channel from the histogram.
- **Napari integration**: interactive visualization at each processing step.
- **Quantification & export**: per-cell marker statistics (for positive cells and for all cells), spatial and size distributions, Excel reports, per-cell KDE plots and a PDF report, and 3D mesh export (VTK/STL/INP, 3D images only).

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.6.3.ipynb    # Nuclei segmentation — latest recommended version
├── Fluo_3D_nuc_seg_v1.6.2.ipynb    # Nuclei segmentation — previous version
├── Fluo_3D_nuc_seg_v1.6.1.ipynb    # Nuclei segmentation — older version
├── Fluo_3D_LD_seg_v1.2.1.ipynb     # Live/Dead segmentation — latest recommended version
├── Fluo_3D_LD_seg_v1.2.ipynb       # Live/Dead segmentation — previous version
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
   Open `Fluo_3D_nuc_seg_v1.6.3.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.1.ipynb` for Live/Dead segmentation.

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
   Open `Fluo_3D_nuc_seg_v1.6.3.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.1.ipynb` for Live/Dead segmentation.

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
   Open `Fluo_3D_nuc_seg_v1.6.3.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.2.1.ipynb` for Live/Dead segmentation.

> **Note (Headless servers only):** If you are running on a remote Linux server without a physical display (e.g. an HPC cluster accessed via SSH), Napari's Qt backend will fail to open. Run the following commands **before step 4** to start a virtual framebuffer:
> ```bash
> sudo apt-get install libxcb-xinerama0 xvfb
> export DISPLAY=:99
> Xvfb :99 -screen 0 1024x768x24 &
> ```
> This is not needed on a standard desktop Linux installation.

## Usage

- Run the notebook cells in order. Each code cell has a short description right above it (**Cell N** (title)), and the notebook is divided into six sections: imports → inputs and setup → preparation → processing and segmentation → quantification → export.
- The first cell calls `load_nuclei_notebook_setup()` to import all required libraries in one step; the second calls `reload_helpers()` so edits to the helper file take effect without restarting the kernel.
- Use Napari for interactive visualization at any step.
- All shared processing logic lives in `helpers/notebook_helpers.py` — customize functions there rather than duplicating code across notebooks.

### 2D images
Both notebooks handle 2D images (a file without a Z stack, or a ROI with a single Z slice) automatically; Cell 4 prints whether the image is 2D or 3D and sets `is_2d`.
- **Loading:** no Z voxel size is asked — the plane is given a 1 µm thickness, so every "volume" the pipeline computes is the area in µm². Resampling never adds planes along Z.
- **Reading plain TIFFs:** a TIFF without OME metadata is read from its own dimensions (singleton axes dropped): 2 = one 2D channel, 3 = a 2D image with channels, 4 = a 3D stack with channels (the smaller of the two extra axes is taken as the channel axis). Channel axes stated by the file (ImageJ hyperstack, RGB samples) are used as they are. OME-TIFFs go through aicsimageio as before.
- **Processing:** median, Gaussian and Sauvola filters run in 2D on the plane.
- **Segmentation:** nuclei (and LD cell bodies) are segmented in the image plane — a watershed seeded at every bright centre separated from its neighbours by an intensity dip (so touching objects that form one round blob are still split), or Cellpose / StarDist in 2D mode — followed by the same size refinement on areas. The cytoplasm is built in the plane (membrane rings closed with a disk, Cellpose in 2D mode).
- **Outputs:** every size is an area (`[um2]` columns, "area" in the summaries, plots and Excel report); the spatial plots show X and Y only; the VTK, STL, single-cell VTK and FEA exports are skipped.

## Detailed Workflow: `Fluo_3D_nuc_seg_v1.6.3.ipynb`

### 1. Import Libraries and Helpers (Cells 1-2)
Cell 1 loads all required imports via `load_nuclei_notebook_setup()`. Cell 2 calls `reload_helpers()` to reload `helpers/notebook_helpers.py` without restarting the kernel.

### 2. Inputs and Setup (Cells 3-5)
- **Cell 3 — Settings.** The only cell to edit for a new dataset:
  - `input_file`: an image file (`.nd2`, `.tif`/`.tiff`/`.ome.tif`, `.czi`, `.lif` or `.oib`; for multi-scene `.czi`/`.lif` only the first scene is read) — or a **folder** for batch mode.
  - `ROI`, `interactive_roi`, `name_setup`, `use_setup`, `automatic_contrast`.
  - `nuclei_diameter`, `cell_diameter` (in the image plane for a 2D image), `scale_factor`, `zoom_factors`.
  - Segmentation method: `trig_cellpose`, `trig_stardist` (both `False` = watershed) and `trig_cellpose_cyto`.
  - `multilabel`, `aggregate_grow_factor`.
  - Nuclei size refinement: `size_reference`, `iterative_split`, `oversize_factor`, `max_split_iterations`, `merge_undersized`, `undersize_factor`.
  - Splitting (v1.6.3): `split_gamma` (gamma on the intensity used to split touching nuclei and cells; 1.0 = off), `split_gamma_step` (added at each re-split level of the nuclei; 0 = off), `min_piece_fraction` (split pieces smaller than this × a nucleus are merged back; 0 = off). All three off = v1.6.2 behaviour.
- **Cell 4 — Load image and stain table.**
  - Loads the image with `initialize_dataset()`, automatically choosing lazy (chunked) or eager reading from the file's size, and sets `is_2d` for a single-plane image. With `interactive_roi = True`, `select_roi_interactively()` first opens a napari window with a draggable ROI rectangle (X/Y only); with a saved stain table, its channels are named and colored after the table.
  - Prints the channel list (`print_channel_list()`). In batch mode nothing is loaded: the metadata of every image is read and any image whose channels differ from the first one is flagged.
  - Shows the **interactive stain table** (`edit_stain_dict()`, requires `ipywidgets`, installed with `jupyter`). One row per condition: NUCLEI and CYTOPLASM are fixed, other stains can be added or removed. Choose each condition's channel (`no` = not used) and color (blue, red, green, white, yellow, magenta), type the marker name, tick **Intracellular?** for markers that define the cytoplasm (`cyto_markers`), and **Membrane?** for membrane stains (`membrane_markers`, also included in `cyto_markers`): their rings are filled to get the cytoplasm, and the bright membrane is used as the border between touching cells. The two ticks exclude each other. The table is checked live (e.g. a channel used twice, a missing marker name). **Confirm** saves it to `<name_setup>_stain_dict.json`; with `use_setup = True` it is reloaded, already confirmed, on the next run.
- **Cell 5 — Read the stain table and run the batch.** Reads the confirmed table into `stain_dict` / `cyto_markers`. If it isn't confirmed yet, execution stops with a message (click **Confirm**, then run again from Cell 5). In batch mode, `run_batch_folder()` then processes every image in the folder with the whole pipeline (no viewers or plots; 3D exports skipped for 2D images) and stops; the image-processing parameters for the batch are set in this cell.

### 3. Preparation and Preview (Cells 6-8)
- `prepare_and_preview()` builds the image stack and the `stain_df` table (channels not used in the stain table are dropped) and opens a napari viewer for channel inspection.
- `prepare_stain_settings()` loads or creates per-channel contrast/gamma settings — reused from `<name_setup>_setup.csv` if it exists, otherwise set interactively in napari, or picked automatically from the histogram if `automatic_contrast = True`.
- Cell 8 shows the complete stain settings table for review.

### 4. Image Processing and Segmentation (Cells 9-22)
- **Preprocessing:** normalization (`run_normalize()`), isotropic resampling (`run_resample()`), median denoising (`run_denoise()`), contrast/gamma (`run_contrast_gamma()`), Gaussian smoothing (`run_smooth()`, tunable `sigma`) and histogram equalization (`run_equalize()`, tunable `num_plateaus` / `plateau_factor`).
- **Thresholding (Cell 15):**
  - Marker channels: `run_threshold()` combines a selectable global method (`threshold_method`: Otsu / median / Huang), local Sauvola thresholding and a statistical-background component (blend weights fixed internally). The histogram marks the global and final thresholds per channel. These masks decide marker positivity.
  - Cytoplasm: `segment_cytoplasm()` uses the **union of the stains** thresholded by `run_threshold()` - the CYTOPLASM channel and/or the intracellular markers, plus the membrane markers with their rings closed and filled (enclosed areas up to ~1.5 cell cross-sections, `build_cytoplasm_mask()`) - and assigns exactly these voxels to the nuclei, without any growth (a cell with no stained cytoplasm keeps just its nucleus). Cells are only grown from the nuclei when there is no CYTOPLASM channel and no ticked marker at all.
- **Histogram export:** `export_channel_histograms()` saves every stage's histograms and a Parameters sheet (including the splitting settings) to Excel.
- **Nuclei:** `segment_nuclei()` with watershed (default), StarDist (`trig_stardist`) or Cellpose (`trig_cellpose`). Splitting is automatic — there is no split profile to choose:
  - the watershed first pass uses a fixed built-in configuration: it also splits touching nuclei where the nuclear signal dips between them, and always splits a nucleus whose z-slices show separate islands (e.g. two lobes above and below its centre);
  - with `split_gamma` > 1 the nuclear intensity used by the watershed (and by the seam test of the fragment merge) is gamma-adjusted first (`apply_split_gamma()`), so the dips between touching nuclei are deeper; Cellpose/StarDist get the original image;
  - the pieces of each split smaller than `min_piece_fraction` × a nucleus are merged into the closest big piece of the same split (`absorb_small_split_pieces()`);
  - then, for every method, a size-based refinement sets a reference nucleus size (`size_reference`), merges fragments smaller than `undersize_factor` × reference into a touching neighbour (only if the result is still one plausible nucleus with no dark seam at the contact), and re-splits labels bigger than `oversize_factor` × reference with increasingly aggressive settings ('aggressive' → 'very aggressive' → 'extreme') and, with `split_gamma_step`, a higher gamma at each level. Labels that can't be fixed are kept, never deleted.
  - On a 2D image the same steps run in the image plane (see [2D images](#2d-images)).
- **Cytoplasm / PCM:** `segment_cytoplasm()` builds the cytoplasm from the CYTOPLASM channel (optionally shaped with Cellpose, `trig_cellpose_cyto = True`), otherwise from the intracellular markers, otherwise by growing the nuclei; touching cells are split where the marker signal fades and where a membrane marker is bright — on intensities gamma-adjusted with `split_gamma`, so the borders between cells are more distinct. `segment_pcm()` adds the pericellular matrix shell.
- **Label assignment and aggregates:** `assign_channel_labels()` links cells to marker channels; `detect_aggregates()` finds cell aggregates.
- **Visualization:** `view_processing_results()` opens napari viewers with every stage and the segmentation, including "Thresh islands" layers to check the threshold masks.

### 5. Quantification and Analysis (Cells 23-29)
Marker intensity is reported both for **positive cells** (cells that passed that marker's threshold) and for **all cells**. Sizes are volumes, or areas for a 2D image.
- `compute_percell_marker_intensity_df()` and `build_labels_df()`: per-cell marker intensity, overlap, size and position.
- `print_population_summary()`: counts and percentages per condition.
- `build_full_labels_df()`: the full table at original (non-resampled) resolution for export.
- `build_histogram_report()`: per-nucleus KDE plots and a PDF report.
- `plot_spatial_distributions()` (X/Y/Z, X/Y for a 2D image), `plot_size_distributions()` and `plot_marker_intensity_clouds()` (flow-cytometry-style intensity vs cytoplasm size plot, saved as PNG).

### 6. Export (Cells 30-34)
- **3D meshes (3D images only):** VTK volumes (`build_vtk_volumes()`) and per-marker STL meshes (`export_marker_stl()`) for ParaView or similar. Set `nuc_3D_export = True` in Cell 32 to also export a single cell as a VTK sub-volume.
- **Excel:** full quantification tables, stain settings and processing parameters (`export_quantification_to_excel()`), with areas instead of volumes for a 2D image.
- **FEA (3D images only):** Abaqus `.inp` mesh via tetrahedralization (`export_fea_mesh()`, using `tetgen`).

---

## Detailed Workflow: `Fluo_3D_LD_seg_v1.2.1.ipynb`

The Live/Dead notebook follows the nuclei notebook step by step, adapted for viability assays (e.g. Calcein-AM / ethidium homodimer, propidium iodide, a Hoechst counterstain) and ending with a **binary LIVE/DEAD call for every cell**. The pericellular matrix (PCM) and the cell aggregates of the nuclei notebook are not part of the LD analysis.

| Aspect | Nuclei notebook (v1.6.3) | LD notebook (v1.2.1) |
|---|---|---|
| Stain table | NUCLEI / CYTOPLASM fixed rows, intracellular / membrane ticks | LIVE / DEAD fixed rows (+ optional extra stains), **Nuclear / Cytoplasmic** per marker |
| Nuclei | NUCLEI channel | Union of the nuclear markers |
| Cells | Cytoplasm assigned to the nuclei | Cell bodies from the cytoplasmic markers, combined with the nuclei into one label per cell |
| Classification | Positivity per marker | Positivity per marker in its own compartment + LIVE/DEAD call |
| PCM / aggregates | Yes | No |
| Excel report | `<file>_segmentation.xlsx` | `<file>_live_dead.xlsx` |

### 1. Import Libraries and Helpers (Cells 1-2)
Same as the nuclei notebook (`load_nuclei_notebook_setup()`, `reload_helpers()`).

### 2. Inputs and Setup (Cells 3-5)
- **Cell 3 — Settings:** as in the nuclei notebook, plus:
  - `nuclei_diameter` (segmentation of the nuclear markers) and `cell_diameter` (segmentation of the cytoplasmic markers, whole cells);
  - `trig_cellpose_cyto`: Cellpose (`cyto3`) for the cell bodies;
  - LIVE/DEAD call: `min_positive_fraction` (a cell is positive for a marker when the marker covers at least this fraction of its compartment; 0 = any voxel) and `ld_rule` (`"dead_priority"`: DEAD-positive = DEAD, otherwise LIVE; `"coverage"`: for cells positive for both or neither marker, the one covering more of its compartment wins);
  - size refinement and `min_piece_fraction` for nuclei and cell bodies, and the split gamma (`split_gamma`, `split_gamma_step`).
- **Cell 4 — Load image and stain table:** loads the image (2D detection included) and shows the **LIVE/DEAD stain table** (`edit_ld_stain_dict()`): LIVE and DEAD rows plus optional extra stains (e.g. a nuclear counterstain of all cells, which helps the segmentation and is quantified but doesn't enter the call). For each marker choose the channel, color, marker name and **localization** — **Nuclear** (e.g. EthD, PI, Hoechst) or **Cytoplasmic** (e.g. calcein). At least one of LIVE and DEAD needs a channel. **Confirm** saves it to `<name_setup>_ld_stain_dict.json`.
- **Cell 5 — Read the stain table and run the batch:** reads `stain_dict` and `localization`; in batch mode `run_ld_batch_folder()` processes the folder and also writes `live_dead_batch_summary.xlsx` with the cell counts and viability of every image.

### 3. Preparation and Preview (Cells 6-8)
Same as the nuclei notebook.

### 4. Image Processing and Segmentation (Cells 9-20)
- **Preprocessing and thresholding (Cells 9-15):** as in the nuclei notebook; the nuclear markers are thresholded at the scale of a nucleus, the cytoplasmic ones at the scale of a cell.
- **Cell 16 — Histogram export**, with the LIVE/DEAD settings (`ld_settings`) in the Parameters sheet.
- **Cell 17 — Nuclei (`segment_ld_nuclei()`):** the masks of the nuclear markers are merged and segmented like the NUCLEI channel of the nuclei notebook (watershed / Cellpose / StarDist, split gamma, size refinement).
- **Cell 18 — Cells (`segment_ld_cells()`):** the masks of the cytoplasmic markers are merged and segmented into cell bodies at the scale of `cell_diameter`; nuclei and bodies are then combined into one label per cell:
  - a nucleus lying (for at least half of its size) in a body belongs to that cell; a body holding two or more nuclei is split between them along the cytoplasm border (cytoplasmic intensity with `split_gamma`);
  - a nucleus outside every body is a cell on its own (typical of a dead cell);
  - a body without a stained nucleus (typical of a live cell when the nuclear marker only stains dead cells) gets an **estimated nucleus**: a sphere (a disk in 2D) of `nuclei_diameter` at the centre of the cell.
- **Cell 19 — LIVE/DEAD call (`classify_live_dead()`):** each marker is measured in its own compartment — nuclear markers in the nucleus (stained or estimated), cytoplasmic markers in the whole cell — and every cell is called LIVE or DEAD with `ld_rule`. The result is the per-cell table `ld_cells_df` (status, position, size, coverage, positivity and intensity of every marker).
- **Cell 20 — View the results (`view_ld_results()`):** the nuclei-notebook viewers plus the stained and estimated nuclei and the LIVE (green) / DEAD (red) cells of the call.

### 5. Quantification and Analysis (Cells 21-28)
- `compute_percell_marker_intensity_df()` / `build_labels_df()`: marker positivity table (the LIVE + DEAD row counts double-positive cells).
- `print_ld_summary()`: LIVE and DEAD counts, percentages and viability, how each marker combination was called, marker positivity, and cell/nucleus size and marker intensity of the LIVE and DEAD cells.
- `build_full_labels_df()`, `build_histogram_report()` (per-cell PDF report).
- `plot_ld_spatial_distributions()` (LIVE/DEAD counts and viability along X, Y and Z), `plot_ld_size_distributions()`, `plot_live_dead_classification()` (LIVE vs DEAD coverage per cell with the positivity threshold, saved as `<file>_live_dead.png`), `plot_marker_intensity_clouds()`.

### 6. Export (Cells 29-33)
- **3D meshes (3D images only):** VTK volumes of nuclei and cells, per-marker STL meshes, optional single-cell VTK.
- **Excel:** `export_live_dead_to_excel()` writes `<file>_live_dead.xlsx` — Summary (call, viability, LIVE/DEAD/ALL populations), Cells (one row per cell) and Setup (stain table with localization, parameters).
- **FEA (3D images only):** Abaqus `.inp` mesh of the whole cells.

---

## Requirements
See `requirements.txt` for the full list. Key dependencies:
- `napari[all]`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`
- `jupyter` (includes `ipywidgets`, used by the interactive stain tables)
- `aicsimageio[nd2]`, `nd2reader`, `aicspylibczi` + `fsspec`, `readlif`, `oiffile` — image readers (ND2, TIFF/OME-TIFF, CZI, LIF, OIB); plain TIFFs are read with `tifffile`, installed with aicsimageio
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
