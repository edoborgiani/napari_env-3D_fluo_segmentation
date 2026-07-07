# 3D Fluorescence Segmentation with Napari

This repository provides Jupyter notebooks and shared Python helpers for 3D segmentation and quantification of nuclei and fluorescence structures in microscopy images, using the Napari ecosystem. It targets researchers in bioimage analysis who need robust, reproducible workflows for volumetric fluorescence data.

> **Disclaimer:** This repository is freely available for use. The associated pipeline is currently being prepared for publication — please contact [Edoardo Borgiani](https://github.com/edoborgiani) for more information on how to use the pipeline or for collaboration enquiries.

## Features
- **Two active workflows**: nuclei segmentation (`Fluo_3D_nuc_seg_v1.5`) and Live/Dead segmentation (`Fluo_3D_LD_seg_v1.1`).
- **Shared helper library** (`helpers/`): processing, quantification, visualization, and report-export functions shared across notebooks.
- **Profile-aware imports** (`helpers/notebook_setup_helpers.py`): the `nuclei` and `ld` profiles load only the dependencies each workflow needs, avoiding unnecessary overhead.
- **3D Image Processing**: normalization, resampling to isotropic voxel size, denoising, thresholding, watershed and StarDist-based segmentation.
- **LD union labeling**: when no dedicated NUCLEI channel is present, `segment_nuclei()` automatically falls back to merging all threshold channels via bitwise OR and running connected-component labeling to identify individual cells.
- **Napari Integration**: interactive visualization and manual correction at each processing step.
- **Quantification & Export**: per-cell marker statistics, spatial distributions, Excel reports, per-nucleus PDF rows (with channel name labels and correct aspect-ratio images), and 3D mesh export (VTK/STL/INP).

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.5.ipynb     # Nuclei segmentation — latest recommended version
├── Fluo_3D_nuc_seg_v1.4.2.ipynb   # Nuclei segmentation — previous stable version
├── Fluo_3D_LD_seg_v1.1.ipynb      # Live/Dead segmentation — latest version
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── helpers/
    ├── __init__.py
    ├── notebook_helpers.py        # Core processing, segmentation, and export functions
    └── notebook_setup_helpers.py  # Package installation and profile-aware import loader
```

> **Note:** The `old_v/` folder (containing earlier notebook versions v1.0–v1.4.1) and Python `__pycache__` directories are excluded from version control and exist only locally.

## Getting Started

> **Tip — faster installs:** `requirements.txt` pulls in several large, dependency-heavy packages (napari, TensorFlow, PyTorch-based Cellpose, VTK/PyVista, SimpleITK). Plain `pip` can take a long time to resolve and download all of them. Installing [`uv`](https://github.com/astral-sh/uv) first and using it in place of `pip install` (same `requirements.txt`, no other changes needed) resolves and installs the same packages dramatically faster:
> ```
> pip install uv
> uv pip install -r requirements.txt
> ```
> The plain `pip install -r requirements.txt` commands below still work exactly the same if you'd rather not add `uv`.

### Windows

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

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```powershell
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.5.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.1.ipynb` for Live/Dead segmentation.

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
   Open `Fluo_3D_nuc_seg_v1.5.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.1.ipynb` for Live/Dead segmentation.

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

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `Fluo_3D_nuc_seg_v1.5.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.1.ipynb` for Live/Dead segmentation.

> **Note (Headless servers only):** If you are running on a remote Linux server without a physical display (e.g. an HPC cluster accessed via SSH), Napari's Qt backend will fail to open. Run the following commands **before step 4** to start a virtual framebuffer:
> ```bash
> sudo apt-get install libxcb-xinerama0 xvfb
> export DISPLAY=:99
> Xvfb :99 -screen 0 1024x768x24 &
> ```
> This is not needed on a standard desktop Linux installation.

## Usage

- Follow the notebook cells in order: setup → image loading → preprocessing → segmentation → quantification → export.
- The first cells call `notebook_setup_helpers.install_required_packages()` and `load_common_imports(profile=...)` to bootstrap the environment automatically.
- Use Napari for interactive visualization and manual corrections at any step.
- All shared processing logic lives in `helpers/notebook_helpers.py` — customize functions there rather than duplicating code across notebooks.

## Detailed Workflow: `Fluo_3D_nuc_seg_v1.5.ipynb`

### 1. Environment Setup
The notebook installs optional packages automatically via `notebook_setup_helpers` and loads all imports using `load_common_imports(profile='nuclei')`.

### 2. Load Image Data
- Set `input_file` to your `.nd2` or `.tif` file path.
- Physical pixel sizes are extracted from metadata for correct spatial scaling.

### 3. Define Sample & Staining Information
- Configure `stain_dict` to map channels to biological markers and display colors.
- `prepare_stain_settings()` and `build_labels_dict()` from `notebook_helpers` build the working data structures.

### 4. ROI & Scaling
- Adjust `ROI` and `scale_factor` to crop or downsample for faster iteration.

### 5. Setup & Per-Channel Contrast/Gamma
- Load or save a CSV setup file for per-channel contrast and gamma settings.
- Napari is used for interactive inspection and adjustment.

### 6. Image Preprocessing
- **Normalization**: channels normalized to [0, 255] via `normalize_image_channels()`.
- **Resampling**: isotropic voxel resampling via `resample_to_isotropic()`.
- **Denoising**: median and Gaussian filters via `apply_median_denoise()` / `apply_gaussian_smoothing()`.
- **Histogram export**: per-channel histograms saved to Excel via `export_channel_histograms()`.

### 7. Thresholding
- Combined thresholding (Otsu, Sauvola, statistical background, intensity gain) for robust binary masks.
- Small artefact islands removed via `remove_small_islands()`.

### 8. Segmentation
- **Nuclei**: 3D watershed (`segment_nuclei_watershed()`), StarDist2D slice-by-slice with 3D merging (`stardist3d_from_2d()`), or Cellpose 3D (`segment_nuclei_cellpose()`).
- **Cytoplasm / PCM**: grown from nuclei via `grow_labels()`, from additional channels, or shaped directly with Cellpose 3D (`segment_cytoplasm_cellpose()`) and relabelled to the nuclei IDs — independent of which method found the nuclei, so e.g. StarDist nuclei + Cellpose cell shape is a valid combination.
- **Label assignment**: `assign_labels()` maps segmented structures to marker channels.
- **Aggregate detection**: large multi-cell aggregates flagged separately.

### 9. Visualization
- Napari overlays for raw, denoised, thresholded, and labelled images at each stage.

### 10. Quantification
- `compute_nuclei_cytoplasm_stats()` and `compute_marker_stats_for_marker()` compute per-cell volumes, intensities, and spatial distributions (X, Y, Z).
- `collect_histogram_data()` collects per-channel statistics.
- `print_population_summary()` prints a summary to the notebook.

### 11. Export
- **Excel**: full quantification tables via `export_quantification_to_excel()`.
- **PDF**: per-nucleus image rows via `create_row_pdf()`, with channel name labels at the top of each image column and images rendered at their correct aspect ratio with inter-image spacing.
- **3D meshes**: VTK / STL files for nuclei, cytoplasm, PCM, and markers for visualization in ParaView or similar.
- **FEA**: optional `.inp` file generated via tetrahedralization (`tetgen`).

---

## Detailed Workflow: `Fluo_3D_LD_seg_v1.1.ipynb`

The Live/Dead notebook follows the same helper-based structure as the nuclei notebook, adapted for two-channel viability assays (e.g. Calcein-AM / EthD).

| Aspect | Nuclei notebook | LD notebook |
|---|---|---|
| Profile | `"nuclei"` | `"ld"` (lighter imports) |
| Segmentation | Watershed / StarDist on NUCLEI channel | Union of all threshold channels → connected components |
| Cytoplasm / PCM | Dedicated channels + grow steps | Not applicable |
| NUCLEI row in `stain_complete_df` | Populated by `segment_nuclei()` | Added as empty placeholder after segmentation |
| Export | Same Excel / PDF / VTK / STL / FEA pipeline | Same pipeline |

### 1. Environment Setup
The notebook installs optional packages automatically via `notebook_setup_helpers` and loads all imports using `load_common_imports(profile='ld')`, which applies a lighter import profile compared to the nuclei workflow.

### 2. Load Image Data
- Set `input_file` to your `.nd2` or `.tif` file path.
- Physical pixel sizes are extracted from metadata for correct spatial scaling.

### 3. Define Sample & Staining Information
- Configure `stain_dict` with `LIVE` / `DEAD` channel entries, `nuclei_diameter`, `cell_diameter`, `multilabel`, and `nuclei_split_config`.
- `prepare_stain_settings()` and `build_labels_dict()` from `notebook_helpers` build the working data structures.

### 4. ROI & Scaling
- Adjust `ROI` and `scale_factor` to crop or downsample for faster iteration.

### 5. Setup & Per-Channel Contrast/Gamma
- Load or save a CSV setup file for per-channel contrast and gamma settings.
- Napari is used for interactive inspection and adjustment.

### 6. Image Preprocessing
- **Normalization**: channels normalized to [0, 255] via `normalize_image_channels()`.
- **Resampling**: isotropic voxel resampling via `resample_to_isotropic()`.
- **Denoising**: median and Gaussian filters via `apply_median_denoise()` / `apply_gaussian_smoothing()`.
- **Histogram export**: per-channel histograms saved to Excel via `export_channel_histograms()`.

### 7. Thresholding
- Combined thresholding (Otsu, Sauvola, statistical background, intensity gain) for robust binary masks.
- Small artefact islands removed via `remove_small_islands()`.

### 8. Segmentation
- **Cells**: `segment_nuclei()` uses the LD fallback — all threshold channels are merged via bitwise OR and connected-component labeling identifies individual cells.
- A NUCLEI placeholder row is added to `stain_complete_df` after segmentation.
- `assign_channel_labels()` maps LIVE / DEAD intensity into the segmented objects.

### 9. Visualization
- Napari overlays for raw, denoised, thresholded, and labelled images at each stage.

### 10. Quantification
- `compute_nuclei_cytoplasm_stats()` and `compute_marker_stats_for_marker()` compute per-cell volumes, intensities, and spatial distributions (X, Y, Z).
- `collect_histogram_data()` collects per-channel statistics.
- `print_population_summary()` prints a summary to the notebook.

### 11. Export
- **Excel**: full quantification tables via `export_quantification_to_excel()`.
- **PDF**: per-nucleus image rows via `create_row_pdf()`, with channel name labels at the top of each image column and images rendered at their correct aspect ratio with inter-image spacing.
- **3D meshes**: VTK / STL files for segmented cells and markers for visualization in ParaView or similar.
- **FEA**: optional `.inp` file generated via tetrahedralization (`tetgen`).

---

## Requirements
See `requirements.txt` for the full list. Key dependencies:
- `napari[all]`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`
- `aicsimageio[nd2]`, `nd2reader`
- `pyvista`, `SimpleITK`, `csbdeep`, `stardist`
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
