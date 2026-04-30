# 3D Fluorescence Segmentation with Napari

This repository provides Jupyter notebooks and shared Python helpers for 3D segmentation and quantification of nuclei and fluorescence structures in microscopy images, using the Napari ecosystem. It targets researchers in bioimage analysis who need robust, reproducible workflows for volumetric fluorescence data.

> **Disclaimer:** This repository is freely available for use. The associated pipeline is currently being prepared for publication — please contact [Edoardo Borgiani](https://github.com/edoborgiani) for more information on how to use the pipeline or for collaboration enquiries.

## Features
- **Two active workflows**: nuclei segmentation (`Fluo_3D_nuc_seg_v1.4.2`) and Live/Dead segmentation (`Fluo_3D_LD_seg_v1.0`).
- **Shared helper library** (`helpers/`): processing, quantification, visualization, and report-export functions shared across notebooks.
- **Profile-aware imports** (`helpers/notebook_setup_helpers.py`): the `nuclei` and `ld` profiles load only the dependencies each workflow needs, avoiding unnecessary overhead.
- **3D Image Processing**: normalization, resampling to isotropic voxel size, denoising, thresholding, watershed and StarDist-based segmentation.
- **Napari Integration**: interactive visualization and manual correction at each processing step.
- **Quantification & Export**: per-cell marker statistics, spatial distributions, Excel reports, per-nucleus PDF rows, and 3D mesh export (VTK/STL/INP).

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.4.2.ipynb   # Nuclei segmentation — latest recommended version
├── Fluo_3D_LD_seg_v1.0.ipynb      # Live/Dead segmentation — v1.0
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── helpers/
    ├── __init__.py
    ├── notebook_helpers.py        # Core processing, segmentation, and export functions
    └── notebook_setup_helpers.py  # Package installation and profile-aware import loader
```

> **Note:** The `old_v/` folder (containing earlier notebook versions v1.0–v1.4.1) and Python `__pycache__` directories are excluded from version control and exist only locally.

## Getting Started

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
   Open `Fluo_3D_nuc_seg_v1.4.2.ipynb` for nuclei segmentation, or `Fluo_3D_LD_seg_v1.0.ipynb` for Live/Dead segmentation.

## Usage

- Follow the notebook cells in order: setup → image loading → preprocessing → segmentation → quantification → export.
- The first cells call `notebook_setup_helpers.install_required_packages()` and `load_common_imports(profile=...)` to bootstrap the environment automatically.
- Use Napari for interactive visualization and manual corrections at any step.
- All shared processing logic lives in `helpers/notebook_helpers.py` — customize functions there rather than duplicating code across notebooks.

## Detailed Workflow: `Fluo_3D_nuc_seg_v1.4.2.ipynb`

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
- **Nuclei**: 3D watershed (`segment_nuclei_watershed()`) or StarDist2D slice-by-slice with 3D merging (`stardist3d_from_2d()`).
- **Cytoplasm / PCM**: grown from nuclei via `grow_labels()` or from additional channels.
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
- **PDF**: per-nucleus image rows via `create_row_pdf()`.
- **3D meshes**: VTK / STL files for nuclei, cytoplasm, PCM, and markers for visualization in ParaView or similar.
- **FEA**: optional `.inp` file generated via tetrahedralization (`tetgen`).

### Tips
- For large datasets, use `ROI` cropping and a reduced `scale_factor` during development.
- Use Napari's layer controls to inspect intermediate results before committing to full-resolution runs.
- Review exported Excel files for quantitative QC before downstream analysis.

---

## Requirements
See `requirements.txt` for the full list. Key dependencies:
- `napari[all]`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`
- `aicsimageio[nd2]`, `nd2reader`
- `pyvista`, `SimpleITK`, `csbdeep`, `stardist`
- `meshio`, `tetgen`, `meshlib`
- `xlsxwriter`, `reportlab`, `Pillow`, `vispy`

## Contributing
Contributions are welcome. Please open issues or pull requests for bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Napari team and contributors
- scikit-image, numpy, scipy, and matplotlib communities

---
For questions or support, please open an issue on GitHub.