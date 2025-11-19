# 3D Fluorescence Nuclei Segmentation with Napari

This repository provides a set of Jupyter notebooks and tools for 3D nuclei segmentation in fluorescence microscopy images using the Napari ecosystem. The project is designed for researchers and practitioners in bioimage analysis who need robust, reproducible workflows for segmenting and analyzing 3D nuclear structures.

## Features
- **Jupyter Notebooks**: Step-by-step segmentation pipelines with code and explanations.
- **3D Image Processing**: Tools for handling and segmenting volumetric fluorescence data.
- **Napari Integration**: Interactive visualization and manual correction using Napari.
- **Versioned Workflows**: Multiple notebook versions to track improvements and changes.

## Repository Structure
```
.
├── Fluo_3D_nuc_seg_v1.2.1.ipynb   # Notebook: v1.2.1 workflow
├── Fluo_3D_nuc_seg_v1.2.2.ipynb   # Notebook: v1.2.2 workflow (latest)
├── Fluo_3D_nuc_seg_v1.2.ipynb     # Notebook: v1.2 workflow
├── Fluo_3D_nuc_seg_v1.3.ipynb     # Notebook: v1.3 workflow (experimental)
└── old_v/
    ├── Fluo_3D_nuc_seg_v1.1.1.ipynb
    ├── Fluo_3D_nuc_seg_v1.1.ipynb
    ├── Fluo_3D_nuc_seg_v1.2.ipynb
    └── Fluo_3D_nuclei_segmentation.ipynb
```

## Getting Started
1. **Clone the repository**
   ```powershell
   git clone https://github.com/edoborgiani/napari_env-3D_fluo_segmentation.git
   cd napari_env-3D_fluo_segmentation
   ```
2. **Set up a Python environment**
   - Python 3.8 or newer is recommended.
   - Install dependencies (see below).

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   # or install manually:
   pip install napari[all] numpy scipy scikit-image matplotlib jupyter
   ```

4. **Launch Jupyter Notebook**
   ```powershell
   jupyter notebook
   ```
   Open one of the `Fluo_3D_nuc_seg_*.ipynb` notebooks to get started.

## Usage
- Follow the instructions in the notebooks to load your 3D fluorescence images, run segmentation, and visualize results.
- Use Napari for interactive visualization and manual corrections.
- Each notebook version may contain different approaches or improvements—see notebook headers for details.


## Detailed Guide: Using `Fluo_3D_nuc_seg_v1.3.ipynb`

The `Fluo_3D_nuc_seg_v1.3.ipynb` notebook provides an advanced, modular workflow for 3D segmentation and quantification of nuclei and cellular structures in fluorescence microscopy images. Below is a step-by-step guide to using this notebook:

### 1. Install Required Packages
The notebook will attempt to install some dependencies automatically, but you may need to ensure the following are installed:
```
pip install napari[all] aicsimageio[nd2] nd2reader xlsxwriter pyvista simpleitk scikit-image csbdeep stardist meshio tetgen meshlib matplotlib pandas
```

### 2. Load Your Image Data
- Set the `input_file` variable to your `.nd2` or `.tif` file.
- The notebook reads metadata and extracts the 3D image stack and channel information.
- Physical pixel sizes are automatically extracted for correct scaling.

### 3. Define Sample and Staining Information
- Set parameters such as `nuclei_diameter`, `cell_diameter`, and the `stain_dict` dictionary to match your experiment.
- The notebook will create a DataFrame mapping channels to biological markers and colors.

### 4. Select Region of Interest (ROI) and Scaling
- Adjust the `ROI` and `scale_factor` variables to crop and/or downsample your data for faster processing.

### 5. Setup and Visualization
- The notebook supports loading or saving a CSV setup file for channel contrast/gamma settings.
- Napari is used for interactive visualization and adjustment of each channel.

### 6. Image Preprocessing
- **Normalization**: All channels are normalized to [0, 255].
- **Resampling**: Images are resampled to isotropic voxel size if needed.
- **Denoising**: Median and Gaussian filters are applied.
- **Contrast/Gamma Correction**: Per-channel adjustments based on setup.
- **Histogram Export**: Histograms for each channel are saved to Excel.

### 7. Thresholding
- Multiple thresholding methods are combined (Otsu, Sauvola, statistical background, intensity gain) for robust segmentation.
- Small islands are removed from binary masks.

### 8. Segmentation
- **Nuclei**: Segmented using either 3D watershed or StarDist2D (slice-by-slice, then merged in 3D).
- **Cytoplasm/PCM**: Segmented by growing from nuclei or using additional channels.
- **Assignment**: Labels are assigned to other channels for marker quantification.
- **Aggregates**: Large cell aggregates are identified.

### 9. Visualization
- Napari is used to visualize all processing steps, including overlays of original, denoised, thresholded, and segmented images.

### 10. Quantification
- The notebook computes:
   - Number and size of nuclei and cells
   - Marker intensities and volumes per cell
   - Spatial distributions (X, Y, Z)
- Results are exported to Excel for further analysis.

### 11. 3D Export
- **.VTK and .STL**: 3D meshes for nuclei, cytoplasm, PCM, and markers are generated for visualization in external tools (e.g., ParaView).
- **Finite Element Analysis**: Optionally, a `.inp` file for FEA is created using tetrahedralization.

### 12. Customization
- The notebook is modular—functions for each processing step can be adapted to your needs.
- Parameters for segmentation, filtering, and quantification are easily adjustable at the top of the notebook.

### Tips
- For large datasets, consider cropping or downsampling to speed up processing.
- Use Napari’s interactive tools to inspect and manually correct segmentations if needed.
- Review the exported Excel files for quantitative results and quality control.

---

## Requirements
- Python 3.8+
- napari
- numpy
- scipy
- scikit-image
- matplotlib
- jupyter
- aicsimageio[nd2]
- nd2reader
- xlsxwriter
- pyvista
- simpleitk
- csbdeep
- stardist
- meshio
- tetgen
- meshlib
- pandas

## Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Napari team and contributors
- scikit-image, numpy, scipy, and matplotlib communities

---
For questions or support, please open an issue on GitHub.
