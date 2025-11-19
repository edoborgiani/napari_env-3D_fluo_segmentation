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

## Requirements
- Python 3.8+
- napari
- numpy
- scipy
- scikit-image
- matplotlib
- jupyter

## Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Napari team and contributors
- scikit-image, numpy, scipy, and matplotlib communities

---
For questions or support, please open an issue on GitHub.
