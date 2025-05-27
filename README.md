# Burned Area Classifier GUI

A user-friendly tool for detecting burned areas in satellite images using Support Vector Machines (SVMs). Designed for researchers, analysts, and GIS users, this application streamlines the process of image preprocessing, model training, classification, and validation with no coding required.

## 🔍 Overview

This tool allows users to:
- Train new SVM models on pre- and post-fire satellite images
- Apply existing models to classify post-fire imagery
- Test model performance using dNBR-based ground truth

Supports both **Sentinel-2** and **Landsat 8/9** imagery.

Built with:
- Python + Tkinter for GUI
- Rasterio, NumPy, Scikit-learn for backend
- PyInstaller for executable packaging

> 📘 Detailed instructions and examples are included in the [project report (PDF)](./Report.pdf).

---

## 🖥️ Key Features

- Supports both Sentinel-2 and Landsat 8/9 imagery
- Preprocess images with optional cloud and water masking
- Generate training data automatically using dNBR thresholds
- Train and apply SVM models for burned area classification
- View RGB previews of images within the app
- Evaluate results using precision, recall, and confusion matrix
- Runs as a standalone `.exe` – no Python installation needed

---

## 📦 Download

👉 [Download the latest .exe from the Releases page](https://github.com/your-username/burnarea-classifier-gui/releases/latest)

---


## 🚀 How to Run (From Source)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the GUI:
   ```bash
   python app.py
   ```

---

## 🧪 Example Workflows

- **Train Model**: Load pre- and post-fire images, mask clouds/water, define thresholds, and train an SVM.
- **Classify Image**: Load a trained `.pkl` model and classify a post-fire image.
- **Test Model**: Compare predictions against dNBR ground truth using validation metrics.

Case studies include:
- Alexandroupoli 2023 (training)
- Rhodes 2023 (classification and testing)

---

## 🛠️ Technologies Used

- **Python**
- **Tkinter** – GUI
- **Rasterio** – Geospatial raster handling
- **Scikit-learn** – SVM classifier
- **GeoPre** – Custom image processing library
- **PyInstaller** – Packaging into an executable

---

## 📚 Report

The [project report (PDF)](./Report.pdf) contains:

- A complete description of the tool's development
- Case studies using real satellite data
- **Step-by-step instructions on how to use the application**

---

## 💡 Future Improvements

- Add clipping
- Improved UI and logging

---

## 📄 License

MIT License

---

> Developed as part of the Geoinformatics Project at Politecnico di Milano  
> By Matteo Gobbi Frattini