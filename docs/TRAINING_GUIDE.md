# OceanTrace AI — Deep Learning Model Training Guide
**Model:** SegFormer-B2 (`nvidia/mit-b2` backbone)  
**Dataset:** M4D Marine Oil Spill Detection Dataset / GlobalOSD-SAR  
**Target Hardware:** Kaggle GPU (T4 / P100) or Google Colab (T4 / A100)

---

## 1. Prerequisites & Datasets

### A. M4D Oil Spill Detection Dataset
- **Official URL:** [https://m4d.iti.gr/oil-spill-detection-dataset/](https://m4d.iti.gr/oil-spill-detection-dataset/)
- **License:** Free for academic and research use
- **Format:** Paired SAR Satellite Images (.png) and 5-class Ground Truth Segmentation Masks (.png)
- **Classes:**
  - `0`: Clean Sea (Water surface)
  - `1`: Oil Spill (Crude / Fuel oil slick)
  - `2`: Look-Alike (Algae bloom, low wind calm water, internal waves)
  - `3`: Ship (Vessel metallic scatterer)
  - `4`: Land (Coastline & Islands)

### B. GlobalOSD-SAR (Zenodo)
- **DOI:** [10.5281/zenodo.15286918](https://doi.org/10.5281/zenodo.15286918)
- **License:** CC BY 4.0
- Large-scale SAR image patches for binary slick/look-alike classification generalization.

---

## 2. Option A: Training on Kaggle (Recommended — 30h Free GPU Weekly)

1. **Log in to Kaggle**: Go to [https://www.kaggle.com](https://www.kaggle.com).
2. **Upload Dataset**:
   - Click **Create** -> **New Dataset**.
   - Upload the extracted M4D dataset archive containing `images/` and `masks/` directories.
   - Name it `m4d-oil-spill`.
3. **Create Notebook**:
   - Click **Create** -> **New Notebook**.
   - Under **File**, select **Upload Notebook** and choose `notebooks/oil_spill_segformer_training.ipynb` from this repo.
4. **Configure GPU Accelerator**:
   - In the right-hand panel under **Session Options** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.
   - Turn **Internet ON**.
5. **Attach Dataset**:
   - Click **Add Data** on the right panel, search for your `m4d-oil-spill` dataset, and add it.
6. **Run Training**:
   - Run all cells (`Shift + Enter` or `Run All`).
   - Training takes approximately 3 to 4 hours on a T4 GPU for 40 epochs.
7. **Download Checkpoint**:
   - Once complete, look in the output panel (`/kaggle/working/artifacts/checkpoints/`).
   - Download `best_oilspill_segformer.pth`.

---

## 3. Option B: Training on Google Colab

1. **Upload Notebook**: Go to [https://colab.research.google.com](https://colab.research.google.com) and upload `oil_spill_segformer_training.ipynb`.
2. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Change Runtime**:
   - Click **Runtime** -> **Change runtime type** -> Select **T4 GPU**.
4. **Run Cells**:
   - Ensure the dataset path points to your Google Drive location.
   - Model weights will be saved to your Google Drive.

---

## 4. Deploying the Fine-Tuned Model into OceanTrace AI

Once you download `best_oilspill_segformer.pth`:

1. Copy the file into this project:
   ```
   c:\Users\donth_\Downloads\Smart-India-Hackthn-main\Smart-India-Hackthn-main\artifacts\checkpoints\best_oilspill_segformer.pth
   ```
2. Update the environment configuration in `oceantrace-backend/backend/.env`:
   ```ini
   MODEL_PATH=../../artifacts/checkpoints/best_oilspill_segformer.pth
   USE_GPU=false  # or true if your local machine has NVIDIA CUDA
   ```
3. Restart the FastAPI backend:
   ```bash
   cd oceantrace-backend/backend
   uvicorn app.main:app --reload --port 8000
   ```
4. The system will automatically switch from morphological pixel analysis to full SegFormer transformer inference!
