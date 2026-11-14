# OceanTrace AI — Deep Learning Model Training Guide
**Model:** SegFormer-B2 (`nvidia/mit-b2` backbone)  
**Primary Dataset:** Refined Deep-SAR Oil Spill (SOS) (`bakhtiyar2222/deep-sar-oil-spill-segmentation-refined`)  
**Target Hardware:** Kaggle GPU (T4 x2 / P100) or Google Colab (T4 / A100)

---

## 1. Primary Dataset & Target Classes

### Refined Deep-SAR Oil Spill (SOS) Dataset
- **Kaggle URL:** [https://www.kaggle.com/datasets/bakhtiyar2222/deep-sar-oil-spill-segmentation-refined](https://www.kaggle.com/datasets/bakhtiyar2222/deep-sar-oil-spill-segmentation-refined)
- **Zenodo DOI:** [10.5281/zenodo.15298010](https://zenodo.org/records/15298010)
- **Inventory:** **6,455** training samples, **1,615** validation samples (**8,070** total Sentinel-1 SAR pairs)
- **Target Classes (Binary):**
  - `0`: Background / Sea (open ocean water, non-spill maritime background)
  - `1`: Oil Spill (confirmed crude/fuel oil slick & emulsion)

---

## 2. Option A: Training on Kaggle (Recommended — 30h Free GPU Weekly)

1. **Log in to Kaggle**: Go to [https://www.kaggle.com](https://www.kaggle.com).
2. **Create / Upload Notebook**:
   - Click **Create** -> **New Notebook**.
   - Under **File**, select **Upload Notebook** and choose `notebooks/oil_spill_segformer_training.ipynb` from this repository.
3. **Attach Primary Dataset**:
   - In the right-hand panel under **Input**, click **Add Input**.
   - Search for `bakhtiyar2222/deep-sar-oil-spill-segmentation-refined` and attach it.
4. **Configure GPU Accelerator**:
   - Under **Session options** -> **Accelerator**, select **GPU T4 x2** (or **GPU P100**).
   - Ensure **Internet ON** is enabled.
5. **Run All Cells**:
   - Click **Run All**.
   - The notebook automatically:
     - Discovers the dataset directory and runs integrity validation (`dataset_validation_report.json`).
     - Trains `nvidia/mit-b2` with mixed precision (AMP) and Cosine Annealing scheduler.
     - Saves periodic and best checkpoints (`best_oilspill_segformer.pth`).
     - Generates metrics, training curves, confusion matrix, and sample prediction overlays.
     - Creates the final downloadable archive: `/kaggle/working/oceantrace_artifacts.zip`.
6. **Download Artifacts**:
   - Download `best_oilspill_segformer.pth` or `oceantrace_artifacts.zip` from `/kaggle/working/`.

---

## 3. Option B: Training on Google Colab

1. **Upload Notebook**: Go to [https://colab.research.google.com](https://colab.research.google.com) and upload `oil_spill_segformer_training.ipynb`.
2. **Mount Google Drive** or download dataset from Kaggle via API.
3. **Change Runtime**:
   - Click **Runtime** -> **Change runtime type** -> Select **T4 GPU**.
4. **Run Cells**:
   - Model weights and artifacts will be saved to `./oceantrace_artifacts`.

---

## 4. Deploying the Fine-Tuned Model into OceanTrace AI

Once you download `best_oilspill_segformer.pth`:

1. Copy the file into your local repository or server:
   - `oceantrace_artifacts/best_oilspill_segformer.pth` (or project root `best_oilspill_segformer.pth`)
2. Start the FastAPI backend:
   ```powershell
   cd oceantrace-backend\backend
   python -m uvicorn app.main:app --port 8000
   ```
3. Verify model status:
   - Query `GET http://localhost:8000/api/model/status`
   - Response will show `checkpoint_loaded: true` and `model_status: "operational"`.
4. Full forward-pass PyTorch inference will now execute for all uploaded satellite images on both the API and live frontend!
