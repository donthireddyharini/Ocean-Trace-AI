# Model Card — OceanTrace AI Oil Spill Segmentation Model

## Model Summary

| Field | Details |
|---|---|
| **Model name** | OceanTrace-SegFormer-B2 |
| **Architecture** | SegFormer-B2 (Mix Transformer encoder + lightweight MLP decoder) |
| **Task** | Semantic segmentation of SAR satellite imagery (Binary: Sea vs Oil Spill) |
| **Input** | 512 × 512 × 3 (SAR replicated to 3 channels, or RGB) |
| **Output** | 512 × 512 segmentation map, 2 classes |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Primary Dataset** | Refined Deep-SAR Oil Spill (`bakhtiyar2222/deep-sar-oil-spill-segmentation-refined`) |
| **Status** | 🟡 Ready for Kaggle GPU execution via `notebooks/oil_spill_segformer_training.ipynb` |

---

## Architecture Details

### Encoder: Mix Transformer (MiT-B2)
- Hierarchical Transformer with overlapping patch embeddings.
- 4 stages producing multi-scale feature maps at 1/4, 1/8, 1/16, 1/32 of input resolution.
- ~25 M parameters.
- Pre-trained on ImageNet-1k (available via `nvidia/mit-b2` on HuggingFace Hub).

### Decoder: All-MLP Head
- Aggregates features from all 4 encoder stages.
- Uses simple linear projections + bilinear upsampling to 1/4 resolution, then to full resolution.
- Embedding dim: 768.
- No heavy decoder — inference is fast.

### Full model (~47 M parameters)
```
Input (512×512×3)
  └── MiT-B2 Encoder
        ├── Stage 1 → 128×128×64
        ├── Stage 2 →  64×64×128
        ├── Stage 3 →  32×32×320
        └── Stage 4 →  16×16×512
  └── All-MLP Decoder (embed_dim=768)
  └── Segmentation head → 512×512×2 (Binary: Background vs Oil Spill)
```

---

## Classes

| ID | Name | Description |
|---|---|---|
| 0 | Background / Sea | Ambient ocean, clean water, low wind look-alike boundary |
| 1 | Oil Spill | Verified pixel-level marine oil spill / emulsion |

---

## Training Configuration (Kaggle GPU T4 x2)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (`betas=(0.9, 0.999)`) |
| Learning rate | 6 × 10⁻⁵ |
| LR schedule | Cosine Annealing with Warmup |
| Batch size | 8 (with gradient accumulation = 2, effective batch size = 16) |
| Epochs | 25 |
| Loss | Weighted CrossEntropy (0.5) + Soft Dice Loss (0.5) |
| Class weights | `[1.0, 5.0]` (compensates for oil slick pixel sparsity) |
| Augmentation | Horizontal/Vertical flip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast |
| Input size | 512 × 512 |
| Mixed precision | FP16 (`torch.cuda.amp.autocast`) |
| Checkpoint dir | `/kaggle/working/oceantrace_artifacts/` |

---

## Datasets Used

| Dataset | Split | Samples | Purpose |
|---|---|---|---|
| Refined Deep-SAR (SOS) | Train / Val | 6,455 / 1,615 | Primary binary segmentation training |
| M4D Oil Spill | Supplementary | ~1,200 | Multi-class benchmark & evaluation |
| Sentinel-1 GRD | Operational | Live EO | Real-world satellite surveillance |

---

## Evaluation Metrics (Target)

| Metric | Target |
|---|---|
| Mean IoU | ≥ 0.70 |
| Oil Spill IoU | ≥ 0.65 |
| Look-Alike IoU | ≥ 0.55 |
| Pixel Accuracy | ≥ 0.90 |
| F1 (oil spill) | ≥ 0.75 |

> [!NOTE]
> Targets are estimates based on published SegFormer results on comparable SAR datasets. Actual performance depends on training data quality and augmentation strategy.

---

## Limitations

> [!WARNING]
> Read all limitations before deploying this model in production.

1. **Speckle noise** — SAR imagery has inherent speckle; performance degrades without speckle filtering pre-processing.
2. **Wind/wave look-alikes** — Low-wind areas and biological films can be misclassified as oil.
3. **Incidence angle sensitivity** — Model trained on a specific range of incidence angles; out-of-distribution angles reduce accuracy.
4. **Small spill detection** — Spills smaller than ~1 pixel patch may be missed.
5. **Sensor cross-calibration** — Model trained on Sentinel-1; performance on other SAR sensors (ALOS-2, COSMO-SkyMed) is not validated.
6. **Temporal drift** — Model does not incorporate temporal context; sequential frames are processed independently.
7. **Not for legal evidence** — Outputs are decision-support tools and should be verified by a trained analyst before regulatory action.

---

## Intended Use

- Maritime surveillance and oil spill early-warning systems.
- Support tool for coast-guard and environmental agencies.
- Research on automatic SAR image analysis.

## Out-of-Scope Use

- Real-time flight-safety or navigation decisions.
- Sole basis for legal prosecution without human verification.
- Non-SAR optical imagery without domain adaptation.

---

## Checkpoint Location

```
artifacts/checkpoints/segformer_b2_oilspill_best.pth   ← best val mIoU
artifacts/checkpoints/segformer_b2_oilspill_last.pth   ← last epoch
artifacts/metrics/training_history.json
```

---

## Citation

If you use this model, please cite:
```
OceanTrace AI — Smart India Hackathon SIH26143, 2024.
Built on SegFormer: Xie et al., NeurIPS 2021 (arXiv:2105.15203).
```
