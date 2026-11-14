# Model Card — OceanTrace AI Oil Spill Segmentation Model

## Model Summary

| Field | Details |
|---|---|
| **Model name** | OceanTrace-SegFormer-B2 |
| **Architecture** | SegFormer-B2 (Mix Transformer encoder + lightweight MLP decoder) |
| **Task** | Semantic segmentation of SAR satellite imagery |
| **Input** | 512 × 512 × 3 (SAR replicated to 3 channels, or RGB) |
| **Output** | 512 × 512 segmentation map, 5 classes |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Status** | 🔴 Not yet trained — weights placeholder |

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
  └── Segmentation head → 512×512×5
```

---

## Classes

| ID | Name | Description |
|---|---|---|
| 0 | Clean Sea | Open ocean, background |
| 1 | Oil Spill | Confirmed oil contamination |
| 2 | Look-Alike | Dark patch that resembles oil (bio-film, low-wind area, etc.) |
| 3 | Ship | Vessel returns in SAR |
| 4 | Land | Land pixels (masked during evaluation) |

---

## Training Configuration (planned)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 6 × 10⁻⁵ |
| LR schedule | Polynomial decay |
| Batch size | 8 (4 GPU × 2) |
| Epochs | 80 |
| Loss | CrossEntropy + Dice (weighted) |
| Class weights | Inverse frequency (oil spill heavily upweighted) |
| Augmentation | Random flip, rotation, brightness/contrast, elastic distortion |
| Input size | 512 × 512 |
| Mixed precision | FP16 (torch.cuda.amp) |

---

## Datasets Used

| Dataset | Split | Purpose |
|---|---|---|
| M4D Oil Spill | Train / Val / Test | Primary segmentation training |
| GlobalOSD-SAR | Train | Classification auxiliary loss |
| Sentinel-1 | Inference | Real-world evaluation |

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
