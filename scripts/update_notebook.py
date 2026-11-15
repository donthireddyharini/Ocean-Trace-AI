import json
import os
import shutil

nb_path = "notebooks/oil_spill_segformer_training.ipynb"
if not os.path.exists(nb_path):
    nb_path = "Smart-India-Hackthn-main/notebooks/oil_spill_segformer_training.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3 enhancements: crystal-clear error message if dataset not attached
cell3_src = "".join(nb["cells"][3]["source"])
if "dataset not attached" not in cell3_src.lower():
    check_code = """
if not DATASET_ROOT.exists() or len(list(DATASET_ROOT.glob("*"))) == 0:
    raise FileNotFoundError(
        "\\n" + "="*70 + "\\n"
        "ERROR: Refined Deep-SAR Oil Spill Dataset is NOT attached to /kaggle/input!\\n"
        "To attach the dataset:\\n"
        "  1. Look at the right sidebar in the Kaggle Editor.\\n"
        "  2. Click '+ Add Data'.\\n"
        "  3. Search for: bakhtiyar2222/deep-sar-oil-spill-segmentation-refined\\n"
        "  4. Click the 'Add' (+) button next to it.\\n"
        "  5. Rerun the notebook.\\n" + "="*70
    )
"""
    cell3_src += "\n" + check_code
    nb["cells"][3]["source"] = [line + "\n" for line in cell3_src.split("\n")]

# Cell 10 enhancements: save final_oilspill_segformer.pth
cell10_src = "".join(nb["cells"][10]["source"])
if "final_oilspill_segformer.pth" not in cell10_src:
    cell10_src = cell10_src.replace(
        'last_checkpoint_path = os.path.join(cfg.output_dir, "last_oilspill_segformer.pth")',
        'last_checkpoint_path = os.path.join(cfg.output_dir, "last_oilspill_segformer.pth")\nfinal_checkpoint_path = os.path.join(cfg.output_dir, "final_oilspill_segformer.pth")'
    )
    cell10_src += "\n# Save final checkpoint\nimport shutil\nshutil.copyfile(best_checkpoint_path if os.path.exists(best_checkpoint_path) else last_checkpoint_path, final_checkpoint_path)\nprint(f'Final model checkpoint saved to: {final_checkpoint_path}')\n"
    nb["cells"][10]["source"] = [line + "\n" for line in cell10_src.split("\n")]

# Cell 11 enhancements: save training_history.csv, class_mapping.json, training_report.md
cell11_src = "".join(nb["cells"][11]["source"])
if "training_history.csv" not in cell11_src:
    extra_cell11 = """
# Save training history CSV
import pandas as pd
history_df = pd.DataFrame(history)
history_csv_path = os.path.join(cfg.output_dir, "training_history.csv")
history_df.to_csv(history_csv_path, index=False)

# Save class mapping
class_mapping_path = os.path.join(cfg.output_dir, "class_mapping.json")
with open(class_mapping_path, "w", encoding="utf-8") as f:
    json.dump({"0": "Background / Sea", "1": "Oil Spill"}, f, indent=2)

# Generate detailed training_report.md
report_md_path = os.path.join(cfg.output_dir, "training_report.md")
with open(report_md_path, "w", encoding="utf-8") as f:
    f.write(f"# OceanTrace AI - SegFormer-B2 Training Report\\n\\n"
            f"## Model Configuration\\n"
            f"- Backbone: {cfg.pretrained_model_name}\\n"
            f"- Task: Binary Semantic Segmentation\\n"
            f"- Input Resolution: {cfg.image_size}x{cfg.image_size}\\n"
            f"- Epochs: {cfg.epochs}\\n\\n"
            f"## Validation Performance\\n"
            f"- Best Oil Spill IoU: {best_oil_iou:.4f}\\n"
            f"- Mean IoU: {metrics['mean_iou']:.4f}\\n"
            f"- Oil Spill Dice: {metrics['class_dices'][1]:.4f}\\n"
            f"- Oil Spill Precision: {metrics['class_precisions'][1]:.4f}\\n"
            f"- Oil Spill Recall: {metrics['class_recalls'][1]:.4f}\\n"
            f"- Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\\n")
print(f"  - {history_csv_path}")
print(f"  - {class_mapping_path}")
print(f"  - {report_md_path}")
"""
    cell11_src += "\n" + extra_cell11
    nb["cells"][11]["source"] = [line + "\n" for line in cell11_src.split("\n")]

# Cell 12 enhancements: make sure prediction folder has validation_visualizations alias
cell12_src = "".join(nb["cells"][12]["source"])
if "validation_visualizations" not in cell12_src:
    extra_cell12 = """
# Mirror prediction samples to validation_visualizations for standard artifact specification
val_vis_dir = os.path.join(cfg.output_dir, "validation_visualizations")
shutil.copytree(samples_dir, val_vis_dir, dirs_exist_ok=True)
print(f"Mirrored sample predictions to: {val_vis_dir}")
"""
    cell12_src += "\n" + extra_cell12
    nb["cells"][12]["source"] = [line + "\n" for line in cell12_src.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with all exact artifact outputs successfully!")
