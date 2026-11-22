import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from lightning_modules.detector import DeepfakeDetector
from network import build_efficientnet_b0

ckpt_path = "models/best_model_finetuned.ckpt"
pt_output = "models/fine_tuned.pt"

backbone = build_efficientnet_b0(dropout=0.4)
model = DeepfakeDetector.load_from_checkpoint(
    ckpt_path,
    model=backbone,
    lr_head=3e-4,
    lr_backbone=3e-5,
)
os.makedirs(os.path.dirname(pt_output), exist_ok=True)
torch.save(model.model.state_dict(), pt_output)
print(f"✅ Exported PyTorch model to {pt_output}")
