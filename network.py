import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_FEATURE_BLOCKS = 9  # EfficientNet-B0 features indices 0..8
CLASS_NAMES = {0: "FAKE", 1: "REAL"}  # locked mapping used everywhere


def build_efficientnet_b0(dropout=0.4, num_classes=2):
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def _strip_prefix(key):
    for prefix in ("model.model.", "model.", "module.", "net."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state dict or checkpoint dict, got {type(payload)}")

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    elif "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]

    cleaned = OrderedDict()
    for key, value in payload.items():
        if any(key.startswith(skip) for skip in ("loss_fn.", "criterion.")):
            continue
        cleaned[_strip_prefix(key)] = value
    return cleaned


def load_weights(model, path, map_location="cpu"):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    state_dict = extract_state_dict(payload)
    result = model.load_state_dict(state_dict, strict=False)

    missing = [k for k in result.missing_keys if "num_batches_tracked" not in k]
    unexpected = result.unexpected_keys
    loaded = len(state_dict) - len(unexpected)

    print(f"✅ Loaded weights from '{path}' ({loaded} tensors)")
    if missing:
        print(f"   Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"   Unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    classifier_missing = any(k.startswith("classifier.") for k in missing)
    return not classifier_missing


def freeze_feature_blocks(model, freeze_until_block):
    """Freeze features[0 .. freeze_until_block-1]. Classifier always stays trainable."""
    freeze_until_block = max(0, min(int(freeze_until_block), NUM_FEATURE_BLOCKS))
    for name, param in model.features.named_parameters():
        block_num = int(name.split(".", 1)[0])
        param.requires_grad = block_num >= freeze_until_block
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"🧊 Freeze blocks 0..{freeze_until_block - 1} | "
        f"trainable params: {trainable:,} | frozen: {frozen:,}"
    )


def param_groups(model, lr_backbone, lr_head, weight_decay):
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for name, param in model.named_parameters():
        # Include frozen tensors so unfreeze on later epochs can update them.
        target_decay = decay_head if name.startswith("classifier") else decay_backbone
        target_no_decay = no_decay_head if name.startswith("classifier") else no_decay_backbone
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            target_no_decay.append(param)
        else:
            target_decay.append(param)

    return [
        {"params": decay_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": decay_head, "lr": lr_head, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": lr_head, "weight_decay": 0.0},
    ]
