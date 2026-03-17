"""
checkpoint_utils.py
-------------------
Universal loader for SDNN checkpoints.

Handles two formats:
  Format A (training/train.py):       key = 'model_state'
  Format B (calibrated_sdnn_final):   key = 'model_state_dict'

Also supports an optional stored temperature value for
temperature-scaled checkpoints.
"""

import torch
from models.sdnn_model import SDNN


def load_sdnn_checkpoint(path: str, device: torch.device = None, num_classes: int = 10):
    """
    Load an SDNN checkpoint regardless of which key the state dict is stored under.

    Returns:
        model       – SDNN instance with weights loaded, in eval mode
        temperature – float or None (if checkpoint was temperature-scaled)
        meta        – dict of any other checkpoint metadata (epoch, val_loss, etc.)
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # --- Resolve state dict key ---
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # Maybe the file IS the state dict directly
        state_dict = ckpt

    # --- Load model ---
    model = SDNN(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # --- Temperature (post-hoc calibration scalar) ---
    temperature = ckpt.get("temperature", None)

    # --- Other metadata ---
    meta = {k: v for k, v in ckpt.items()
            if k not in ("model_state", "model_state_dict", "temperature")}

    return model, temperature, meta
