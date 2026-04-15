"""
app.py
------
Flask web server for the SDNN inference demo.

Run:
    python app.py          (auto-selects GPU/MPS/CPU)
    python app.py --port 8080
"""

import argparse
import io
import json
import os
import sys

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.checkpoint_utils import load_sdnn_checkpoint
from models.sdnn_model import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "best_sdnn.pth")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
RELIABILITY_THRESHOLD = 0.30   # error_prob above this → trigger safe fallback

# CIFAR-10 class emojis for the UI
CLASS_EMOJIS = {
    "airplane":   "✈️",
    "automobile": "🚗",
    "bird":       "🐦",
    "cat":        "🐱",
    "deer":       "🦌",
    "dog":        "🐶",
    "frog":       "🐸",
    "horse":      "🐴",
    "ship":       "🚢",
    "truck":      "🚛",
}

# Autonomous-systems context per class
CLASS_CONTEXT = {
    "airplane":   "Airspace / drone detection",
    "automobile": "Ego / surrounding vehicle",
    "bird":       "Road hazard — animal",
    "cat":        "Pedestrian-adjacent obstacle",
    "deer":       "Road hazard — animal",
    "dog":        "Pedestrian-adjacent obstacle",
    "frog":       "Pedestrian-adjacent obstacle",
    "horse":      "Road hazard — animal",
    "ship":       "Maritime / port environment",
    "truck":      "Large vehicle detection",
}

# Normalisation that matches training (CIFAR-10 dataset mean/std)
_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Device selection
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    _DEVICE = torch.device("mps")
else:
    _DEVICE = torch.device("cpu")

print(f"[SDNN] Loading checkpoint from: {CHECKPOINT_PATH}")
print(f"[SDNN] Using device: {_DEVICE}")

_MODEL, _TEMPERATURE, _META = load_sdnn_checkpoint(CHECKPOINT_PATH, device=_DEVICE)
_MODEL.eval()

print(f"[SDNN] Model loaded. Checkpoint meta: {_META}")
print(f"[SDNN] Temperature: {_TEMPERATURE}")
print(f"[SDNN] Ready — listening for requests...")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 415

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    # Pre-process
    tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)  # (1, 3, 32, 32)

    # Inference
    with torch.no_grad():
        out = _MODEL(tensor)

    logits     = out["logits"][0]       # (10,)
    sdnn_confidence = float(out["confidence"][0, 0])
    error_prob = float(out["error_prob"][0, 0])

    # Apply temperature scaling if available
    if _TEMPERATURE is not None:
        logits = logits / float(_TEMPERATURE)

    softmax_probs = torch.softmax(logits, dim=0)
    confidence = float(torch.max(softmax_probs))
    
    # Calculate entropy
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = float(-torch.sum(softmax_probs * torch.log(softmax_probs + epsilon)))
    
    softmax_probs_list = softmax_probs.cpu().tolist()

    predicted_idx  = int(torch.argmax(logits).item())
    predicted_class = CIFAR10_CLASSES[predicted_idx]
    max_softmax    = float(softmax_probs_list[predicted_idx])

    # Add an uncertainty threshold
    ENTROPY_THRESHOLD = 1.5
    ERROR_THRESHOLD = 0.4
    
    is_ood = confidence < 0.7 or entropy > ENTROPY_THRESHOLD or error_prob > ERROR_THRESHOLD

    # Print out the stats
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence:.4f}")
    print(f"Entropy: {entropy:.4f}")
    print(f"Error Probability: {error_prob:.4f}")
    print(f"Final Decision: {'Rejected / OOD' if is_ood else 'Accepted'}")

    if is_ood:
        final_class = "Unknown / Out-of-Distribution"
        emoji = "❓"
        context = "Prediction rejected due to high uncertainty or OOD"
    else:
        final_class = predicted_class
        emoji = CLASS_EMOJIS[predicted_class]
        context = CLASS_CONTEXT[predicted_class]

    is_reliable  = not is_ood
    reliability  = "reliable" if is_reliable else "unsafe"

    response = {
        "predicted_class": final_class,
        "predicted_idx":   predicted_idx,
        "emoji":           emoji,
        "context":         context,
        "confidence":      round(confidence, 4),
        "entropy":         round(entropy, 4),
        "error_prob":      round(error_prob, 4),
        "max_softmax":     round(max_softmax, 4),
        "reliability":     reliability,
        "threshold":       ERROR_THRESHOLD,
        "classes":         CIFAR10_CLASSES,
        "softmax_probs":   [round(p, 4) for p in softmax_probs_list],
    }

    return jsonify(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDNN Flask inference server")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
