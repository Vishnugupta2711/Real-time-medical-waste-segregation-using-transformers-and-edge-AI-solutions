# webcam_inference.py
import cv2
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from collections import OrderedDict

# --------- Config ----------
MODEL_PATH = "best_model.pth"   # Adjust if file is elsewhere
CAM_INDEX = 0                   # 0 is default laptop camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_THRESHOLD = 0.3            # don't show predictions with < 30% confidence (optional)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this list with your actual CLASSES in the same order used to build label2id
CLASSES = [
    '(BT) Body Tissue or Organ',
    '(GE) Glass equipment-packaging 551',
    '(ME) Metal equipment -packaging',
    '(OW) Organic wastes',
    '(PE) Plastic equipment-packaging',
    '(PP) Paper equipment-packaging',
    '(SN) Syringe needles',
    '.DS_Store',     # if you still have this, ideally remove; keep position consistent with label2id
    'Gauze',
    'Gloves',
    'Mask',
    'Syringe',
    'Tweezers'
]
# If you removed ".DS_Store" during training, remove it here too and ensure indices match.

# --------- Build model (same architecture as training) ----------
def build_model(num_classes=len(CLASSES)):
    # Use pretrained weights to get same normalization and architecture. This will download if missing.
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Replace the head exactly how you trained it.
    # If you used just nn.Linear(768, num_classes) during training, use the same here.
    model.heads.head = nn.Linear(in_features=768, out_features=num_classes)
    return model, weights

# --------- Load model weights ----------
model, weights = build_model(num_classes=len(CLASSES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)   # assumes you saved model.state_dict()
model.to(DEVICE)
model.eval()

# Get transforms from the same pretrained weights (ensures same normalization & resizing)
transform = weights.transforms()  # this is a callable that accepts PIL image and returns tensor

# --------- Camera setup ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try CAMERA index 1 or check permissions.")

font = cv2.FONT_HERSHEY_SIMPLEX

print(f"Running inference on {DEVICE}. Press 'q' to quit.")

# --------- Main loop ----------
with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # apply transforms (this returns a tensor shaped [C,H,W])
        x = transform(pil_img).unsqueeze(0).to(DEVICE)  # shape [1,3,224,224]

        # forward
        logits = model(x)                 # shape [1, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # 1D np array
        pred_idx = int(probs.argmax())
        pred_conf = float(probs[pred_idx])

        label_text = f"{CLASSES[pred_idx]}: {pred_conf:.2f}"

        # overlay on frame: draw label and bounding rectangle
        text_pos = (10, 30)
        cv2.putText(frame, label_text, text_pos, font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Optionally display top-3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        y0 = 60
        for i, idx in enumerate(top3_idx):
            txt = f"{i+1}. {CLASSES[idx]}: {probs[idx]:.2f}"
            cv2.putText(frame, txt, (10, y0 + i*25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ViT Webcam Inference", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

