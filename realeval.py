import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from network import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES, build_efficientnet_b0, load_weights

weight_path = "models/fine_tuned.pt"
if not os.path.exists(weight_path):
    weight_path = "models/best_model.pt"

model = build_efficientnet_b0(dropout=0.4)
load_weights(model, weight_path, map_location="cpu")
model.eval()

normalize = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def distort(image, simulate=True):
    if simulate:
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.uint8)

        if np.random.rand() < 0.5:
            arr = cv2.GaussianBlur(arr, (5, 5), 0)

        if np.random.rand() < 0.5:
            _, arr = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            arr = cv2.imdecode(arr, 1)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(arr)

    return normalize(image).unsqueeze(0)


def evaluate(folder="realworld_samples/", simulate_noise=False):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        try:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image = Image.open(path).convert("RGB")
                tensor = distort(image, simulate=simulate_noise)
            elif file.lower().endswith((".mp4", ".mov")):
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    print(f"{file}: ❌ Error reading video")
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = distort(image, simulate=simulate_noise)
            else:
                print(f"{file}: Unsupported type")
                continue

            with torch.no_grad():
                out = model(tensor)
                prob = torch.softmax(out, dim=1)[0]
                pred = torch.argmax(prob).item()
                conf = prob[pred].item()

            print(f"{file:<30} ➤ {CLASS_NAMES[pred]:<6} ({conf * 100:.2f}%)")

        except Exception as e:
            print(f"{file}: ⚠️ {e}")


if __name__ == "__main__":
    evaluate(simulate_noise=False)
