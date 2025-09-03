# -*- coding: utf-8 -*-
# 直接运行此脚本即可；在下方修改 IMAGE_PATH 和 MODEL_NAME。
# 依赖：pip install torch torchvision transformers pillow numpy

import os
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, AutoConfig

# ======= 在这里改成你的图片路径与模型 =======
IMAGE_PATH = "images/ADE_val_00000473.jpg"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"  # 也可用 b2/b5 等
OUTDIR = "outputs"
ALPHA = 0.6  # 叠加透明度[0,1]
# ======================================

def load_model(model_name: str, device: torch.device):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    id2label = AutoConfig.from_pretrained(model_name).id2label
    return processor, model, id2label

def create_palette(num_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)  # 背景置黑
    return palette

def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> Image.Image:
    h, w = mask.shape
    color = palette[mask.reshape(-1)].reshape(h, w, 3)
    return Image.fromarray(color)

def overlay(image: Image.Image, color_mask: Image.Image, alpha: float = 0.6) -> Image.Image:
    return Image.blend(image.convert("RGBA"), color_mask.convert("RGBA"), alpha=alpha).convert("RGB")

@torch.no_grad()
def infer(processor, model, image: Image.Image, device: torch.device) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits  # [1, C, h, w]
    up = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    return up.argmax(1)[0].cpu().numpy()  # (H, W)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] image: {IMAGE_PATH}")
    print(f"[INFO] model: {MODEL_NAME}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    processor, model, id2label = load_model(MODEL_NAME, device)
    num_classes = len(id2label)

    mask = infer(processor, model, image, device)
    palette = create_palette(num_classes)
    color_mask = colorize_mask(mask, palette)
    overlay_img = overlay(image, color_mask, alpha=ALPHA)

    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    mask_png = os.path.join(OUTDIR, f"{base}_mask.png")
    overlay_png = os.path.join(OUTDIR, f"{base}_overlay.png")
    mask_npy = os.path.join(OUTDIR, f"{base}_mask.npy")

    color_mask.save(mask_png)
    overlay_img.save(overlay_png)
    np.save(mask_npy, mask)

    # 简要统计
    uniq, cnt = np.unique(mask, return_counts=True)
    print("[INFO] Class pixel distribution:")
    for k, c in zip(uniq.tolist(), cnt.tolist()):
        print(f"  id={k:<3} label={id2label.get(k, str(k)):<25} pixels={c}")
    print(f"[DONE] Saved:\n  {mask_png}\n  {overlay_png}\n  {mask_npy}")

if __name__ == "__main__":
    main()
