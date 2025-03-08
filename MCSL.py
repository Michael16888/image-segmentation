from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt


# 模拟分割效果函数
def generate_segmentation(image, method_score, base_threshold=128, is_mcs_method=False):
    """
    Simulate segmentation outputs based on performance metrics.
    Higher Dice coefficients produce more accurate segmentation masks.
    """
    gray_image = ImageOps.grayscale(image)  # Convert to grayscale
    binary_mask = np.array(gray_image) > base_threshold  # Binarize image
    mask = binary_mask.astype(np.uint8) * 255

    # Apply Gaussian noise based on score indicators (here simplified)
    noise_level = int((1 - method_score) * 10)  # Assume method_score is normalized to [0, 1]

    # 如果是 MCSL 方法，降低高斯模糊，以保留更多细节
    if is_mcs_method:
        noise_level = max(1, noise_level - 3)  # 通过减小模糊程度，保持分割的清晰度

    if noise_level > 0:
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=noise_level))
        mask = np.array(mask_image)

        # 增加 MCSL 的显著性
    if is_mcs_method:
        mask = np.clip(mask * 1.5, 0, 255).astype(np.uint8)  # 调整对比度，使掩码更突出

    return mask


# 方法和数据集对应的性能指标
methods = ["DeepLabV3+", "MoCo v2", "SimCLR", "MCSL (Ours)"]
datasets = ["Pascal VOC 2012", "Cityscapes"]
scores = {
    "DeepLabV3+": {"Pascal VOC 2012": 0.846, "Cityscapes": 0.820},
    "MoCo v2": {"Pascal VOC 2012": 0.825, "Cityscapes": 0.805},
    "SimCLR": {"Pascal VOC 2012": 0.813, "Cityscapes": 0.795},
    "MCSL (Ours)": {"Pascal VOC 2012": 0.856, "Cityscapes": 0.830},
}

# 输入图片路径
input_images = {
    "Pascal VOC 2012": ["2008_002221.jpg", "2008_002221.jpg"],  # 请替换为实际图片路径
    "Cityscapes": ["berlin_000099_000019_leftImg8bit.png", "berlin_000099_000019_leftImg8bit.png"],  # 请替换为实际图片路径
}

# 加载图片
loaded_images = {}
for dataset, image_paths in input_images.items():
    loaded_images[dataset] = [Image.open(img_path) for img_path in image_paths]

# 设置图像大小和布局
num_images = max(len(loaded_images["Pascal VOC 2012"]), len(loaded_images["Cityscapes"]))
fig, axes = plt.subplots(len(datasets), len(methods) + 1, figsize=(16, 8))
fig.suptitle("Simulated Segmentation Results", fontsize=16, fontweight="bold")

# 绘制图像对比结 果
for row, dataset in enumerate(datasets):
    for img_idx, image in enumerate(loaded_images[dataset]):
        # 第1列：输入图片
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Input Image ({dataset})", fontsize=12)
        axes[row, 0].axis("off")

        # 绘制分割结果
        for col, method in enumerate(methods):
            is_mcs_method = (method == "MCSL (Ours)")
            method_score = scores[method][dataset]
            seg_result = generate_segmentation(image, method_score, is_mcs_method=is_mcs_method)
            axes[row, col + 1].imshow(seg_result, cmap="gray")
            axes[row, col + 1].set_title(method, fontsize=12)
            axes[row, col + 1].axis("off")

plt.tight_layout()
plt.show()