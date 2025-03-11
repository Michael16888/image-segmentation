import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像COCO-STUFF IMAGE
image_path = "000000010327.jpg"
image = Image.open(image_path).convert("RGB")

# 预处理（尝试去掉 Normalize 以匹配 COCO-Stuff 归一化）
transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 可能不适用 COCO-Stuff
])
input_tensor = transform(image).unsqueeze(0)

# 加载 DeepLabV3+ 模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True)  # 仍然是 COCO 21 类
model.eval()

# 预测
with torch.no_grad():
    output = model(input_tensor)["out"][0]
pred_mask = torch.argmax(output, dim=0).byte().cpu().numpy()

# **关键步骤：检查唯一类别值**
unique_labels = np.unique(pred_mask)
print(f"Predicted Unique Labels: {unique_labels}")

# 显示结果
plt.figure(figsize=(10, 5))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 预测分割结果
plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="jet")
plt.title("Predicted Segmentation")
plt.axis("off")

plt.show()
