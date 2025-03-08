import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的DeepLabV3+模型
model = deeplabv3_resnet101(pretrained=True)
model.eval()  # 设置为评估模式

# 如果有GPU，将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
image_path = "2008_002221.jpg"
image = Image.open(image_path).convert("RGB")

# 预处理图像
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)  # 添加batch维度并移到设备上

# 进行推理
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# 将预测结果转换为彩色图像
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                            (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 将预测结果转换为彩色图像
rgb = decode_segmap(output_predictions.cpu().numpy())

# 显示原始图像和分割结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb)
plt.title("Segmentation")
plt.axis('off')

# plt.show()

# 保存分割结果
segmentation_image = Image.fromarray(rgb)
segmentation_image.save("deeplabv3_result.jpg")