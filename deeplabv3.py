import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import numpy as np

# 加载预训练的DeepLabV3+模型
model = deeplabv3_resnet101(pretrained=True, progress=True)
model = model.eval()

# 加载图像
image_path = '2010_001806.jpg'  # 替换为你的图片路径
image = Image.open(image_path).convert("RGB")

# 图像预处理
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加batch维度

# 如果有GPU，将模型和输入数据移动到GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    input_batch = input_batch.to('cuda')

# 进行推理
with torch.no_grad():
    output = model(input_batch)['out'][0]

# 将输出转换为分割图
output_predictions = output.argmax(0).byte().cpu().numpy()

# 定义COCO-Stuff的调色板（171类）
def coco_stuff_palette():
    palette = [
        0, 0, 0,  # 0: background
        128, 0, 0,  # 1: class 1
        0, 128, 0,  # 2: class 2
        128, 128, 0,  # 3: class 3
        0, 0, 128,  # 4: class 4
        128, 0, 128,  # 5: class 5
        0, 128, 128,  # 6: class 6
        128, 128, 128,  # 7: class 7
        64, 0, 0,  # 8: class 8
        192, 0, 0,  # 9: class 9
        64, 128, 0,  # 10: class 10
        # 添加更多颜色...
    ]
    # 如果类别超过预定义的颜色，随机生成颜色
    while len(palette) < 256 * 3:
        palette.extend([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
    return palette

# 将分割图映射到颜色
palette = coco_stuff_palette()
segmented_image = Image.fromarray(output_predictions, mode='P')
segmented_image.putpalette(palette)

# 保存分割结果
segmented_image.save('segmented_image.png')

print("分割结果已保存为 'segmented_image.png'")