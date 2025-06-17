
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from skimage import graph
from skimage.util import img_as_float
import numpy as np

# 读取图片
image = io.imread('ADE_train_00001037.jpg')
image = img_as_float(image)

# 超像素生成 (SLIC)
segments = segmentation.slic(image, n_segments=200, compactness=10, sigma=1, start_label=1)

# 绘制超像素图
superpixel_image = color.label2rgb(segments, image, kind='avg')

# 构建图结构
rag = graph.rag_mean_color(image, segments)

# 绘制输入图、超像素、超像素图
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# 输入图
axs[0].imshow(image)
axs[0].set_title('Input Image')
axs[0].axis('off')

# 超像素边界图
axs[1].imshow(segmentation.mark_boundaries(image, segments))
axs[1].set_title('Superpixel Image')
axs[1].axis('off')

# 超像素平均色图
axs[2].imshow(superpixel_image)
axs[2].set_title('Superpixel Graph (mean color)')
axs[2].axis('off')

# 节点特征图 (这里简单用灰度值平均可视化)
features = np.mean(superpixel_image, axis=2)
axs[3].imshow(features, cmap='gray')
axs[3].set_title('Node Feature')
axs[3].axis('off')

plt.tight_layout()
plt.show()

