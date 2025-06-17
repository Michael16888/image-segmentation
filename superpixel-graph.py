import matplotlib.pyplot as plt
from skimage import io, segmentation, color, graph
from skimage.util import img_as_float
import numpy as np

# 读取图片
image = io.imread('ADE_train_00001037.jpg')
image = img_as_float(image)

# 超像素生成
segments = segmentation.slic(image, n_segments=200, compactness=10, sigma=1, start_label=1)

# 构建RAG
rag = graph.rag_mean_color(image, segments)

# 可视化
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# 输入图
axs[0].imshow(image)
axs[0].set_title("Input Image")
axs[0].axis('off')

# 超像素边界
axs[1].imshow(segmentation.mark_boundaries(image, segments))
axs[1].set_title("Superpixel Image")
axs[1].axis('off')

# 超像素图 (节点+边)
axs[2].imshow(image)
axs[2].set_title("Superpixel Graph")

# 在超像素图上画RAG边和节点
for edge in rag.edges:
    n1 = edge[0]
    n2 = edge[1]

    r1, c1 = np.mean(np.argwhere(segments == n1), axis=0)
    r2, c2 = np.mean(np.argwhere(segments == n2), axis=0)

    axs[2].plot([c1, c2], [r1, r2], color='red', linewidth=1)

# 画节点
for node in rag.nodes:
    r, c = np.mean(np.argwhere(segments == node), axis=0)
    axs[2].plot(c, r, 'bo', markersize=4)

axs[2].axis('off')

plt.tight_layout()
plt.show()
