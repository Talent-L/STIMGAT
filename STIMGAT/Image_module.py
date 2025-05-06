import numpy as np
from PIL import Image

def get_image(adata, img_key="hires"):
    """
    提取每个空间坐标点对应图像上的RGB值，并将这些RGB值存储为矩阵。
    """

    library_id = list(adata.uns["spatial"].keys())[0]
    spatial_coords = adata.obsm["spatial"]  # shape: (n_cells, 2)
    image = adata.uns["spatial"][library_id]["images"][img_key]  # 图像数据 (H, W, 3)
    
    # 如果图像是路径，需要加载图像
    if isinstance(image, str):
        image = np.array(Image.open(image))
    
    # 获取图像尺寸和缩放因子
    image_height, image_width, _ = image.shape
    scaling_factor = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]  # 缩放因子
    pixel_coords = (spatial_coords * scaling_factor).astype(int)  # 转换为像素坐标 (n_cells, 2)
    pixel_coords = np.clip(pixel_coords, 0, [image_height - 1, image_width - 1]) # 限制在图像内
    
    # 提取RGB值，结果是一个 (n_cells, 3) 的矩阵
    rgb_matrix = image[pixel_coords[:, 1], pixel_coords[:, 0]]  # 使用 (y, x) 索引顺序
    adata.uns["spatial"][library_id]["rgb_matrix"] = rgb_matrix
    
    return rgb_matrix