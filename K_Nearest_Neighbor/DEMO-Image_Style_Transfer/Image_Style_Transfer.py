import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt



# -----------------------------
# 配置参数 (Configuration)
# -----------------------------

# 输入输出路径
CONTENT_IMAGE_PATH = "K_Nearest_Neighbor/data/image1.jpg"      # 待上色的黑白内容图
STYLE_IMAGE_PATH = "K_Nearest_Neighbor/data/style.jpg"         # 提供色彩风格的彩色图
OUTPUT_IMAGE_PATH = "K_Nearest_Neighbor/data/output_colorized.png"   # 输出的彩色结果图

# KNN 参数
K_NEIGHBORS = 5          # 每个窗口匹配 K 个最近邻
WINDOW_SIZE = 3          # 局部窗口大小 (3x3)
WEIGHTING = 'distance'   # 距离加权方式 ('mean', 'distance')

# 边界填充方式（用于处理边缘像素）
PADDING_MODE = 'reflect' # 'reflect', 'constant', 'edge' 等

# 是否显示进度条（适用于大图）
SHOW_PROGRESS = True

# -----------------------------
# 工具函数定义 (Utility Functions)
# -----------------------------

def load_and_preprocess_images(content_path, style_path):
    """
    加载并预处理输入图像：
    - 内容图：确保是灰度图（单通道）
    - 风格图：保留三通道彩色图，并生成其灰度版本用于匹配
    """
    # 读取内容图像（灰度图）
    content_gray = cv2.imread(content_path, cv2.IMREAD_GRAYSCALE)
    if content_gray is None:
        raise FileNotFoundError(f"无法加载内容图像: {content_path}")
    
    # 读取风格图像（彩色图）
    style_color = cv2.imread(style_path, cv2.IMREAD_COLOR)
    if style_color is None:
        raise FileNotFoundError(f"无法加载风格图像: {style_path}")
    
    # 将风格图转换为灰度图（用于结构匹配）
    style_gray = cv2.cvtColor(style_color, cv2.COLOR_BGR2GRAY)
    
    print(f"[INFO] 内容图像尺寸: {content_gray.shape}")
    print(f"[INFO] 风格图像尺寸: {style_color.shape}")
    
    return content_gray, style_color, style_gray


def extract_windows(image, window_size=3):
    """
    从图像中提取所有可能的局部窗口（滑动窗口）
    返回：
        windows: 所有窗口的扁平化数组，形状为 (N, window_size*window_size)
        positions: 每个窗口对应的中心像素坐标列表 [(y, x), ...]
    """
    h, w = image.shape[:2]
    pad = window_size // 2  # 填充半径
    
    # 边界填充，避免边缘丢失
    if len(image.shape) == 2:
        padded = np.pad(image, pad, mode=PADDING_MODE)
        # 返回一个数组，在图像四周填充 pad 个像素：[424,650] -> [426,652]
    else:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode=PADDING_MODE)
        # 对彩色图也进行填充，保持通道数不变
    
    windows = []
    positions = []
    
    # 遍历所有中心像素位置（原图范围）
    for y in range(h):
        for x in range(w):
            win = padded[y:y+window_size, x:x+window_size]
            # 数组切片，提取以 (y,x) 为中心的 window_size x window_size 窗口,3*3
            if len(win.shape) == 3:  # 彩色图
                win_flat = win.reshape(-1)  # 展平为一维
            else:  # 灰度图
                win_flat = win.flatten() # （9,）
            windows.append(win_flat) # [[], [], [], []---]每个位置是对应中心像素点所在的三乘三窗口内的9个数
            positions.append((y, x)) # [(0,1), (0,2), (0,3), ()---]
    return np.array(windows), positions# [N, 9],[N,]


def knn_color_transfer(content_gray, style_color, style_gray, k=5, window_size=3):
    """
    核心函数：使用 KNN 进行基于局部窗口的颜色迁移
    输入：
        content_gray: 待上色的灰度图 (H, W)
        style_color: 风格彩色图 (H_s, W_s, 3)
        style_gray: 风格灰度图 (H_s, W_s)
        k: K近邻数量
        window_size: 局部窗口大小
    输出：
        colorized: 上色后的彩色图 (H, W, 3)
    """
    h, w = content_gray.shape
    
    # 1. 提取内容图的所有 3x3 窗口（灰度）
    print("[INFO] 正在提取内容图像的局部窗口...")
    content_windows, content_positions = extract_windows(content_gray, window_size)
    # [275600,9] [275600,2]
    print(f"[INFO] 共提取 {len(content_windows)} 个窗口")
    
    # 2. 提取风格图的所有 3x3 窗口（灰度）和对应的中心像素颜色
    print("[INFO] 正在提取风格图像的局部窗口及对应颜色...")
    style_windows, style_positions = extract_windows(style_gray, window_size)
    # [4096000, 9] [4096000, 2]
    
    # 获取每个窗口中心像素在原彩色图中的颜色值
    style_colors = []
    for y, x in style_positions:
        color = style_color[y, x]  # BGR 格式 # 得到该位置的三个通道的颜色值 [B, G, R]
        style_colors.append(color)
    style_colors = np.array(style_colors)  # (N_s, 3) [4096000, 3]
    
    # 3. 使用 sklearn 的 NearestNeighbors 进行高效 KNN 搜索
    print("[INFO] 正在构建 KNN 模型并进行匹配...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1, algorithm='auto')
    nbrs.fit(style_windows)  # 用风格图像中的窗口训练模型
    
    # 4. 对内容图每个窗口，查找 K 个最近邻风格窗口
    distances, indices = nbrs.kneighbors(content_windows)
    # indices: 每个内容窗口对应的 K 个风格窗口索引 [275600, 5]
    
    # 5. 初始化输出图像
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 6. 逐像素上色
    total_pixels = len(content_positions) # 275600
    for i, (y, x) in enumerate(content_positions):
        if SHOW_PROGRESS and i % (total_pixels // 10) == 0:
            print(f"[PROGRESS] 处理进度: {i}/{total_pixels} ({100*i//total_pixels}%)")
        
        # 获取该窗口匹配到的 K 个风格窗口的索引
        matched_indices = indices[i] # [5,]
        
        # 获取这 K 个匹配窗口中心像素的颜色
        matched_colors = style_colors[matched_indices]  # (K, 3)

        if WEIGHTING == 'distance':
        # 获取该窗口匹配到的 K 个风格窗口的距离
            matched_distances = distances[i]  # (K,) 形状

            # 计算权重：距离越近权重越大（使用距离倒数）
            # 添加小的epsilon避免除零
            epsilon = 1e-8
            weights = 1.0 / (matched_distances + epsilon)

            # 归一化权重，使权重和为1
            weights = weights / np.sum(weights)

            # 加权平均颜色
            avg_color = np.average(matched_colors, axis=0, weights=weights).astype(np.uint8)
        else:
            # 计算平均颜色（简单平均）
            avg_color = np.mean(matched_colors, axis=0).astype(np.uint8)
        
        # 赋值给输出图像
        colorized[y, x] = avg_color
    
    print("[INFO] 颜色迁移完成！")
    return colorized


def main():
    """
    主函数：执行整个着色流程
    """
    try:
        # 1. 加载并预处理图像
        content_gray, style_color, style_gray = load_and_preprocess_images(
            CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH)
        
        # 2. 执行 KNN 颜色迁移
        colorized_image = knn_color_transfer(
            content_gray, style_color, style_gray,
            k=K_NEIGHBORS, window_size=WINDOW_SIZE)
        
        # 3. 保存结果
        cv2.imwrite(OUTPUT_IMAGE_PATH, colorized_image)
        print(f"[SUCCESS] 成功保存着色图像至: {OUTPUT_IMAGE_PATH}")
        
        # 4. 使用plt显示原始灰色图、风格图像、结果图像
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.title("Original Gray Image")
        plt.imshow(content_gray, cmap='gray')
        plt.axis('off')
        
        plt.subplot(132)
        plt.title("Style Color Image")
        plt.imshow(style_color)
        plt.axis('off')
        
        plt.subplot(133)
        plt.title("Colorized Result")
        plt.imshow(colorized_image)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
            
    except Exception as e:
        print(f"[ERROR] 发生错误: {str(e)}")
        raise


# -----------------------------
# 程序入口
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("KNN 图像着色脚本启动")
    print("=" * 60)
    main()