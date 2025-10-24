# KNN图像风格迁移项目

## 📖 项目简介

本项目实现了一种基于K近邻（K-Nearest Neighbors, KNN）算法的图像风格迁移技术，能够为黑白图像自动上色，使其色彩风格模仿指定的彩色风格图像。该算法通过分析图像的局部结构特征，在风格图像中寻找最相似的区域，并将对应的颜色信息迁移到内容图像中。

## ✨ 核心特性

- **智能颜色迁移**：基于局部窗口结构相似性进行精确的颜色匹配
- **高效KNN搜索**：使用scikit-learn的NearestNeighbors实现快速近邻搜索
- **距离加权融合**：支持距离加权平均，提高颜色迁移质量
- **灵活参数配置**：可调节窗口大小、近邻数量等关键参数
- **进度可视化**：实时显示处理进度，支持大图像处理
- **结果对比展示**：自动生成原图、风格图、结果图的对比展示

## 🎯 算法原理

### 核心思想
通过在风格图像中寻找与内容图像局部结构最相似的区域，来为内容图像的每个像素"借"颜色。

### 实现步骤
1. **图像预处理**：将内容图像转换为灰度图，保留风格图像的彩色信息
2. **局部窗口提取**：以每个像素为中心，提取3×3的局部窗口
3. **结构匹配**：使用欧氏距离计算内容窗口与风格窗口的相似性
4. **K近邻搜索**：为每个内容窗口找到K个最相似的风格窗口
5. **颜色融合**：对匹配窗口的中心像素颜色进行距离加权平均
6. **结果生成**：将融合后的颜色赋值给内容图像的对应像素

## 🛠️ 技术栈

- **Python 3.7+**
- **OpenCV** - 图像处理和I/O操作
- **NumPy** - 数值计算和数组操作
- **scikit-learn** - KNN算法实现
- **Matplotlib** - 结果可视化

## 📦 安装依赖

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd Machine-Learning-Coding/K_Nearest_Neighbor/DEMO-Image_Style_Transfer
```

### 2. 准备数据
将您的图像文件放置在 `K_Nearest_Neighbor/data/` 目录下：
- `image1.jpg` - 待上色的黑白内容图像
- `style.jpg` - 提供色彩风格的彩色图像

### 3. 运行程序
```bash
python Image_Style_Transfer.py
```

### 4. 查看结果
程序将自动：
- 处理图像并生成着色结果
- 保存结果到 `K_Nearest_Neighbor/data/output_colorized.png`
- 显示原图、风格图、结果图的对比展示

## ⚙️ 配置参数

在 `Image_Style_Transfer.py` 文件顶部，您可以调整以下参数：

```python
# 输入输出路径
CONTENT_IMAGE_PATH = "K_Nearest_Neighbor/data/image1.jpg"      # 待上色的黑白内容图
STYLE_IMAGE_PATH = "K_Nearest_Neighbor/data/style.jpg"         # 提供色彩风格的彩色图
OUTPUT_IMAGE_PATH = "K_Nearest_Neighbor/data/output_colorized.png"   # 输出的彩色结果图

# KNN 参数
K_NEIGHBORS = 5          # 每个窗口匹配 K 个最近邻
WINDOW_SIZE = 3          # 局部窗口大小 (3x3)
WEIGHTING = 'distance'   # 距离加权方式 ('uniform', 'distance')

# 边界填充方式
PADDING_MODE = 'reflect' # 'reflect', 'constant', 'edge' 等

# 是否显示进度条
SHOW_PROGRESS = True
```

### 参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `K_NEIGHBORS` | K近邻数量 | 3-10 | 值越大颜色越平滑，但可能丢失细节 |
| `WINDOW_SIZE` | 局部窗口大小 | 3, 5, 7 | 值越大考虑更多上下文，但计算量增加 |
| `WEIGHTING` | 距离加权方式 | 'distance' or 'mean' | 距离越近权重越大，颜色迁移更精确,或者平均权 |
| `PADDING_MODE` | 边界填充方式 | 'reflect' | 处理图像边缘像素的填充策略 |


