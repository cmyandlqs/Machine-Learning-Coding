# 机器学习项目集合

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-算法实现-green.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-图像处理-orange.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目概述

大家好，我是最菜灰夫人，这是一个适合新手的机器学习项目集合，涵盖了多种经典的机器学习算法和计算机视觉技术。项目采用工程化设计，包含完整的算法实现、数据可视化、性能评估和详细文档。每个子项目都是独立的模块，可以单独运行和学习。

## 🎯 项目特色

- ✅ **算法多样性**：涵盖监督学习、无监督学习、计算机视觉等多个领域
- ✅ **工程化设计**：模块化代码结构，参数化配置，易于扩展
- ✅ **完整实现**：从数据预处理到模型评估的完整流程
- ✅ **丰富可视化**：数据分布、训练过程、结果展示等全方位可视化
- ✅ **详细文档**：每个项目都有完整的 README 和使用说明
- ✅ **工具支持**：包含依赖分析、环境配置等实用工具

## 🏗️ 项目结构

```
Machine-Learning-Coding/
├── README.md                           # 项目总览文档
├── Tool/                               # 工具模块
│   └── analyze_dependencies.py         # 依赖分析工具
├── Linear_Regression/                   # 线性回归项目
│   ├── DEMO-classification.py          # 分类演示
│   ├── lr_dataset.csv                  # 数据集
│   └── README.md                       # 项目文档
├── Logist_Regression/                   # 逻辑回归项目
│   ├── DEMO-classification.py          # 梯度下降实现
│   ├── lr_dataset.csv                  # 数据集
│   ├── plots/                          # 结果图片
│   │   ├── data_distribution.png        # 数据分布图
│   │   ├── training_process.png        # 训练过程图
│   │   ├── roc_curve.png              # ROC曲线图
│   │   └── decision_boundary.png       # 决策边界图
│   └── README.md                       # 项目文档
└── K_Nearest_Neighbor/                 # K近邻算法项目
    ├── data/                           # 数据目录
    │   ├── Gray_Images/               # 灰度图像
    │   ├── RBG_Images/                # 彩色图像
    │   ├── vangogh/                   # 梵高风格图像
    │   └── style.jpg                  # 风格参考图
    └── DEMO-Image_Style_Transfer/     # 图像风格迁移
        ├── Image_Style_Transfer.py    # 主程序
        ├── requirements.txt           # 依赖文件
        └── README.md                  # 项目文档
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.7+
- **NumPy**: 1.19+
- **Pandas**: 1.3+
- **Matplotlib**: 3.5+
- **Seaborn**: 0.11+
- **Scikit-learn**: 1.0+
- **OpenCV**: 4.5+
- **PIL**: 8.0+

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/Machine-Learning-Coding.git
cd Machine-Learning-Coding

# 安装基础依赖
pip install numpy pandas matplotlib seaborn scikit-learn

# 安装计算机视觉依赖
pip install opencv-python pillow

# 或使用conda安装
conda install numpy pandas matplotlib seaborn scikit-learn opencv pillow
```

### 运行项目

```bash
# 运行逻辑回归项目
cd Logist_Regression
python DEMO-classification.py

# 运行图像风格迁移项目
cd K_Nearest_Neighbor/DEMO-Image_Style_Transfer
python Image_Style_Transfer.py

```

## 📚 项目详情

### 1. 逻辑回归分类 (Logist_Regression)

**项目描述**：实现基于梯度下降的逻辑回归二分类算法，包含完整的训练监控和可视化功能。

**核心特性**：

- 自定义梯度下降实现
- 实时训练监控（损失、准确率、AUC）
- 决策边界可视化
- ROC 曲线分析
- 自动图片保存

**技术栈**：

- NumPy, Pandas (数据处理)
- Matplotlib, Seaborn (可视化)
- Scikit-learn (评估指标)

**快速体验**：

```bash
cd Logist_Regression
python DEMO-classification.py
```

### 2. 图像风格迁移 (K_Nearest_Neighbor)

**项目描述**：基于 K 近邻算法的图像风格迁移技术，能够为黑白图像自动上色，模仿指定风格图像。

**核心特性**：

- 智能颜色迁移
- 高效 KNN 搜索
- 距离加权融合
- 进度可视化
- 结果对比展示

**技术栈**：

- OpenCV (图像处理)
- Scikit-learn (KNN 算法)
- PIL (图像操作)
- NumPy (数值计算)

**快速体验**：

```bash
cd K_Nearest_Neighbor/DEMO-Image_Style_Transfer
python Image_Style_Transfer.py
```



## 📚 学习资源

### 算法原理

- **逻辑回归**：梯度下降、交叉熵损失、Sigmoid 函数
- **K 近邻**：距离计算、近邻搜索、加权平均
- **图像处理**：局部特征、结构相似性、颜色空间

### 技术文档

- [Scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [OpenCV 官方文档](https://docs.opencv.org/)
- [Matplotlib 官方文档](https://matplotlib.org/stable/)

## 🤝 贡献指南

1. **Fork 本项目**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **开启 Pull Request**

### 贡献类型

- 🐛 Bug 修复
- ✨ 新功能添加
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试用例


## 👥 作者

- **开发者** - [最菜灰夫人](https://github.com/cmyandlqs)
- **项目链接** - [https://github.com/cmyandlqs/Machine-Learning-Coding](https://github.com/cmyandlqs/Machine-Learning-Coding)

## 🙏 致谢

- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [NumPy](https://numpy.org/) - 数值计算
- [Pandas](https://pandas.pydata.org/) - 数据处理

## 📞 联系方式

- **邮箱**：2395599123@qq.com
- **Bilibili**：[https://space.bilibili.com/563285166?spm_id_from=333.1007.0.0](https://space.bilibili.com/563285166?spm_id_from=333.1007.0.0)

---

**⭐ 如果这个项目对你有帮助，请给它一个星标！**

