# 逻辑回归分类 Demo - 梯度下降实现

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目概述

本项目实现了一个完整的二分类逻辑回归模型，使用**梯度下降算法**进行训练，包含数据可视化、模型训练、性能评估和结果展示等完整流程。项目采用工程化设计，支持参数配置、图片自动保存和详细的训练过程监控。

## 🎯 主要特性

- ✅ **自定义梯度下降实现**：从零实现逻辑回归的梯度下降算法
- ✅ **完整的训练监控**：实时显示损失、准确率、AUC 等指标
- ✅ **丰富的可视化**：数据分布、决策边界、ROC 曲线、训练过程
- ✅ **自动图片保存**：所有图表自动保存到指定目录
- ✅ **工程化设计**：模块化代码结构，参数化配置
- ✅ **详细文档**：完整的代码注释和使用说明

## 🏗️ 项目结构

```
Logist_Regression/
├── DEMO-classification.py          # 主程序文件
├── lr_dataset.csv                  # 训练数据集
├── plots/                          # 图片输出目录
│   ├── data_distribution.png       # 数据分布图
│   ├── training_process.png        # 训练过程图
│   ├── roc_curve.png              # ROC曲线图
│   └── decision_boundary.png       # 决策边界图
└── README.md                       # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- NumPy 1.19+
- Pandas 1.3+
- Matplotlib 3.5+
- Seaborn 0.11+
- Scikit-learn 1.0+

### 安装依赖

```bash
# 使用pip安装
pip install numpy pandas matplotlib seaborn scikit-learn

# 或使用conda安装
conda install numpy pandas matplotlib seaborn scikit-learn
```

### 运行程序

```bash
# 直接运行主程序
python DEMO-classification.py
```

## 📊 数据集格式

程序支持 CSV 格式的二分类数据集，要求：

- **3 列数据**：x 坐标, y 坐标, 标签
- **标签格式**：0 和 1（或可转换为 0/1 的其他值）
- **数据示例**：

```csv
0.4304,0.2055,1
0.0898,-0.1527,1
-0.8257,-0.9596,0
```

## ⚙️ 配置参数

### 模型参数

```python
TEST_SIZE = 0.2          # 测试集比例
RANDOM_STATE = 42        # 随机种子
SCALER_ENABLED = True    # 是否启用特征标准化
```

### 梯度下降参数

```python
LEARNING_RATE = 0.01     # 学习率
EPOCHS = 1000           # 训练轮数
BATCH_SIZE = 32         # 批次大小
PRINT_INTERVAL = 100    # 打印间隔
```

### 绘图参数

```python
FIGSIZE = (10, 6)        # 图表大小
DPI = 100                # 图表分辨率
SAVE_DIR = "plots"       # 图片保存目录
```

## 🔧 核心功能

### 1. 自定义逻辑回归类

```python
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, random_state=42)
    def fit(self, X, y)                    # 梯度下降训练
    def predict(self, X)                   # 预测类别
    def predict_proba(self, X)             # 预测概率
```

### 2. 训练过程监控

- **实时指标**：损失、准确率、AUC
- **训练曲线**：损失曲线、准确率曲线、AUC 曲线
- **收敛监控**：自动检测训练收敛情况

### 3. 可视化功能

- **数据分布图**：原始数据点的分布情况
- **决策边界**：模型分类边界的可视化
- **ROC 曲线**：模型性能的 ROC 分析
- **训练过程**：训练指标的变化趋势

## 📈 输出结果

### 控制台输出

```
============================================================
二分类任务 - 逻辑回归模型（梯度下降）
============================================================

[INFO] 成功加载数据，形状: (1000, 3)
[INFO] 开始梯度下降训练...
[INFO] 学习率: 0.01, 训练轮数: 1000

Epoch    1/1000 | Loss: 0.6931 | Acc: 0.5000 | AUC: 0.5000
Epoch  100/1000 | Loss: 0.4523 | Acc: 0.8125 | AUC: 0.8756
Epoch  200/1000 | Loss: 0.3214 | Acc: 0.8750 | AUC: 0.9234
...

--- 模型性能报告 ---
准确率 (Accuracy):  0.8750
精确率 (Precision): 0.8571
召回率 (Recall):    0.9000
F1 分数 (F1):       0.8780
AUC (Area Under Curve): 0.9234
```

### 图片输出

- `data_distribution.png` - 数据分布图
- `training_process.png` - 训练过程图
- `roc_curve.png` - ROC 曲线图
- `decision_boundary.png` - 决策边界图
   ```

## 🔍 性能评估

### 评估指标

- **准确率 (Accuracy)**：正确预测的比例
- **精确率 (Precision)**：预测为正例中实际为正例的比例
- **召回率 (Recall)**：实际正例中被正确预测的比例
- **F1 分数**：精确率和召回率的调和平均
- **AUC**：ROC 曲线下的面积
