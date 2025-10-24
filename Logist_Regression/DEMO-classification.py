import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore') # 忽略潜在的警告信息

# -----------------------------
# 配置参数 (Configuration)
# -----------------------------

# 输入文件路径
CSV_FILE_PATH = "Logist_Regression/lr_dataset.csv"

# 图片保存路径
SAVE_DIR = "Logist_Regression/plots"  # 图片保存目录
os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录

# 模型参数
TEST_SIZE = 0.2          # 测试集比例
RANDOM_STATE = 42        # 随机种子，确保结果可复现
SCALER_ENABLED = True    # 是否启用特征标准化

# 梯度下降参数
LEARNING_RATE = 0.01     # 学习率
EPOCHS = 1000           # 训练轮数
BATCH_SIZE = 32         # 批次大小（如果使用批量梯度下降）
PRINT_INTERVAL = 100    # 每多少轮打印一次训练信息

# 绘图参数
FIGSIZE = (10, 6)        # 图表大小
DPI = 100                # 图表分辨率

# -----------------------------
# 自定义逻辑回归类（梯度下降实现）
# -----------------------------

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.train_accuracies = []
        self.train_aucs = []
        
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """计算交叉熵损失"""
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """使用梯度下降训练模型"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        print(f"\n[INFO] 开始梯度下降训练...")
        print(f"[INFO] 学习率: {self.learning_rate}, 训练轮数: {self.epochs}")
        print(f"[INFO] 数据形状: {X.shape}, 标签分布: {np.bincount(y)}")
        
        for epoch in range(self.epochs):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.train_losses.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 计算训练指标
            y_pred_binary = (y_pred > 0.5).astype(int)
            train_acc = accuracy_score(y, y_pred_binary)
            train_auc = roc_auc_score(y, y_pred)
            
            self.train_accuracies.append(train_acc)
            self.train_aucs.append(train_auc)
            
            # 打印训练信息
            if (epoch + 1) % PRINT_INTERVAL == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {train_acc:.4f} | "
                      f"AUC: {train_auc:.4f}")
        
        print(f"[INFO] 训练完成！最终损失: {self.train_losses[-1]:.4f}")
        print(f"[INFO] 最终训练准确率: {self.train_accuracies[-1]:.4f}")
        print(f"[INFO] 最终训练AUC: {self.train_aucs[-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

# -----------------------------
# 数据加载与预处理
# -----------------------------

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # 删除空行
        df = df.dropna()
        
        if df.shape[1] != 3:
            raise ValueError(f"CSV 文件必须有 3 列，当前有 {df.shape[1]} 列。")
        
        # 获取列名
        x_col, y_col, label_col = df.columns[0], df.columns[1], df.columns[2]
        X = df[[x_col, y_col]].values
        y = df[label_col].values
        
        # 更安全的标签处理
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"数据必须是二分类，发现 {len(unique_labels)} 个类别: {unique_labels}")
        
        # 如果标签不是0/1，转换为0/1
        if not set(unique_labels).issubset({0, 1}):
            print(f"[INFO] 将标签从 {unique_labels} 转换为 0/1")
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = np.array([label_map[label] for label in y])
        
        return X, y
    except Exception as e:
        raise Exception(f"数据加载失败: {str(e)}")

def visualize_data(X, y, title="Data Visualization", save_name="data_distribution"):
    """
    使用 matplotlib/seaborn 可视化数据。
    """
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Class Label')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(SAVE_DIR, f"{save_name}.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"[INFO] 图片已保存: {save_path}")
    plt.show()

# -----------------------------
# 模型训练与评估
# -----------------------------

def train_and_evaluate(X, y):
    """
    划分数据集、训练模型、评估性能。
    """
    # 1. 划分训练集和测试集
    print("\n[INFO] 正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"[INFO] 训练集形状: X_train {np.array(X_train).shape}, y_train {np.array(y_train).shape}")
    print(f"[INFO] 测试集形状: X_test {np.array(X_test).shape}, y_test {np.array(y_test).shape}")
    print(f"[INFO] 训练集标签分布: {np.bincount(y_train)}")
    print(f"[INFO] 测试集标签分布: {np.bincount(y_test)}")

    # 2. 特征标准化 (可选，但推荐)
    if SCALER_ENABLED:
        print("\n[INFO] 正在对特征进行标准化...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None

    # 3. 使用自定义逻辑回归模型训练
    print("\n[INFO] 正在训练自定义逻辑回归模型（梯度下降）...")
    model = CustomLogisticRegression(
        learning_rate=LEARNING_RATE, 
        epochs=EPOCHS, 
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    # 4. 模型预测
    print("\n[INFO] 正在进行预测...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # 5. 计算评估指标
    print("\n[INFO] 计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- 模型性能报告 ---")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数 (F1):       {f1:.4f}")
    print(f"AUC (Area Under Curve): {auc:.4f}")

    # 6. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:\n{cm}")

    # 7. ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存ROC曲线
    save_path = os.path.join(SAVE_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"[INFO] ROC曲线已保存: {save_path}")
    plt.show()

    # 8. 绘制训练过程
    plot_training_process(model, save_name="training_process")

    return model, scaler, y_test, y_pred, y_pred_proba

def plot_training_process(model, save_name="training_process"):
    """绘制训练过程"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)
    
    # 损失曲线
    ax1.plot(model.train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(model.train_accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # AUC曲线
    ax3.plot(model.train_aucs)
    ax3.set_title('Training AUC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存训练过程图
    save_path = os.path.join(SAVE_DIR, f"{save_name}.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"[INFO] 训练过程图已保存: {save_path}")
    plt.show()

def visualize_results(model, scaler, X, y):
    """
    可视化决策边界和预测结果。
    """
    print("\n[INFO] 正在绘制决策边界...")
    
    # 为了可视化决策边界，需要创建一个网格
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 将网格点展平为模型输入格式
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 如果使用了标准化，也需要对网格点进行标准化
    if scaler is not None:
        grid_points_scaled = scaler.transform(grid_points)
    else:
        grid_points_scaled = grid_points

    # 预测网格点的类别
    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # 绘制原始数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Logistic Regression Decision Boundary (Gradient Descent)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存决策边界图
    save_path = os.path.join(SAVE_DIR, "decision_boundary.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"[INFO] 决策边界图已保存: {save_path}")
    plt.show()

# -----------------------------
# 主函数
# -----------------------------

def main():
    """
    主函数：执行整个分类流程
    """
    print("=" * 60)
    print("二分类任务 - 逻辑回归模型（梯度下降）")
    print("=" * 60)

    try:
        # 1. 加载数据
        X, y = load_data(CSV_FILE_PATH)

        # 2. 可视化原始数据
        visualize_data(X, y, title="Original Data Distribution")

        # 3. 训练模型并评估
        model, scaler, y_test, y_pred, y_pred_proba = train_and_evaluate(X, y)

        # 4. 可视化结果（决策边界）
        visualize_results(model, scaler, X, y)

        print(f"\n[SUCCESS] 任务完成！所有图片已保存到: {SAVE_DIR}")

    except Exception as e:
        print(f"\n[ERROR] 执行过程中发生错误: {str(e)}")
        raise

# -----------------------------
# 程序入口
# -----------------------------
if __name__ == "__main__":
    main()