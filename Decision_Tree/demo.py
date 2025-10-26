from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征：花萼长/宽，花瓣长/宽
y = iris.target  # 标签：3种鸢尾花

# 2. 创建决策树模型（限制深度便于可视化）
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# 3. 训练模型
clf.fit(X, y)

# 4. 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.title("Decision Tree on Iris Dataset")
plt.show()

# 5. 简单预测
sample = [[5.1, 3.5, 1.4, 0.2]]  # 一个新样本
pred = clf.predict(sample)
print("预测类别:", iris.target_names[pred[0]])