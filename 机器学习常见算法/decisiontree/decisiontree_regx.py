import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier


X, y = make_moons(n_samples=100, noise=0.25, random_state=53)
tree_clf1 = DecisionTreeClassifier(random_state=42)  # 设置随机种子42
tree_clf1 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
plt.figure(figsize=(12, 4))





