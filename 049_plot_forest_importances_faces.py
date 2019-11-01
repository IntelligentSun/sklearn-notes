"""
=================================================
Pixel importances with a parallel forest of trees
=================================================

This example shows the use of forests of trees to evaluate the importance
of the pixels in an image classification task (faces). The hotter the pixel,
the more important.

The code below also illustrates how the construction and the computation
of the predictions can be parallelized within multiple jobs.
"""

print(__doc__)

# 特征重要性评估
# 特征对目标变量预测的相对重要性可以通过（树中的决策节点的）特征使用的相对顺序（即深度）来进行评估
# scikit-learn通过将特征贡献的样本比例与纯度减少相结合得到特征的重要性
# 通过对多个随机树中的 预期贡献率 （expected activity rates） 取平均，可以减少这种估计的 方差 ，并将其用于特征选择

from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Number of cores to use to perform parallel fitting of the forest model
# 模型并行化，若n_jobs = k，则计算被划分为k个作业，并运行在机器的k个核上
# 若n_jobs = -1，则使用机器的所有核
n_jobs = 1

# Load the faces dataset
data = fetch_olivetti_faces()
X = data.images.reshape((len(data.images), -1))
y = data.target

mask = y < 5  # Limit to 5 classes
X = X[mask]
y = y[mask]

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(X, y)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

# 基于树的特征选取选择（特征选取）
model = SelectFromModel(forest, prefit=True)
X_new = model.transform(X)
X_new.shape

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.show()
