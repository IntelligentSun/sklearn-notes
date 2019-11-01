"""
=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.
"""
print(__doc__)

# 在Iris数据集比对 LDA 和 PCA 之间的降维差异

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# 通过把输入数据投影到由最大化类之间分离的方向所组成的线性子空间，可以执行有监督降维

# def __init__(self, solver='svd', shrinkage=None, priors=None,
#                  n_components=None, store_covariance=False, tol=1e-4)
# 参数 n_components 维度数量
# 参数 shrinkage 收缩参数，是一种在训练样本相比特征而言很小的情况下可以提升的协方差矩阵预测的工具
# 收缩可以设置 shrinkage=‘auto’ 来实现，也可以手动设置为0-1,
# 0值对应没有收缩，1值对应完全使用收缩，意味着方差的对角矩阵将被当作协方差矩阵的估计
# 默认的参数 solver 是 ‘svd’，它可以进行classification (分类) 以及 transform (转换),
# 而且它不会依赖于协方差矩阵的计算（结果），这在特征数量特别大的时候十分具有优势。
# 然而，’svd’ solver 无法与 shrinkage （收缩）同时使用
# ‘lsqr’ solver支持shrinkage
# ‘eigen’ solver是基于类散度和类内离散率之间的优化，可以被用在分类和转换，同事支持收缩
# ‘eigen’ solver解决方案需要计算协方差矩阵，因此不适用于具有大量特征的情况

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
