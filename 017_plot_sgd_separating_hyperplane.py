"""
=========================================
SGD: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a linear Support Vector Machines classifier
trained using SGD.
"""
print(__doc__)

# 随机梯度下降，拟合线性模型，在样本量和特征数很大时尤为有用

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs

# we create 100 separable points
# 标准差为 0.6
X, Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.70)

# fit the model
# def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
#                  fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
#                  verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None,
#                  random_state=None, learning_rate="optimal", eta0=0.0,
#                  power_t=0.5, early_stopping=False, validation_fraction=0.1,
#                  n_iter_no_change=5, class_weight=None, warm_start=False,
#                  average=False)
# SGCClassifier，正则线性分类模型，支持向量机算法或逻辑回归算法，使用SGD(随机梯度下降）优化，随着梯度下降过程中，学习率会逐渐降低
# 参数loss，算法使用的损失函数，默认是hinge，代表使用线性支持向量机，其他还可以用：log、modified_huber、perceptron、squared_loss等。
# log代表使用逻辑回归算法，modified_huber代表使用概率估计算法，并且对于异常点具有一定的容忍度
# 参数penalty，惩罚项，l1、l2、elasticnet等
# 参数alpha，正则项参数，以及当进行优化时，可以用于计算学习率
# 参数l1_ratio，l1正则项占比
# 参数fit_intercept，是否有截距项
# 参数tol，停止迭代信息，当损失值大于目前最佳损失值减去tol，停止迭代
# 参数shuffle，是否重排
# 参数epsilon，某些特定损失函数下用来设定阈值，当模型预测与真实标签小于这个阈值时，差异会被忽略
# 参数learning_rate，学习率参数，可选如：constant、optimal、invscaling、adaptive等。每一种对应不同的计算公式
# 参数eta0，学习率的初始值设定
# 参数early_stopping，是否早停止，即当每一次梯度下降获得的提升非常小时，是否及时停止优化
# 参数validation_fraction，交叉验证比例
# 参数class_weight，类别权重，如果某些类别赋予很高的权重，则该类别中的样本被错误分类会造成很大的损失
# 属性coef_，每个类别下特征的相关系数
# 属性intercept_，截距
# 属性n_iter_，迭代次数
# 属性loss_function_，损失函数
# 方法dicision_function，为样本预测置信分数
# 方法densify，相关系数矩阵转换为密集数组形式
# 方法fit
# 方法partial_fit，在一个batch上执行梯度下降法
# 方法predict
# 方法score
# 方法sparsify，相关系数矩阵转换为稀疏矩阵形式
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=1000,
                    fit_intercept=True, tol=1e-3)
clf.fit(X, Y)

# plot the line, the points, and the nearest vectors to the plane
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    # 每一个样本决定置信分数，即样本到分隔超平面的距离
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
# 3条等高线，作为分隔超平面
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
# 样本散点图
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)

plt.axis('tight')
plt.show()
