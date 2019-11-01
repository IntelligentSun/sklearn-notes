"""
===============================
Univariate Feature Selection
===============================

An example showing univariate feature selection.

Noisy (non informative) features are added to the iris data and
univariate feature selection is applied. For each feature, we plot the
p-values for the univariate feature selection and the corresponding
weights of an SVM. We can see that univariate feature selection
selects the informative features and that these have larger SVM weights.

In the total set of features, only the 4 first ones are significant. We
can see that they have the highest score with univariate feature
selection. The SVM assigns a large weight to one of these features, but also
Selects many of the non-informative features.
Applying univariate feature selection before the SVM
increases the SVM weight attributed to the significant features, and will
thus improve classification.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

# #############################################################################
# Import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target

plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features

# 单变量的特征选择
# 是通过基于单变量的统计测试来选择最好的特征
# 可以当做是评估器的预处理步骤

# SelectKBest: 移除那些除了评分最高的K个特征之外的所有特征

# SelectPercentile：移除了用户指定的最高得分百分比之外的所有特征

# 对每个特征应用常见的单变量统计测试:
# 假阳性率（false positive rate） SelectFpr,
# 伪发现率（false discovery rate） SelectFdr ,
# 或者族系误差（family wise error） SelectFwe

# GenericUnivariateSelect：允许使用可配置方法来进行单变量特征选择。
# 它允许超参数搜索评估器来选择最好的单变量特征

# score function
# 对于回归： f_regression , mutual_info_regression
# 对于分类: chi2 , f_classif , mutual_info_classif
# 如果你使用的是稀疏的数据 (例如数据可以由稀疏矩阵来表示),chi2,mutual_info_regression,mutual_info_classif可以处理数据并保持它的稀疏性

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')

# #############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
        color='navy', edgecolor='black')

clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='c',
        edgecolor='black')


plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()
