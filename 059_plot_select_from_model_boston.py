"""
===================================================
Feature selection using SelectFromModel and LassoCV
===================================================

Use SelectFromModel meta-transformer along with Lasso to select the best
couple of features from the Boston dataset.
"""
# Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

print(__doc__)

# ----------------------------------------------------
# 从 Boston 数据中自动选择最重要两个特征而不需要提前得知这一信息
# ----------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV(cv=5)

# Set a minimum threshold of 0.25
# SelectFromModel是一个元转换器（meta-transformer）
# 它可以用来处理任何带有 coef_ 或者 feature_importances_ 属性的训练之后的评估器
# 如果相关的coef_ 或者 featureimportances 属性值低于预先设置的阈值，
# 这些特征将会被认为不重要并且移除掉。
# 除了指定数值上的阈值之外，还可以通过给定字符串参数来使用内置的启发式方法找到一个合适的阈值。
# 可以使用的启发式方法有 mean、median 以及使用浮点数相乘（例如，0.1*mean ）

sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1] 
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()
