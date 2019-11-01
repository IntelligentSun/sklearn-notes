"""
=============================
Recursive feature elimination
=============================

A recursive feature elimination example showing the relevance of pixels in
a digit classification task.

.. note::

    See also :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`

"""

# 通过递归式特征消除来体现数字分类任务中像素重要性

print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)

# 递归式的特征消除（Recursive Feature Elimination）
# 通过考虑越来越小的特征集合来递归的选择特征
# 首先，评估器在初始的特征集合上面训练并且每一个特征的重要程度是通过一个 coef_ 属性 或者 feature_importances_ 属性来获得
# 然后，从当前的特征集合中移除最不重要的特征。
# 在特征集合上不断的重复递归这个步骤，直到最终达到所需要的特征数量为止

rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
