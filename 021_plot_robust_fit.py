"""
Robust linear estimator fitting
===============================

Here a sine function is fit with a polynomial of order 3, for values
close to zero.

Robust fitting is demoed in different situations:

- No measurement errors, only modelling errors (fitting a sine with a
  polynomial)

- Measurement errors in X

- Measurement errors in y

The median absolute deviation to non corrupt new data is used to judge
the quality of the prediction.

What we can see that:

- RANSAC is good for strong outliers in the y direction

- TheilSen is good for small outliers, both in direction X and y, but has
  a break point above which it performs worse than OLS.

- The scores of HuberRegressor may not be compared directly to both TheilSen
  and RANSAC because it does not attempt to completely filter the outliers
  but lessen their effect.

"""

# 稳健线性估计拟合

from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
# Make sure that it X is 2D
X = X[:, np.newaxis]

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10

######
# Scikit-learn提供了三种稳健回归的预测器（estimator）: RANSAC ， Theil Sen 和 HuberRegressor
#
# HuberRegressor 一般快于 RANSAC 和 Theil Sen ，除非样本数很大，即 n_samples >> n_features
# 这是因为 RANSAC 和 Theil Sen 都是基于数据的较小子集进行拟合。
# 但使用默认参数时， Theil Sen 和 RANSAC 可能不如 HuberRegressor 鲁棒。
#
# RANSAC 比 Theil Sen 更快，在样本数量上的伸缩性（适应性）更好。
# RANSAC 能更好地处理y方向的大值离群点（通常情况下）。
# Theil Sen 能更好地处理x方向中等大小的离群点，但在高维情况下无法保证这一特点。 实在决定不了的话，请使用 RANSAC
######

######
# RANSAC：随机抽样一致性算法（Random Sample Consensus）利用全体数据中局内点的一个随机子集拟合模型
# RANSAC 是一种非确定性算法，以一定概率输出一个可能合理结果，依赖于迭代次数
# 主要解决线性或非线性回归问题

# 每轮迭代执行以下步骤:
# 1.从原始数据中抽样 min_samples 数量的随机样本，检查数据是否合法（见 is_data_valid ）。
# 2.用一个随机子集拟合模型（ base_estimator.fit ）。检查模型是否合法（见 is_model_valid ）。
# 3.计算预测模型的残差（residual），将全体数据分成局内点和离群点（ base_estimator.predict(X) - y ）。
# 绝对残差小于 residual_threshold 的全体数据认为是局内点。
# 4.若局内点样本数最大，保存当前模型为最佳模型。以免当前模型离群点数量恰好相等（而出现未定义情况），规定仅当数值大于当前最值时认为是最佳模型。
######

######
# Theil-Sen预估器：广义中值预估器（generalized-median-based estimator）
# 使用中位数在多个维度泛化，对多元异常值更具有鲁棒性，但随着维数的增加，估计器的准确性在迅速下降
# 准确性的丢失，导致在高维上的估计比不上普通的最小二乘法

# 与 OLS（ordinary least squares）不同的是， Theil-Sen 是一种非参数方法，这意味着它没有对底层数据的分布假设。
# 由于 Theil-Sen 是基于中值的估计，它更适合于损坏的数据即离群值。
# 在单变量的设置中，Theil-Sen 在简单的线性回归的情况下，其崩溃点大约 29.3% ，这意味着它可以容忍任意损坏的数据高达 29.3%
######

######
# HuberRegressor 与 Ridge 不同，因为它对于被分为异常值的样本应用了一个线性损失。
# 如果这个样品的绝对误差小于某一阈值，样品就被分为内围值。
# 它不同于 TheilSenRegressor 和 RANSACRegressor ，因为它没有忽略异常值的影响，并分配给它们较小的权重

######

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(X_test), y_test)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()
