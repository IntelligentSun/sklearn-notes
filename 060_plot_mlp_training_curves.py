"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.
"""

# 多层感知机（MLP）是一种监督学习算法
# 可以学习用于分类或回归的非线性函数
# 与逻辑回归不同，在输入层与输出层之间，可以有一个或多个非线性层，称为隐藏层

# 多层感知器的优点:
# 1. 可以学习得到非线性模型。
# 2. 使用partial_fit 可以学习得到实时模型(在线学习)。

# 多层感知器(MLP)的缺点:
# 1. 具有隐藏层的 MLP 具有非凸的损失函数，它有不止一个的局部最小值。因此不同的随机权重初始化会导致不同的验证集准确率。
# 2. MLP 需要调试一些超参数，例如隐藏层神经元的数量、层数和迭代轮数。
# 3. MLP 对特征归一化很敏感.

print(__doc__)

import warnings

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)

        # 实现了通过 Backpropagation 进行训练的多层感知器算法
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)

        # MLPClassifier 只支持交叉熵损失函数，通过运行 predict_proba 方法进行概率估计
        # MLP 算法使用的是反向传播方式，通过反向传播计算得到的梯度和某种形式的梯度下降来进行训练
        # MLP 使用 Stochastic Gradient Descent（随机梯度下降）(SGD), Adam, 或者 L-BFGS 进行训练
        #
        # 对于分类来说，它最小化交叉熵损失函数，为每个样本 x 给出一个向量形式的概率估计 P(y|x)
        # mlp.predict_proba([2.,2.])
        #
        # MLPClassifier 通过应用 Softmax 作为输出函数来支持多分类。
        # 此外，该模型支持 多标签分类 ，一个样本可能属于多个类别。
        # 对于每个类，原始输出经过 logistic 函数变换后，大于或等于 0.5 的值将进为 1，否则为 0

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# load / generate some toy datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
data_sets = [(iris.data, iris.target),
             (digits.data, digits.target),
             datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
             datasets.make_moons(noise=0.3, random_state=0)]

for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
                                                    'circles', 'moons']):
    plot_on_dataset(*data, ax=ax, name=name)

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()
