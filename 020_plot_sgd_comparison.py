"""
==================================
Comparing various online solvers
==================================

An example showing how different online solvers perform
on the hand-written digits dataset.

"""
# Author: Rob Zinkov <rob at zinkov dot com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
digits = datasets.load_digits()
X, y = digits.data, digits.target

classifiers = [
    ("SGD", SGDClassifier(max_iter=100, tol=1e-3)),
    ("ASGD", SGDClassifier(average=True, max_iter=1000, tol=1e-3)),

    # Perceptron 感知器，是适用于大规模学习的一种简单算法
    # 默认情况下，不需要设置学习率，不需要正则化处理，仅使用错误样本更新模型（使用合页损失 hinge loss 的感知机比SGD略快，所得模型更稀疏）
    ("Perceptron", Perceptron(tol=1e-3)),

    # 被动攻击算法
    # 是大规模学习的一类算法，不需要设置学习率，比感知机多出一个正则化参数 C
    # 对于分类问题， PassiveAggressiveClassifier 可设定 loss='hinge' （PA-I）或 loss='squared_hinge' （PA-II）
    # 对于回归问题， PassiveAggressiveRegressor 可设置 loss='epsilon_insensitive' （PA-I）或 loss='squared_epsilon_insensitive' （PA-II）
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0],
                               multi_class='auto'))
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
