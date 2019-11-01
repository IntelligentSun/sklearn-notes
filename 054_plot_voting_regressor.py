"""
=================================================
Plot individual and voting regression predictions
=================================================

Plot individual and averaged regression predictions for Boston dataset.

First, three exemplary regressors are initialized (`GradientBoostingRegressor`,
`RandomForestRegressor`, and `LinearRegression`) and used to initialize a
`VotingRegressor`.

The red starred dots are the averaged predictions.

"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

# Loading some example data
boston = datasets.load_boston()
X = boston.data
y = boston.target

# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
reg3 = LinearRegression()

# 投票回归器
# 其背后思想是将概念上不同的机器学习回归器组合起来，并返回平均预测值
# 这样一个回归树对于一组同样表现良好的模型是有用的，以便平衡它们各自的弱点

# 相比而言，对于投票分类器 VotingClassifier
# 其原理是结合了多个不同的机器学习分类器,并且采用多数表决（majority vote）（硬投票） 或者平均预测概率（软投票）的方式来预测分类标签。
# 这样的分类器可以用于一组同样表现良好的模型,以便平衡它们各自的弱点。
# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)
ereg.fit(X, y)

xt = X[:20]

plt.figure()
plt.plot(reg1.predict(xt), 'gd', label='GradientBoostingRegressor')
plt.plot(reg2.predict(xt), 'b^', label='RandomForestRegressor')
plt.plot(reg3.predict(xt), 'ys', label='LinearRegression')
plt.plot(ereg.predict(xt), 'r*', label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()
