from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)




'''
投票策略： 软投票与硬投票
硬投票：直接用类别值，少数服从多数
软投票：各自分类器的概率值进行加权平均
'''

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# 硬投票
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

# 软投票
log_clf_soft = LogisticRegression(random_state=42)
rnd_clf_soft = RandomForestClassifier(random_state=42)
svm_clf_soft = SVC(probability=True, random_state=42)  # probability需要设置为True，才能实现软投票
voting_clf_soft = VotingClassifier(estimators=[('lr', log_clf_soft), ('rf', rnd_clf_soft), ('svc', svm_clf_soft)], voting='soft')
voting_clf_soft.fit(X_train, y_train)

for clf_soft in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf_soft.fit(X_train, y_train)
    y_pred = clf_soft.predict(X_test)
    print(clf_soft.__class__.__name__, accuracy_score(y_test, y_pred))

for clf in (log_clf_soft, rnd_clf_soft, svm_clf_soft, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


iris = load_iris()

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1)
rf_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rf_clf.feature_importances_):
    print(name, score)

mnist = fetch_openml('mnist_784')
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1)
rf_clf.fit(mnist['data'], mnist['target'])
print(rf_clf.feature_importances_.shape)


# alpha透明度
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo', alpha=0.6)
plt.plot(X[:, 0][y==0], X[:, 1][y==1], 'bs', alpha=0.6)

# plt.show()


