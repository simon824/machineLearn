from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

def func():
    mnist = fetch_openml("mnist_784", data_home="./data", cache=True)
    # mnist = loadmat("./data/mnist-original.mat")
    X, y = mnist["data"], mnist["target"]
    shuffle_index = np.random.permutation(60000)

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # 数据洗牌

    X_train = X_train.iloc[shuffle_index]
    y_train = y_train[shuffle_index]
    y_train_5 = (y_train == '5')
    y_test_5 = (y_train == '5')
    sgd_clf = SGDClassifier(max_iter=5, random_state=42)

    sgd_clf.fit(X_train, y_train_5)
    sgd_clf.predict([X.iloc[35000]])

    cross_result = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print(cross_result)
    # 交叉验证
    # 数据切分
    skflods = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    for train_index, test_index in skflods.split(X_train, y_train_5):
        # 克隆分类器
        clone_clf = clone(sgd_clf)
        X_train_flods = X_train.iloc[train_index]
        y_train_flods = y_train_5[train_index]
        X_test_flods = X_train.iloc[test_index]
        y_test_flods = y_train_5[test_index]
        clone_clf.fit(X_train_flods, y_train_flods)
        y_pred = clone_clf.predict(X_test_flods)
        n_correct = sum(y_pred == y_test_flods)
        print(n_correct / len(y_pred))
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    con_matr = confusion_matrix(y_train_5, y_train_pred)
    print(con_matr)
    precision = precision_score(y_train_5, y_train_pred)
    recall = recall_score(y_train_5, y_train_pred)
    f1score = f1_score(y_train_5, y_train_pred)
    print(precision, recall, f1score)
    # precisions, recalls, threshold = precision_recall_curve(y_train_5, f1score)
    # print(precisions, recalls, threshold)
    # fpr, tpr, threshold1 = roc_curve(y_train_5, y_scores)
    # fpr, tpr, threshold1 = roc_auc_score(y_train_5, y_scores)


if __name__ == '__main__':
    func()
    # df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
    # print(df[:3])


