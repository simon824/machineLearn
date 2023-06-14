from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pickle
from math import log
import operator
import numpy as np
import os
import warnings
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
warnings.filterwarnings('ignore')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def createDataSet():
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 0, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ["F1-AGE", 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataset, labels

# 创建树模型
def createTree(dataset, labels, featLabels):
    classList = [example[-1] for example in dataset]
    # 判断标签是否一样
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    # 得到最好的特征的索引
    bestFeat = chooseBestFeatureToSplit(dataset)
    # 得到最好特征的标签
    bestFeatLabel = labels[bestFeat]
    # 定义特征的先后顺序
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    # 删除对应的特征
    del labels[bestFeat]
    featValue = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValue)
    # 构建树模型
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), sublabels, featLabels)
    return myTree

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]

def chooseBestFeatureToSplit(dataset):
    numFeature = len(dataset[0]) - 1
    # 计算熵值
    baseEntropy = calcEntropy(dataset)
    # 分裂
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        new_entropy = 0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataset, i, val)
            prob = len(subDataSet) / float(len(dataset))
            new_entropy += prob *  calcEntropy(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - new_entropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 数据集切分
def splitDataSet(dataset, axis, val):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcEntropy(dataset):
    # 样本的数量
    numexamples = len(dataset)
    labelCounts = {}
    for feaVec in dataset:
        currentlabel = feaVec[-1]
        if currentlabel not in labelCounts:
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    entropy = 0
    # 计算熵值
    for key in labelCounts:
        prop = float(labelCounts[key]) / numexamples
        entropy -= prop * log(prop, 2)
    return entropy


def demo():
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)
    export_graphviz(
        tree_clf,
        out_file='iris_tree.dot',
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
# 执行命令将dot文件转为png  dot -Tpng iris_tree.dot -o iris_tree.png

if __name__ == '__main__':
    dataset, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataset, labels, featLabels)
    demo()

