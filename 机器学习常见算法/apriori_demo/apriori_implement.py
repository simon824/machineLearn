def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, CK, minsupport):
    sscnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if can not in sscnt:
                    sscnt[can] = 1
                else:
                    sscnt[can] += 1

    numItems = float(len(list(D)))
    retlist = []
    supportdata = {}
    for key in sscnt:
        support = sscnt[key] / numItems
        if support >= minsupport:
            retlist.insert(0, key)
        supportdata[key] = support
    return retlist, supportdata


def aprioriGen(LK, k):
    retlist = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            if L1 == L2:
                retlist.append(LK[i] | LK[j])
    return retlist


def apriori(dataset, minsupport=0.5):
    c1 = createC1(dataset)
    L1, supportdata = scanD(dataset, c1, minsupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        CK = aprioriGen(L[k - 2], k)
        LK, supk = scanD(dataset, CK, minsupport)
        supportdata.update(supk)
        L.append(LK)
        k += 1
    return L, supportdata


def generateRules(L, supportdata, minconfidence=0.6):
    rulelist = []
    for i in range(1, len(L)):
        for freset in L[i]:
            H1 = [frozenset([item]) for item in freset]
            rulessFromConseq(freset, H1, supportdata, rulelist, minconfidence)


def rulessFromConseq(freset, H1, supportdata, rulelist, minconfidence=0.6):
    m = len(H1[0])
    while (len(freset) > m):
        H = calConf(freset, H1, supportdata, rulelist, minconfidence)
        if len(H) > 1:
            aprioriGen(H, m + 1)
            m += 1
        else:
            break


def calConf(freset, H1, supportdata, rulelist, minconfidence):
    prunedh = []
    for conseq in H1:
        conf = supportdata[freset] / supportdata[freset - conseq]
        if conf >= minconfidence:
            print(freset - conseq, '-->', conseq, "conf:", conf)
            rulelist.append((freset - conseq, conseq, conf))
            prunedh.append(conseq)
    return prunedh

if __name__ == '__main__':
    dataset = loadDataSet()
    L, supportdata = apriori(dataset)
    i = 0
    for freq in L:
        print("项数: ", i + 1, "==", freq)
        i += 1

    rules = generateRules(L, supportdata, minconfidence=0.5)
