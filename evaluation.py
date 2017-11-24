def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def precision(TP, FP):
    return (TP) / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def f_mesure(p, r):
    return (2 * p * r) / (p + r)


def evaluateResults(TP, TN, FP, FN):
    print("=== Confusion Matrix ===")
    print("TP=", TP, "FN=", FN)
    print("FP=", FP, "TN=", TN)
    print("=======================")
    a = accuracy(TP, TN, FP, FN)
    r = recall(TP, FN)
    p = precision(TP, FP)
    print("Accuracy = ", "{0:.3f}".format(a))
    print("Recall  =", "{0:.3f}".format(r))
    print("Precision = ", "{0:.3f}".format(p))
    return float("{0:.3f}".format(a)), float("{0:.3f}".format(r)), float(
        "{0:.3f}".format(p))  # returned if they needed in any stage


if __name__ == '__main__':
    accuracy, recall, precision = evaluateResults(5, 10, 3, 2)
