import numpy as np
from sklearn.svm import SVC

def run_svm(df):
    # apparently works better transforming the columns to lists
    features = df["text"].tolist()
    # print(len(features))
    labels = df["author"].tolist()
    # print(len(labels))

    clf = SVC(gamma="auto", kernel="linear")
    clf.fit(features, labels)
    
    return SVC.score(clf, features, labels)