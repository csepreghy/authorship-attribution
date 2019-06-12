import numpy as np
from sklearn.svm import SVC

def run_svm(df):
    # apparently works better transforming the columns to lists
    features = df["p_truth_E"].tolist()
    labels = df["Truth"].tolist()

    clf = SVC(gamma="auto")
    clf.fit(features, labels)
    
    return SVC.score(clf, features, labels)