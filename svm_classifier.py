import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from import_data import load_pickle
from feature_extraction import extract_features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

train1_df = load_pickle('data/train1.pkl')

count_tr1_char_2 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(2,2), binary = False))

def run_svm(df):
    # apparently works better transforming the columns to lists
    X = df["text"].array
    print("X:", type(X))

    enc = LabelEncoder()
    y = df["author"]#.tolist()
    y = enc.fit_transform(y)
    print(enc.classes_, "y", y)
    print("y", type(y))
    #y = y.array

    #y = MultiLabelBinarizer().fit_transform(y)
    clf = SVC(gamma="auto", kernel="linear")
    # score = cross_val_score(clf, X, y, cv = 5)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    # skf.get_n_splits(X, y)
    
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", type(train_index), train_index, "TEST:", type(test_index), test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(np.array(X_train), np.array(y_train))
        #print("train acc:", SVC.score(clf, X_train, y_train))
        #print("test acc:", SVC.score(clf, X_test, y_test))

    clf = SVC(gamma="auto", kernel="linear")
    #clf.fit(X_train, y_train)
    
    #return score

print(run_svm(count_tr1_char_2))