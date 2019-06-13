import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from import_data import load_pickle
from feature_extraction import extract_features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

train1_df = load_pickle('data/train1.pkl')

count_tr1_char_2 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(3,3), binary = False))

def run_svm(df):
    # casting X as list
    X = df["text"].tolist()

    # labelencoding and casting y as list
    enc = LabelEncoder()
    y = df["author"].tolist()
    y = enc.fit_transform(y).tolist()

    clf = SVC(gamma="auto", kernel="linear")
    score = cross_val_score(clf, X, y, cv = 5)
    
    return score, score.mean(), score.std()

print(run_svm(count_tr1_char_2))