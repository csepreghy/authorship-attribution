from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extraction import extract_features
from import_data import load_pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

train1_df = load_pickle('data/train1.pkl')

df = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(3,3), binary = False))

print(df.columns)

# .drop(columns={"Truth", "p_truth_E"})

def run_xgboost(df):
  y_train = df['author_one_hot']
  .drop(['text'],axis=1,inplace=True)
  clf = XGBClassifier()
  clf.fit(X_train.values, y_train.values)
  
  # X, y = df.loc[:,"text"].to_numpy().reshape(1,-1) ,df.loc[:,"author_one_hot"].to_numpy().reshape(1,-1)
  # print(X.shape, y.shape)
  #X = df["text"]

  #mlb = MultiLabelBinarizer()
  #y = df["author"].to_numpy()
  #print(y, type(y))
  #y = mlb.fit_transform(y)
  #print(y, type(y))

  #enc = LabelEncoder()
  #y = df["author_one_hot"]
  dmatrix = DMatrix(data=X,label=y)
  #y = enc.fit_transform(y).tolist()
  #print(y, type(y))

  #clf = XGBClassifier()
  #score = cross_val_score(clf, X, y, cv = 5)
  # model.fit(x_train, y_train)

  # y_pred_train = model.predict(x_train)
  # predictions_train = [round(value) for value in y_pred_train]
  # accuracy_train = accuracy_score(y_train, predictions_train)
  # print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
  #return score
run_xgboost(df)
