from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def run_xgboost(df):
  X_train = df['text']
  y_train = df["author_one_"]

  model = XGBClassifier()
  model.fit(X_train, y_train)

  y_pred_train = model.predict(X_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))


df = pd.read_pickle('word_2_grams.pkl')
run_xgboost(df)
