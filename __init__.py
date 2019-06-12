import pandas as pd
import time as time

from neural_network_classifier import run_neural_network
from xgboost_classifier import run_xgboost
from import_data import load_pickle
from feature_extraction import extract_features
from svm_classifier import run_svm

# df = pd.read_hdf('data/FAKE_train.h5')

train1_df = load_pickle('data/train1.pkl')
test1_df = load_pickle('data/test1.pkl')
train2_df = load_pickle('data/train2.pkl')
test2_df = load_pickle('data/test2.pkl')

charf_tr1 = extract_features(train1_df, chars=True)
# wordf_tr1 = extract_features(train1_df, words=True)
posf_tr1 = extract_features(train1_df, pos_tags = True)
posf_ts1 = extract_features(test1_df, pos_tags=True)

# problem 1

# for df in [train1_df, test1_df]:
#     for ngram in [(1,1),(2,2),(3,3),(4,4),(5,5)]:
#         for feature in ["char", "word", "pos"]:
#             for max_feats in [50, 100, 200, 500, 1000, 2000]:
#                 feats, labels = extract_features(df, TfidfVectorizer(analyzer = feature, ngram_range = ngram, max_features = max_feats))
#                 for ml_algorithm in [run_neural_network, run_xgboost, run_svm]:
#                     acc = ml_algorithm(feats, labels)

# eventually map tdict = {data:df, }

# problem 2

# for df in [train2_df, test2_df]:

df = posf_tr1.rename(columns={"text": "p_truth_E", "author": "Truth"})

# print(df.columns)

# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)
model = run_xgboost(df)
# model = run_svm(df)
print(model)