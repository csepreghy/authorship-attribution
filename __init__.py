import pandas as pd
import time as time

from neural_network_classifier import run_neural_network
from xgboost_classifier import run_xgboost
from script_ideas import load_pickle
from feature_extraction import extract_features
from svm_classifier import run_svm

# df = pd.read_hdf('data/FAKE_train.h5')

train1_df = load_pickle('data/train1.pkl')
test1_df = load_pickle('data/test1.pkl')
# train2_df = load_pickle('data/train2.pkl')
# test2_df = load_pickle('data/test2.pkl')

charf_tr1 = extract_features(train1_df, chars=True)
# wordf_tr1 = extract_features(train1_df, words=True)
posf_tr1 = extract_features(train1_df, pos_tags = True)
posf_ts1 = extract_features(test1_df, pos_tags=True)

# print(posf_ts1)

df = posf_tr1.rename(columns={"text": "p_truth_E", "author": "Truth"})

# print(df.columns)

# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)
# model = run_xgboost(df)
model = run_svm(df)
print(model)