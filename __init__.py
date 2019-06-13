import pandas as pd
import time as time

# from neural_network_classifier import run_neural_network
from xgboost_classifier import run_xgboost
from import_data import load_pickle
from feature_extraction import extract_features
from svm_classifier import run_svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# df = pd.read_hdf('data/FAKE_train.h5')

train1_df = load_pickle('data/train1.pkl')
test1_df = load_pickle('data/test1.pkl')
train2_df = load_pickle('data/train2.pkl')
test2_df = load_pickle('data/test2.pkl')

# charf_tr1 = extract_features(train1_df, chars=True)
# wordf_tr1 = extract_features(train1_df, words=True)
# posf_tr1 = extract_features(train1_df, pos_tags = True)
# posf_ts1 = extract_features(test1_df, pos_tags=True)

# print("svm w/ countvectorizer on n-gram chars")
count_tr1_char_1 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(1,1), binary = False))
print(count_tr1_char_1.head())

"""
print(run_svm(count_tr1_char_1))
count_tr1_char_2 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(2,2), binary = False))
print(run_svm(count_tr1_char_1))
count_tr1_char_3 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(3,3), binary = False))
# print(run_svm(count_tr1_char_1))
count_tr1_char_4 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(4,4), binary = False))
# print(run_svm(count_tr1_char_1))
count_tr1_char_5 = extract_features(train1_df, CountVectorizer(analyzer = "char", ngram_range=(5,5), binary = False))
# print(run_svm(count_tr1_char_1))


# tr1_char_1 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(1,1), binary = False))
# print(run_svm(tr1_char_1))
# tr1_char_2 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(2,2), binary = False))
# print(run_svm(tr1_char_2))
# tr1_char_3 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(3,3), binary = False))
# print(run_svm(tr1_char_3))
# tr1_char_4 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(4,4), binary = False))
# print(run_svm(tr1_char_4))
# tr1_char_5 = extract_features(train1_df, TfidfVectorizer(analyzer = "char", ngram_range=(5,5), binary = False))
# print(run_svm(tr1_char_5))

# tr1_word_1 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(1,1), binary = False))
# print(run_svm(tr1_word_1))
# tr1_word_2 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(2,2), binary = False))
# print(run_svm(tr1_word_2))
# tr1_word_3 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(3,3), binary = False))
# print(run_svm(tr1_word_3))
# tr1_word_4 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(4,4), binary = False))
# print(run_svm(tr1_word_4))
# tr1_word_5 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(5,5), binary = False))
# print(run_svm(tr1_word_5))

# tr1_pos_1 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(1,1), binary = False), pos_tags = True)
# print(run_svm(tr1_pos_1))
# tr1_pos_2 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(2,2), binary = False), pos_tags = True)
# print(run_svm(tr1_pos_2))
# tr1_pos_3 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(3,3), binary = False), pos_tags = True)
# print(run_svm(tr1_pos_3))
# tr1_pos_4 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(4,4), binary = False), pos_tags = True)
# print(run_svm(tr1_pos_4))
# tr1_pos_5 = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range=(5,5), binary = False), pos_tags = True)
# print(run_svm(tr1_pos_5))

# df = posf_tr1.rename(columns={"text": "p_truth_E", "author": "Truth"})

# print(df.columns)

# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)
# model = run_xgboost(df)

# for model in [tr1_pos_1, tr1_pos_2, tr1_pos_3, tr1_pos_4, tr1_pos_5]:
#     acc = run_svm(model)
#     print(acc)
"""