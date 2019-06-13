from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import nltk
from import_data import load_pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from import_data import load_pickle
import numpy as np
import pandas as pd
import nltk

# train1_df = load_pickle('data/train1.pkl')
# train2_df = load_pickle('data/train2.pkl')

def extract_features(df, func, pos_tags=False):
    
    # encode labels as integers
    df = df.replace({'candidate00001': 1,'candidate00002': 2,'candidate00003': 3,
    'candidate00004': 4,'candidate00005': 5,'candidate00006': 6,'candidate00007': 7,
    'candidate00008': 8,'candidate00009': 9,'candidate00010': 10,'candidate00011': 11,
    'candidate00012': 12,'candidate00013': 13,'candidate00014': 14,'candidate00015': 15,
    'candidate00016': 16,'candidate00017': 17,'candidate00018': 18,'candidate00019': 19,
    'candidate00020': 20,})

    # convert labels into np array column, one hot encoded
    one_hot_df = pd.get_dummies(df['author'], prefix='author')
    labels = []

    for index, row in one_hot_df.iterrows():
      a1 = row['author_1']
      a2 = row['author_2']
      a3 = row['author_3']
      a4 = row['author_4']
      a5 = row['author_5']
      a6 = row['author_6']
      a7 = row['author_7']
      a8 = row['author_8']
      a9 = row['author_9']
      a10 = row['author_10']
      a11 = row['author_11']
      a12 = row['author_12']
      a13 = row['author_13']
      a14 = row['author_14']
      a15 = row['author_15']
      a16 = row['author_16']
      a17 = row['author_17']
      a18 = row['author_18']
      a19 = row['author_19']
      a20 = row['author_20']
      
      label_row = [a1, a2, a3, a4, a5, a6, a7, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20]
      labels.append(label_row)

    df['author_one_hot'] = labels
    
    if pos_tags:
        # tokenize every text
        df["text"] = df["text"].apply(lambda txt: nltk.word_tokenize(txt)) # or sent tokenize, no word tokenize is better
        # annotate every text as [(word, pos-tag), (word, pos-tag), ...]
        df["text"] = df["text"].apply(lambda txt: nltk.pos_tag(txt))
        # remove word in (word, tag)
        df["text"] = df["text"].apply(lambda txt: str(["".join(tup[1]) for tup in txt]))
    
    # apply feature extraction function to text column
    feats = func.fit_transform(df['text'])
    df['text'] = list(feats.toarray())
    return df
