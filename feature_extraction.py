from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import nltk
from script_ideas import load_pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

train1_df = load_pickle('data/train1.pkl')
train2_df = load_pickle('data/train2.pkl')

# outputs lists of numpy arrays

def extract_features(df, func, pos_tags=False):
    
    # encode labels as integers
    df = df.replace({'candidate00001': 1,'candidate00002': 2,'candidate00003': 3,
    'candidate00004': 4,'candidate00005': 5,'candidate00006': 6,'candidate00007': 7,
    'candidate00008': 8,'candidate00009': 9,'candidate00010': 10,'candidate00011': 11,
    'candidate00012': 12,'candidate00013': 13,'candidate00014': 14,'candidate00015': 15,
    'candidate00016': 16,'candidate00017': 17,'candidate00018': 18,'candidate00019': 19,
    'candidate00020': 20,})

    # convert labels into np array column, one hot encoded
    one_hot = pd.get_dummies(df['author'])
    one_hot = pd.Series([row for row in one_hot.values])
    # df = df.drop("author")
    # df = df.join(one_hot)
    df.merge(one_hot.to_frame(), left_index=True, right_index=True)

    # returning features dataframe
    feats = func.fit_transform(df['text'])
    df['text'] = list(feats.toarray())
    
    return df
    
    if pos_tags:
        # tokenize every text
        tokenized_corpus = [nltk.word_tokenize(txt) for txt in corpus]
        
        # annotate every text as [(word, pos-tag), (word, pos-tag), ...]
        pos_corpus = [nltk.pos_tag(txt) for txt in tokenized_corpus]
        
        # remove the words from the tuples
        corpus = []
        for txt in pos_corpus:
            # print(len(txt))
            text = []
            for (word, tag) in txt:
                text.append(tag)
            corpus.append(str(text))
        
        # vectorize tag corpus
        tag_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range = (2,2), binary = False)
        
        tag_grams = tag_vectorizer.fit_transform(df["text"])

        df["text"] = list(tag_grams.toarray())

        # tag_grams = tag_vectorizer.fit_transform(corpus)
        # tag_array = tag_grams.toarray()
        # tag_array = np.hstack((tag_array, labels))
        return df

word_2_grams = extract_features(train1_df, TfidfVectorizer(analyzer = "word", ngram_range = (2,2), binary = False))
print(word_2_grams)#, word_2_grams.shape)
# extract_features(train2_df, chars = True)