from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import nltk
from script_ideas import load_pickle
import numpy as np

train1_df = load_pickle('data/train1.pkl')
train2_df = load_pickle('data/train2.pkl')

def extract_features(df, words = False, chars = False, pos_tags = False, ngram=(2,2)):
    
    # encode labels as integers
    df = df.replace({'candidate00001': 1,'candidate00002': 2,'candidate00003': 3,
    'candidate00004': 4,'candidate00005': 5,'candidate00006': 6,'candidate00007': 7,
    'candidate00008': 8,'candidate00009': 9,'candidate00010': 10,'candidate00011': 11,
    'candidate00012': 12,'candidate00013': 13,'candidate00014': 14,'candidate00015': 15,
    'candidate00016': 16,'candidate00017': 17,'candidate00018': 18,'candidate00019': 19,
    'candidate00020': 20,})

    # convert labels into np array column
    # i absolutely presume that the arrays of vectorizers her are outputted in same order as they are inputted!
    labels = df["author"].to_numpy()
    # labels = labels.reshape(labels.shape[0],1)
    
    # corpus to train on as list of strings
    corpus = df['text'].tolist()

    if words:
        word_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range = ngram, binary = False)
        
        # returning features as dataframe
        vect = word_vectorizer.fit_transform(df['text'])
        
        # df['text']=list(vect)
        df['text'] = list(vect.toarray())
        print = False
        if print:
            for row in df["text"]:
                print(row.shape)
        
        # word_grams = word_vectorizer.fit_transform(corpus)
        # word_array = word_grams.toarray()
        # word_array = np.hstack((word_array, labels))
        # return word_array
        return df
    
    if chars:
        char_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = ngram, binary = False, min_df = 0)
        # char_grams = char_vectorizer.fit_transform(corpus)
        char_grams = char_vectorizer.fit_transform(df["text"])

        df["text"] = list(char_grams.toarray())

        return df
        
        
        # char_array = char_grams.toarray()
        # char_array = np.hstack((char_array, labels))
        # return char_array
    
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
        tag_grams = tag_vectorizer.fit_transform(corpus)
        tag_array = tag_grams.toarray()
        tag_array = np.hstack((tag_array, labels))
        return tag_array

word_2_grams = extract_features(train1_df, words=True, chars=False, pos_tags = False)
print(word_2_grams)#, word_2_grams.shape)
# extract_features(train2_df, chars = True)