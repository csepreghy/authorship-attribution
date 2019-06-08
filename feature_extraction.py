from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import nltk
from script_ideas import load_pickle

train1_df = load_pickle('data/train1.pkl')

def extract_features(df, words = True, chars = False, pos_tags = False, ngram=(2,2)):
    # encode labels as integers
    df = df.replace({'candidate00001': 1,'candidate00002': 2,'candidate00003': 3,
    'candidate00004': 4,'candidate00005': 5,'candidate00006': 6,'candidate00007': 7,
    'candidate00008': 8,'candidate00009': 9,'candidate00010': 10,'candidate00011': 11,
    'candidate00012': 12,'candidate00013': 13,'candidate00014': 14,'candidate00015': 15,
    'candidate00016': 16,'candidate00017': 17,'candidate00018': 18,'candidate00019': 19,
    'candidate00020': 20,})

    # print(df.head())

    corpus = df['text'].tolist()

    # print(type(corpus), len(corpus))

    if words:
        word_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range = ngram, binary = False)
        word_grams = word_vectorizer.fit_transform(corpus)
        print(type(word_grams), word_grams.shape)
    
    if chars:
        char_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = ngram, max_features = 2000, binary = False, min_df = 0)
        char_grams = char_vectorizer.fit_transform(corpus)
    
    # print(type(train1_df))

    # print(x.toarray())
    # corpus = []
    # classes = []
    # for item in df.itertuples():
        #print(item[2])
        # corpus.append(item[1])
        # classes.append(item[2])

    # vectorizer = FeatureUnion([("chars", char_vector), ("words", word_vector)])

    # X1 = vectorizer.fit_transform(corpus)
    #X2 = tag_vector.fit_transform(tags)

    #X = sp.hstack((X1, X2), format = "csr")

    # print(type(X1), X1)

    # return feature_matrix

extract_features(train1_df)