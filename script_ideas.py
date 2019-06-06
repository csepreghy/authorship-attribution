# all taken straight from Kostas Persifanos video on authorship attribution
# https://www.youtube.com/watch?v=dBqyvpfHy8k

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import pandas as pd
import nltk

def load_corpus(input_dir):
    trainfiles = [ f for f in listdir(input_dir) if isfile(join(input_dir ,f))]
    trainset = []
    for filename in trainfiles:
        df = pd.read_csv( input_dir + "/" + filename , sep="\t",
            dtype={ "id":object, "text":object } )
        for row in df["text"]:
            trainset.append({"label":filename, "text": row})
    
    return trainset

# 10:50 = feature extraction
# 12:50 = putting features together into word features
# 17:30 evaluation

