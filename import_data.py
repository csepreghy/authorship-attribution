from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import pandas as pd
import numpy as np
import nltk
import os
from os import listdir
from os.path import isfile, join, isdir
import json
import pickle

def load_train(input_dir):
    # lists of dictionaries for dataframes
    dicts = []
    
    # for every candidate in problem1 or 2
    for direc in listdir(input_dir):
        if "candidate" in (direc):
            # print(direc)
            
            # for every file in the candidate directory, read and add every txt to lst
            for fin in listdir(input_dir + "/" + direc):
                #print(fin)
                f_txt = open(input_dir + "/" + direc + "/" + fin, "r+", encoding="utf-8")
                f = f_txt.read()
                # add that file to lst
                #lst.append(f)
                f_txt.close()

                dict1 = {"text": f, 
                "author": str(direc),}
                dicts.append(dict1)
    # print(len(dicts))

    train = pd.DataFrame(dicts, columns=["text", "author"])
    #train['text']=[" ".join(txt) for txt in train['text'].values]
    
    return train

def load_test(input_dir):
    test_dicts = []
    
    # convert json file into dataframe
    with open(input_dir + "/" + "ground-truth.json") as data:
        data = json.loads(data.read())
        for el in data.get("ground_truth"):
            # print(el, type(el))
            dict1 = {"text": el.get("unknown-text"), 
                "author": el.get("true-author")}
            test_dicts.append(dict1)

        test = pd.DataFrame(test_dicts, columns=["text", "author"])

    # read and insert txt files in text row
    for i, row in enumerate(test.itertuples()):
        for fin in listdir(input_dir + "/" + "unknown"):
            if str(fin) == str(row[1]):
                # print(fin)
                with open(input_dir + "/" + "unknown" + "/" + fin, "r", encoding="utf-8") as txt:
                    test.loc[i, "text"] = txt.read()
    
    return test
  
def load_pickle(filename):
  with open(filename, 'rb') as f:
    df = pickle.load(f)

  return df

train1 = load_train("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00001")
test1 = load_test("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00001")
train2 = load_train("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00002")
test2 = load_test("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00002")

train1.to_pickle("data/train1.pkl")
test1.to_pickle("data/test1.pkl")
train2.to_pickle("data/train2.pkl")
test2.to_pickle("data/test2.pkl")
# train2.to_excel("data/train2.xlsx")
# train1.to_excel("data/train1.xlsx")

train1_df = load_pickle('data/train1.pkl')
test1_df = load_pickle('data/test1.pkl')
train2_df = load_pickle('data/train2.pkl')
test2_df = load_pickle('data/test2.pkl')