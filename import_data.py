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
                f_txt = open(input_dir + "/" + direc + "/" + fin, "r+", encoding="utf-8")
                f = f_txt.read()
                f_txt.close()

                dict1 = {"text": f, 
                "author": str(direc), "length": len(f)}
                dicts.append(dict1)

    train = pd.DataFrame(dicts, columns=["text", "author", "length"])
    
    return train

def load_test(input_dir):
    test_dicts = []
    
    # convert json file into dataframe
    with open(input_dir + "/" + "ground-truth.json") as data:
        data = json.loads(data.read())
        for el in data.get("ground_truth"):
            dict1 = {"text": el.get("unknown-text"), 
                "author": el.get("true-author"), "length": 0}
            test_dicts.append(dict1)

        test = pd.DataFrame(test_dicts, columns=["text", "author", "length"])

    # read and insert txt files in text row
    for i, row in enumerate(test.itertuples()):
        for fin in listdir(input_dir + "/" + "unknown"):
            if str(fin) == str(row[1]):
                with open(input_dir + "/" + "unknown" + "/" + fin, "r", encoding="utf-8") as txt:
                    test.loc[i, "text"] = txt.read()
                    test.loc[i, "length"] = len(test.loc[i, "text"])
    
    return test

def merge_data(df1, df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    df.reset_index(drop=True)
    return df
  
def load_pickle(filename):
  with open(filename, 'rb') as f:
    df = pickle.load(f)

  return df

train1 = load_train("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00001")
test1 = load_test("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00001")
train2 = load_train("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00002")
test2 = load_test("pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/problem00002")

set1 = merge_data(train1, test1)

# print("full set 1: ", set1, set1.columns)

train1.to_pickle("data/train1.pkl")
test1.to_pickle("data/test1.pkl")
train2.to_pickle("data/train2.pkl")
test2.to_pickle("data/test2.pkl")

train1_df = load_pickle('data/train1.pkl')
test1_df = load_pickle('data/test1.pkl')
train2_df = load_pickle('data/train2.pkl')
test2_df = load_pickle('data/test2.pkl')