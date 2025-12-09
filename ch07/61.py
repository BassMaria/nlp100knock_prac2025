import pandas as pd
from collections import defaultdict

train = pd.read_csv("SST-2/train.tsv", sep="\t")
dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        # ここでBoWの特徴量（単語の出現回数）をカウントしている
        data["feature"][token] += 1 
    return data

data_train = []
for sentence, label in zip(train["sentence"], train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(dev["sentence"], dev["label"]):
    data_dev.append(add_feature(sentence, label))

print("train")
print(data_train[0])

print("dev")
print(data_dev[0])
