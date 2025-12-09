import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pickle

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

# 特徴ベクトルの変換
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

# ロジスティック回帰学習
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

with open("out/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("out/vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)
