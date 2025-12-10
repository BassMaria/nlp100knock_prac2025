import pandas as pd
import pickle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def add_feature(sentence):
    data = {"feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

#  1. データ読み込み 
train_df = pd.read_csv("SST-2/train.tsv", sep="\t")
dev_df   = pd.read_csv("SST-2/dev.tsv",   sep="\t")

#  2. ベクトライザ読み込み 
with open("out/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

#  3. 特徴量の構築 
train_features = [add_feature(text)["feature"] for text in train_df["sentence"]]
dev_features   = [add_feature(text)["feature"] for text in dev_df["sentence"]]

X_train = vec.transform(train_features)
X_dev   = vec.transform(dev_features)

y_train = train_df["label"]
y_dev   = dev_df["label"]

#  4. 正則化パラメータ C を変化させる 
C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []

for C in C_list:
    model = LogisticRegression(max_iter=300, C=C)
    model.fit(X_train, y_train)
    pred = model.predict(X_dev)
    acc = accuracy_score(y_dev, pred)
    accuracies.append(acc)
    print(f"C={C}: accuracy={acc:.4f}")

#  5. グラフ描画 
plt.figure(figsize=(6, 4))
plt.plot(C_list, accuracies, marker="o")
plt.xscale("log")
plt.xlabel("Regularization parameter C (log scale)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Regularization Strength C")
plt.grid(True)
plt.savefig("69.png")
plt.show()
