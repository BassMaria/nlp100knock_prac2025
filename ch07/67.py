import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def add_feature(sentence):
    data = {"feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

#  データ読み込み 
train_df = pd.read_csv("SST-2/train.tsv", sep="\t")
dev_df   = pd.read_csv("SST-2/dev.tsv",   sep="\t")

#  モデル読み込み 
with open("out/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("out/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

#  特徴量作成 
train_features = [add_feature(text)["feature"] for text in train_df["sentence"]]
dev_features   = [add_feature(text)["feature"] for text in dev_df["sentence"]]

X_train = vec.transform(train_features)
X_dev   = vec.transform(dev_features)

y_train = train_df["label"]
y_dev   = dev_df["label"]

#  予測 
train_pred = model.predict(X_train)
dev_pred   = model.predict(X_dev)

#  各種スコア 
def print_scores(name, y_true, y_pred):
    print(f"=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print()

print_scores("Train", y_train, train_pred)
print_scores("Dev",   y_dev,   dev_pred)
