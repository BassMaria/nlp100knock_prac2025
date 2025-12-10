import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import confusion_matrix

def add_feature(sentence):
    data = {"feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

# 検証データの読み込み
df = pd.read_csv("SST-2/dev.tsv", sep="\t")

# モデルとベクトライザーの読み込み
with open("out/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("out/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 特徴ベクトルの構築
features = [add_feature(text)["feature"] for text in df["sentence"]]

# 特徴ベクトルの変換
X = vec.transform(features)

# 予測値の算出
y_pred = model.predict(X)

# 正解ラベル
y_true = df["label"]

# 混同行列
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
