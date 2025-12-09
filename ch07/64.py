import pandas as pd
from collections import defaultdict
import pickle


def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

with open("out/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("out/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

# 検証データの先頭の事例
first_sentence = dev["sentence"].iloc[0]
first_label = dev["label"].iloc[0]

# 特徴ベクトルの構築
data = add_feature(first_sentence, first_label)
# 特徴ベクトルの変換
X = vec.transform([data["feature"]])

# 予測
predicted_label = model.predict(X)[0]
predicted_prob = model.predict_proba(X)[0]


print(f"sentence: {first_sentence}")
print(f"True label: {first_label}")

print("形式: [ネガティブ(0) ポジティブ(1)]")
print(f"条件付き確率: {predicted_prob}")
