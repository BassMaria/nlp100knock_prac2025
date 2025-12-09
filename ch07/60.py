import pandas as pd

train = pd.read_csv("SST-2/train.tsv", sep="\t")
dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

print("ポジティブ(1)とネガティブ(0)")
print("trainの事例数")
print(train["label"].value_counts())
print("devの事例数")
print(dev["label"].value_counts())
