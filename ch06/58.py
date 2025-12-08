import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
# 階層型クラスタリングとデンドログラム描画のためにインポート
from scipy.cluster.hierarchy import dendrogram, linkage
# 前処理部分
df = pd.read_csv("questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5033]
country = list(set(df["v4"].values))


model_path = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 国名ベクトル抽出と対応する国名リストの作成
countryVec = []
countryName = []
for c in country:
    # モデルに単語が存在するか確認する
    countryVec.append(model[c])
    countryName.append(c)

X = np.array(countryVec)
# Ward法（最小分散増加法）によるクラスタリングの実行
# Ward法はmetric='euclidean'と組み合わせるのが標準的
linkage_result = linkage(X, method="ward", metric="euclidean")
# クラスタリング結果をデンドログラムとして可視化
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor="w", edgecolor="k")
# デンドログラムを描画
dendrogram(linkage_result, labels=countryName)
plt.savefig("out/58_word.png")
plt.show()