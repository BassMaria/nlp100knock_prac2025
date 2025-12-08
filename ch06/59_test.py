import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
# t-SNEインポート
from sklearn.manifold import TSNE
# 前処理部分
df = pd.read_csv("questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5033]
country = list(set(df["v4"].values))


model_path = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

countryVec = []
for c in country:
    # モデルに単語が存在するか確認する
    countryVec.append(model[c])

X = np.array(countryVec)
# t-SNEでの次元削減(ramdom_stateは再現性のためのシード)
# コサイン距離で意味的な距離を保持
tsne = TSNE(random_state=0, metric="cosine")
embs = tsne.fit_transform(X)
# 散布図の描画
plt.figure(figsize=(15, 15)) # グラフのサイズを大きくしてラベルを見やすくする
plt.scatter(embs[:, 0], embs[:, 1])

# 各点に国名（countryリストの要素）をテキストとして追加
# countryリストとembs配列の順番は対応
for i, label in enumerate(country):
    # テキストをプロットの位置embs[i, 0], embs[i, 1]に描画する
    plt.annotate(label, (embs[i, 0], embs[i, 1]), alpha=0.7, fontsize=8) # alphaとfontsizeで視認性を調整

plt.title("t-SNE Visualization of Country Word Vectors (Cosine Distance)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig("out/59_TSNE_labeled.png")
plt.show()