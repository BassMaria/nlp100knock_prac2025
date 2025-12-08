import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

# 前処理部分
df = pd.read_csv("questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5033]

country_list = []
for item in df["v4"].unique():
    # モデルに存在する国名のみを抽出し順番を固定
    country = list(set(df["v4"].values)) 

model_path = 'GoogleNews-vectors-negative300.bin.gz'
print("Loading model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 国名ベクトル抽出と対応する国名リストの作成
countryVec = []
countryName = []
for c in country:
    # モデルに単語が存在するか確認する
    if c in model:
        countryVec.append(model[c])
        countryName.append(c)

X = np.array(countryVec)
# KMeans実行部分
km = KMeans(n_clusters=5, random_state=0, n_init='auto')
y_km = km.fit_predict(X)

# 国名リストとクラスタラベルを結合
result_df = pd.DataFrame({'Country': countryName, 'Cluster': y_km})

print("\n--- K-Means クラスタリング結果 (k=5) ---")
print(result_df.head(10)) # 先頭10件の表示

# クラスタごとに国名を一覧表示
for cluster_id in sorted(result_df['Cluster'].unique()):
    cluster_countries = result_df[result_df['Cluster'] == cluster_id]['Country'].tolist()
    print(f"\n--- Cluster {cluster_id} ({len(cluster_countries)} countries) ---")
    print(cluster_countries)