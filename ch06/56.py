import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm
# 学習済み単語ベクトル path
model_path = 'GoogleNews-vectors-negative300.bin.gz'

def culcCosSim(row):
    global model
    try:
        return model.similarity(row["Word 1"], row["Word 2"])
    except KeyError:
        return None


print("Loading model...")
tqdm.pandas()
# バイナリ形式のモデルを読み込む
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
df = pd.read_csv("wordsim353/combined.csv")
df["cosSim"] = df.progress_apply(culcCosSim, axis=1)
df = df.dropna()

print(df[["Human (mean)", "cosSim"]].corr(method="spearman"))