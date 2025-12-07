from gensim.models import KeyedVectors

# 学習済み単語ベクトル path
model_path = 'GoogleNews-vectors-negative300.bin.gz'

print("Loading model...")
# バイナリ形式のモデルを読み込む
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# "United_States"と"U.S."のコサイン類似度
print(model.similarity("United_States", "U.S."))