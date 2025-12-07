from gensim.models import KeyedVectors

# 学習済み単語ベクトル path
model_path = 'GoogleNews-vectors-negative300.bin.gz'

print("Loading model...")
# バイナリ形式のモデルを読み込む
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

result = model.most_similar(positive=["United_States"], topn=10)
print(result)