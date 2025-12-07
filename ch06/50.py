from gensim.models import KeyedVectors

# 学習済み単語ベクトル path
model_path = 'GoogleNews-vectors-negative300.bin.gz'

print("Loading model...")
# バイナリ形式のモデルを読み込む
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# "United_States"の単語ベクトル表示
target_word = "United_States"
if target_word in model:
    vector = model[target_word]
    print(f"Vector for '{target_word}':")
    print(vector)
    # ベクトル次元数
    print(f"Vector dimension: {len(vector)}")
else:
    print(f"'{target_word}'はないよ")
