import pickle
from collections import defaultdict


def add_feature(sentence):
    data = {"feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data


def predict_sentiment(text):
    # モデルとベクトライザーの読み込み
    with open("out/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("out/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    # 特徴ベクトルの構築
    data = add_feature(text)

    # 特徴ベクトルの変換
    X = vec.transform([data["feature"]])

    # 予測
    predicted_label = model.predict(X)[0]
    predicted_prob = model.predict_proba(X)[0]

    # 結果の表示
    sentiment = "ポジティブ" if predicted_label == 1 else "ネガティブ"
    print(f"テキスト: {text}")
    print(f"予測された感情: {sentiment}")
    print(
        f"予測確率: ネガティブ={predicted_prob[0]:.4f}, ポジティブ={predicted_prob[1]:.4f}"
    )


test_text = "the worst movie I 've ever seen"
predict_sentiment(test_text)
