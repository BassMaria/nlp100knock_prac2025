import pandas as pd
import pickle

# モデルとベクトライザーの読み込み
with open("out/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("out/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# ロジスティック回帰の係数（shape: (1, n_features)）
coef = model.coef_[0]

# 特徴量名
feature_names = vec.get_feature_names_out()

# 特徴量と重みをデータフレーム化
df_coef = pd.DataFrame({
    "feature": feature_names,
    "weight": coef
})

# 正の重みトップ20
top20 = df_coef.sort_values(by="weight", ascending=False).head(20)

# 負の重みトップ20
bottom20 = df_coef.sort_values(by="weight").head(20)

print("=== 重みが大きい特徴量 Top 20 ===")
print(top20)

print("\n=== 重みが小さい特徴量 Top 20 ===")
print(bottom20)
