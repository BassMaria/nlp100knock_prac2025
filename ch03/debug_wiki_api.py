import requests
import json

# ======== ここはテストしたい値に書き換えてください =========
# Wikipedia API endpoint
API_URL = "https://ja.wikipedia.org/w/api.php"

# 例: 英国国旗画像名（空白は半角スペースのままでOK、API内部で処理される）
filename = "Flag of the United Kingdom.svg"
# =============================================================


# APIに渡すパラメータ
params = {
    "action": "query",
    "titles": "File:" + filename,
    "prop": "imageinfo",
    "iiprop": "url",
    "format": "json"
}

# User-Agent (重要：未指定だとブロックされる可能性あり)
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; WikiAPI-Check/1.0)"
}

# API呼び出し
response = requests.get(API_URL, params=params, headers=headers)

print("=== Request Info ===")
print("REQUEST URL:", response.url)
print("STATUS CODE:", response.status_code)
print()

# レスポンス内容(先頭500文字のみ表示)
print("=== Raw Response Text (top 500 chars) ===")
print(response.text[:500])
print()

# JSONデコードチェック
print("=== JSON Decode Check ===")
try:
    data = response.json()
    print("JSON decode: OK")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:800])
except json.JSONDecodeError:
    print("JSON decode: FAILED (返却値はJSONではありません)")
