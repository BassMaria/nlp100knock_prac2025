import google.genai as genai
import os

prompt = """
「おじさんあるある」をお題として
川柳の案を10個作成してください．
なお"川柳のみ"10個出力してください．
"""

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("応答:")
print(response.text)
