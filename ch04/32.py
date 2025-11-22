import spacy
nlp = spacy.load('ja_ginza')
text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
doc = nlp(text)
for i in range(len(doc) - 2):
    if (doc[i].pos_ == "NOUN") and (doc[i+1].text == "の") and (doc[i+2].pos_ == "NOUN"):
        result = doc[i].text + doc[i+1].text + doc[i+2].text
        print(result)