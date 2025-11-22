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
print("係り元\t係り先")
for token in doc:
    # 一つの文や節の中心となる述語や動詞の属性は
    # ROOTでこれは自分自身を指すので消した
    if token.dep_ != "ROOT":
        source = token.text
        target = token.head.text
        print(source + '\t' + target)