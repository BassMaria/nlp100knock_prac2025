import spacy
from spacy import displacy
nlp = spacy.load('ja_ginza')

text = "メロスは激怒した。"
doc = nlp(text)

# displacy.render(doc, style="dep")

displacy.serve(doc, style="dep")
# 実行後http://127.0.0.1:5000へ