import spacy
nlp = spacy.load('ja_ginza')
doc = nlp("猫が元気に庭を走り回る")

for token in doc:
    print(f"単語:{token.text},  pos_:{token.pos_},  tag_:{token.tag_}")
    
