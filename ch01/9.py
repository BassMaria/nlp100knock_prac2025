import re
import random


def typoglycemia(text):
    def shuffle_word(match):
        word = match.group(0)
        if len(word) <= 4:
            return word

        first = word[0]
        last = word[-1]

        middle = list(word[1:-1])
        random.shuffle(middle)

        shuffled_word = first + "".join(middle) + last
        return shuffled_word
    sentence_list = text.split(' ')
    shuffled_sentence = []

    for word in sentence_list:
        processed_word = re.sub(r'\b[a-zA-Z]+\b', shuffle_word, word)
        shuffled_sentence.append(processed_word)
    return ' '.join(shuffled_sentence)


sample_text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
result = typoglycemia(sample_text)
print(result)
