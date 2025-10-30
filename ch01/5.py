
def n_gram_word(target, n):
    target_list = target.split()
    for i in range(len(target_list) - n + 1):
        print(target_list[i:n+i])


def n_gram_character(target, n):
    target_list = list(target)
    for i in range(len(target_list) - n + 1):
        print(target_list[i:n+i])


text = "I am an NLPer"

print("文字tri-gram")
n_gram_character(text, 3)
print("単語bi-gram")
n_gram_word(text, 2)
