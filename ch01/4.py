text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. \
    New Nations Might Also Sign Peace Security Clause. Arthur King Can."
rm_text = text.replace(".", "").replace(",", "")
list_text = rm_text.split()
# print(list_text)
keys = []
values = []

check_index = [1, 5, 6, 7, 8, 9, 15, 16, 19]
# indexは0からのため調整(冗長なので削除)
# match_index = [x - 1 for x in check_index]
# print(match_index)

# enumerateはindex,要素を取得
for i, word in enumerate(list_text):
    if i + 1 in check_index:
        values.append(word[0])
    else:
        values.append(word[:2])
    keys.append(i + 1)

result = dict(zip(keys, values))
print(result)
