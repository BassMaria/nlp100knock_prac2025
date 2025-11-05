# 集合に変換するだけなのでtupleを使用してみた
def n_gram_character(target, n):
    output = []
    target_list = list(target)
    for i in range(len(target_list) - n + 1):
        output.append(tuple(target_list[i:n+i]))
    return output


def check_in_set(check, set_target):
    return check in set_target


a = "paraparaparadise"
b = "paragraph"

x_bi_gram = n_gram_character(a, 2)
y_bi_gram = n_gram_character(b, 2)

X = set(x_bi_gram)
Y = set(y_bi_gram)
# orは初めの真値,andは最後の真値を返してしまうので×
print(f"和集合：{X | Y}")
print(f"積集合：{X & Y}")
print(f"差集合：{X - Y}")

check = ('s', 'e')
print("seがあればTrue,無いならFalse")
print("Xは" + str(check_in_set(check, X)))
print("Yは" + str(check_in_set(check, Y)))
