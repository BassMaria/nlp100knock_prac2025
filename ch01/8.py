def cipher(target):
    result = []
    for char in target:
        if 'a' <= char <= 'z':
            original_code = ord(char)
            new_code = 219 - original_code
            new_char = chr(new_code)
            result.append(new_char)
        else:
            result.append(char)
    return "".join(result)


print("文字列を入力してください：")
a = input()
print(cipher(a))
