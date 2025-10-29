text = "Now I need a drink, alcoholic of course, \
    after the heavy lectures involving quantum mechanics."

rm_text = text.replace(",", "").replace(".", "")

list_text = rm_text.split()
cnt = []
for i in list_text:
    cnt.append(len(i))

print(cnt)
