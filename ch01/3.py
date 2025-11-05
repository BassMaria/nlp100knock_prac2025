text = "Now I need a drink, alcoholic of course, \
    after the heavy lectures involving quantum mechanics."
# 「\」は行継続文字で他に「"""」や「'''」がある
rm_text = text.replace(",", "").replace(".", "")

list_text = rm_text.split()
cnt = []
for i in list_text:
    cnt.append(len(i))

print(cnt)
