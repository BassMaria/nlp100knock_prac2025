a = "パトカー"
b = "タクシー"

# a,bは同じ文字数と仮定した
ans = ""
for i in range(len(a)):
    ans += a[i]+b[i]


print(ans)
