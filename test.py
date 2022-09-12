center=9391
next=9440
num=next-center

list_of=[center]
iter=6

current=center
for i in range(iter):
    current-=num
    list_of.append(current)

list_of.reverse()
current=center
for i in range(iter):
    current+=num
    list_of.append(current)
print(list_of)