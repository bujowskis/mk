size = 3
a = [[1,2,3],[1,2,3],[1,2,3]]
new_a = []
for i in a:
    new_a.extend(i)
sorted_a = sorted(new_a)
print(sorted_a)
for i in range(3):
    shift = i*size
    print(sorted_a[shift:shift+size])
