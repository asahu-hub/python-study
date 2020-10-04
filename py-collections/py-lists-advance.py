print("Understanding advance operations on Python Lists")

from random import shuffle

hl=[1,2,'3',4,'five',True]

### Create a new List from existing list, but double the value of each item in existing list.
nl=[2*item for item in hl]
print(nl)

randomlist=[i for i in range(1,27,3)]
print(randomlist)

shuffle(randomlist)
print(randomlist)

evenrandomlist=[item for item in randomlist if (item%2)==0]
print(evenrandomlist)

## Zip two lists and iterate over both their items at the same time.
for item1,item2 in zip(hl, nl):
    print(item1, '- double value -', item2)


## Iterator in Lists
listiterator=iter(randomlist)
#print(listiterator, "\n")

for item in listiterator:
    print(item)

floatsrandomlist=[1.2345665, 2.3435656, 3.564786987, 22.3343434343, 1.92345]
print("Total Sum:", sum(floatsrandomlist))


