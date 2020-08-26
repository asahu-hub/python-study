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
