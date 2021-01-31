print("Advance Lists")

wordsList = "This is a generated list from words using split method".split()
print(wordsList)
print("Number of words in the list are {0}".format(len(wordsList)))
print("wordsList[9]: {0}\twordsList[-1]:{1}".format(wordsList[9], wordsList[-1]))
print("wordsList[8]: {0}\twordsList[-2]:{1}".format(wordsList[8], wordsList[-2]))
print(wordsList[-1:-10])
print("wordsList[2:5]: {0}".format(wordsList[2:5]))

### Lists are heterogenous
hl=[1,2,'3',4,'five',True]
print(hl)
print(isinstance(hl[0], int))
print(isinstance(hl[2], str))
print(isinstance(hl[5], bool))

### Reverse the List
hl.reverse()
print(hl)
hl.reverse()
print(hl)