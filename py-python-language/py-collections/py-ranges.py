print("Sequence representing Arithmetic Progression of Integers")

for i in range(3):
    print("i: ", i)
print("\n")
for j in range(4, 8):
    print("j: ", j)
print("\n")
for k in range(1, 10, 2):
    print("k: ", k)
print("\n")

# Generating list from a range
generatedList = range(1, 30, 3)

# Enumerating over the list, by recognizing its index and value
for index, value in enumerate(generatedList):
    print("Index: {}\tValue: {}".format(index, value))
print("\n")