print('Lists are mutable sequences of objects, indicated through square brackets')

# Initializing an empty list
b=[]

a=[1,2,3]
print("List elements: ", a)

a[1] = "apple"
print("Modified List Elements: ", a)

b.append("Orange")
b.append("Guava")
print("Appended List Elements: ", b)

c = list("Characters")
print("Printing list innitialized with a single string: ",c)

print("\n")
# Generating list from a range
generatedList = range(1, 30, 3)

# Enumerating over the list, by recognizing its index and value
for index, value in enumerate(generatedList):
    print("Index: {}\tValue: {}".format(index, value))
print("\n")