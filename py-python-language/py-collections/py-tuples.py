print("Tuples are immutable collections of arbitary objects, indicated using round brackets.\nImmutability means you cannot modify the length of the tuple nor change its values")

# Empty Tuple
t1 = ()

t1 = ("India", 54.4, "New Delhi")
print(t1)

print("First Element: ", t1[0])
print("Third Element: ", t1[2])

print("Appending elements to tuple can be achieved using + operator")
print(t1 + (("Telangana", "Hyderanad"), ("Karnataka", "Bangalore")))

print("Printing elements of tuple 4 times using * operator")
print(t1 * 4)

def minmax(x):
    return min(x), max(x)

lower, upper = minmax([2,4,1,7,5,8,7,9])
print("Lowest value is: ", lower)
print("Highest value is: ", upper)

# Swapping variables
lower, upper = upper, lower
print("Lowest value is: ", lower)
print("Highest value is: ", upper)

(a, (b, (c, d))) = (2, (4, (6, 8)))
print("a: ", a , " b: ", b, " c: ", c, " d: ", d)

# Creating tuple from a list
t2 = tuple([1,2,3,4,6,7,8,9])

# Searching elements in tuple
print("Does tuple contains 5 ? ", (5 in t2))
print("Does tuple does not contains 5 ? ", (5 not in t2))