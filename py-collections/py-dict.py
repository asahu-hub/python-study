print("Dictionaries are mutable sequences of keys mapped to values")

# Sample Dict = {k1: v1, k2: v2}
emptyDict = {}

empD = {}

empD['Akshay'] = 'Sahu'
empD['Sachin'] = 'Tendulkar'

print(empD)

print("Last Name of Akshay is: ", empD['Akshay'])

print("All Dict keys: ", empD.keys())
print("All Dict values: ", empD.values())
print("All Dict Items: ", empD.items())

print("length: ", len(empD))
# Check if Key is present in the Dictionary
print('Akshay' in empD)
print('Virat' in empD)