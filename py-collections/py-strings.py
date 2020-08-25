print('Example usage of str collection')
print('Str are immutable sequence of unicode characters')

a = 'This is a string delimited with single quotes'
b = "This is a string delimited with double quotes"

uncompressed_name = 'message_in_a_bottle.txt'
compressed_extension = '.zip'

print(a)
print(b)

concatenatedString = " ".join(["This", "is", "a", "concatenated", "string"])
print(concatenatedString)
print(concatenatedString.split(" "))

print(uncompressed_name+compressed_extension)

print("""This is a multi-line
string, recognized by three quotes""")

print('Raw String: ', r'C:\Documents\User\212400')

# c="String accessed as Character Array"
# counter = 0
# while(counter < len(c)):
#    print(c[counter])
#    counter += 1

# Partitioning String - Output: {Before, Separator, After}
print("Paranthesis".partition("th"))

# Formatting String using position
print("Age of the current Prime Minister is {0}".format(45))

# Formatting String using name
print("Age of wife of current Prime Minister is {age}".format(age=40))