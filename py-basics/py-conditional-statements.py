print('Understanding Conditional Statements')

a=20
# If Statement
if (a>=20):
    print('Print a is: ', a)

# Elif Statement
if(a<=19):
    print('True')
elif(a>19):
    print('Elif True')
else:
    print('False')

# While loop
c=4
while (c != 0):
    print('C: ', c)
    c -= 1

# While loop + Break Statement
d=4
while (d != -1):
    print('D: ', d)
    if(d == 0):
        print('Breaking while loop')
        break
    d -= 1