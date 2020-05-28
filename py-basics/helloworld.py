print('Hello World, welcome to python programming')

from math import factorial as fact
n,k = 5,3
print("Permutation of 5 over 3 is: ", fact(n) / (fact(k) * fact(n-k)))
