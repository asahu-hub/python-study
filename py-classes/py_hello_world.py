''' __init__() method is called by python environment
after createing the object. Its an initializer and not
a Constructor. There is no concept of Constructors in Python'''

class Calculator:
    def __init__(self, number1, number2):
        self._number1 = number1
        self._number2 = number2

    def add(self):
        return self._number1 + self._number2

    def subtract(self):
        if (self._number1 < self._number2):
            return self._number1 - self._number2
        elif(self._number2 < self._number1):
            return self._number2 - self._number1
        else:
            return 0
    
    def multiply(self):
        return self._number1 * self._number2
    
    def division(self):
        if not (self._number2 == 0):
            return self._number1 / self._number2
        else:
            return float("inf")
