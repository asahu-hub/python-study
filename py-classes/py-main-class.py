from py_hello_world import Calculator
from py_animals import *

calc = Calculator(10,0)
print(type(calc))

print("Sum of two numbers: {0}\nDifference of two numbers: {1}\nProduct of two numbers:{2}\nRemainder of two numbers:{3}".format(calc.add(), calc.subtract(), calc.multiply(), calc.division()))

hen = Hen()
horse = Horse()
snake = Snake()

print("Animal name: {0} & number of legs: {1}\n".format(hen.getName(), hen.getLegsCount()))
print("Animal name: {0} & number of legs: {1}\n".format(horse.getName(), horse.getLegsCount()))
print("Animal name: {0} & number of legs: {1}\n".format(snake.getName(), snake.getLegsCount()))

