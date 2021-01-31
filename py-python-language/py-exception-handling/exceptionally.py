''' Module for demonstrating exception handling in Python '''
import os

def convert(str):
    try:
        x = int(str)
        print("Successful Execution")
    except (ValueError, TypeError) as e:
        print(e)
        x = 0
    finally:
        print("Running finally block")
    return x

print(convert("2"))
print(convert("abc"))