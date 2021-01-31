def add(x, y):
    """ The function will add the two numbers, x & y 
    
    Args:
        x: Source
        y: Destination

    Returns:
        The sum of two numbers
    """
    return x + y

def subtract(x, y):
    if(x < y):
        return y-x
    else:
        return x-y

def multiply(x, y):
    return x * y

def divide(x, y):
    if(y == 0):
        return float("inf")
    else:
        return x / y

print(__name__)