
class Animal:
    def __init__(self, name, legsCount):
        self._name = name
        self._legsCount = legsCount
    
    def getName(self):
        return self._name
    
    def getLegsCount(self):
        return self._legsCount
    

class Horse(Animal):
    def __init__(self):
        self._name = "HORSE"
        self._legsCount = 4
    
class Snake(Animal):
    def __init__(self):
        self._name = "SNAKE"
        self._legsCount = 0

class Hen(Animal):
    def __init__(self):
        self._name = "HEN"
        self._legsCount = 2