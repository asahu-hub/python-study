'''
    Gaussian/Normal Distribution:

        1. Gaussian distribution is also called as Normal Distribution. Many observations in nature fit Normal Distribution.
        2. Distribution of data refers to the shape it has when you graph it, such as with a histogram. The most common and well-known distributin of contnuous values is the bell curve. It is known as the Normal Distribution because it is the distribution that a lof data falls into.
'''
from numpy import arange
from matplotlib import pyplot as pp
from scipy.stats import norm


x_axis = arange(-3, 3, 0.001)
y_axis = norm.pdf(x_axis, 0, 1)

pp.plot(x_axis, y_axis)
pp.title("Randomly Generated Gaussian Distribution")
pp.show()



