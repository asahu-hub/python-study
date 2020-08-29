
# coding: utf-8

# | &nbsp; | &nbsp; | &nbsp; |
# |--------|--------|--------|
# | [Return to Index Notebook](./index.ipynb) | [View on GitHub](https://github.com/hpcgarage/prymer) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hpcgarage/prymer/blob/master/004-modules.ipynb) |

# # Modules and Packages

# Beyond the basic language features covered in earlier notebooks, Python has a rich ecosystem of third-party modules. Think of a module as code someone else has written and packaged up for others to reuse; Python's modules are sometimes called "libraries" in other languages.

# You've already seen one example in a previous notebook, namely, the `math` module [docs](https://docs.python.org/3/library/math.html), which defines basic numerical constants and primitives, like the constant $\pi$ or the transcendental functions (e.g., $\cos$). You access these functions by first importing the `math` module and then referencing the functions you need using `module.object` syntax:

# In[1]:


import math
math.cos(math.pi)


# **Import by alias.** Sometimes a module has a long name, so there are other ways to import it to save a bit of typing.

# In[2]:


import numpy as np  # You'll use this module a lot more later in the course!
np.cos(np.pi)


# **Importing module contents.** You can also import the contents of a module directly into the current "namespace," so that they are visible as if they were defined locally.

# In[3]:


from math import cos, pi
cos(pi)


# It is also possible to import _all_ of a module's contents into the current namespace, that is, without having to name each item individually. However, this practice is strongly discouraged; refer to the "Whirlwind Tour" book for an explanation of why not.

# ## Importing from Python's Standard Library
# 
# Every Python installation will include the "standard library," which is one collection of handy modules [standard library docs](https://docs.python.org/3/library/). Here are some examples, and you'll see these and others pop up throughout the course.
# 
# - ``os`` and ``sys``: Tools for interfacing with the operating system, including navigating file directory structures and executing shell commands
# - ``math`` and ``cmath``: Mathematical functions and operations on real and complex numbers
# - ``itertools``: Tools for constructing and interacting with iterators and generators
# - ``functools``: Tools that assist with functional programming
# - ``random``: Tools for generating pseudorandom numbers
# - ``json`` and ``csv``: Tools for reading JSON-formatted and CSV-formatted files.

# **Third-party modules.** Besides the standard library, there are many more "third-party" modules developed by independent developers and organizations. These include the core data science stack in Python, which includes [Numpy/Scipy](https://www.scipy.org) and [pandas](https://pandas.pydata.org), among others we'll be using. In many cases, to use a particular third-party module you may need to install it first; we won't cover how to do that since in our class you'll be using a particular standard environment.

# In[ ]:




