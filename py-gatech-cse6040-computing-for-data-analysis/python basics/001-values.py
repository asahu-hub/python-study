
# coding: utf-8

# | &nbsp; | &nbsp; | &nbsp; |
# |--------|--------|--------|
# | [Return to Index Notebook](./index.ipynb) | [View on GitHub](https://github.com/hpcgarage/prymer) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hpcgarage/prymer/blob/master/001-values.ipynb) |

# # Values, variables, and types
# 
# Key concepts
# * Primitive values and types: `int`, `float`, `complex`, `bool`, `str`, and `NoneType`
#     - Type promotion
#     - Type queries
# * Strings
#     - Indexing
#     - Slicing and `slice` objects
# * Booleans and bit manipulation
# * Syntatic sugar: Update operations

# | Type        | Example        | Description                                                  |
# |-------------|----------------|--------------------------------------------------------------|
# | ``int``     | ``x = 1``      | integers (i.e., whole numbers)                               |
# | ``float``   | ``x = 1.0``    | floating-point numbers (i.e., real numbers)                  |
# | ``complex`` | ``x = 1 + 2j`` | Complex numbers (i.e., numbers with real and imaginary part) |
# | ``bool``    | ``x = True``   | Boolean: True/False values                                   |
# | ``str``     | ``x = 'abc'``  | String: characters or text                                   |
# | ``NoneType``| ``x = None``   | Special object indicating nulls                              |

# ## Basics

# Values have types:

# In[34]:


3


# In[35]:


type(3)


# In[36]:


3.0


# In[37]:


type(3.0)


# In[38]:


"3"


# In[39]:


type("3")


# Variables name values:

# In[40]:


x = 5
print(x)


# Values may be combined using the "natural" operators. For integers and floats, for example, you have these options:
# 
# | Operator     | Name           | Description                                            |
# |--------------|----------------|--------------------------------------------------------|
# | ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
# | ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
# | ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
# | ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
# | ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
# | ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
# | ``a ** b``   | Exponentiation | ``a`` raised to the power of ``b``                     |
# | ``-a``       | Negation       | The negative of ``a``                                  |
# | ``+a``       | Unary plus     | ``a`` unchanged (rarely used)                          |

# In[41]:


print(2*x + 3)


# > **Aside.** Formatted I/O and formatted string literals:

# In[42]:


print('{X} has type {T}'.format(X=x, T=type(x)))
print(f'{x} has type {type(x)}')


# **Type promotion.** For some types, the type of the result is promoted automatically:

# In[43]:


x = 1
y = 2
z = x / y
print(x, 'has type', type(x))
print(y, 'has type', type(y))
print(z, 'has type', type(z))


# In[44]:


# Aside: integer division (floor)
print(5 / 2, type(5 / 2))
print(5 // 2, type(5 // 2))


# **Type queries** (asking for a type):

# In[45]:


x = 5
print(type(x) is int)
print(type(x), isinstance(x, int)) # Preferred idiom


# In[46]:


print(type(True), isinstance(True, int))
print(type(5.0), isinstance(5.0, type(x)))
print(type('5'), isinstance('5', type(x)))
print(type(5.0), isinstance(5.0, complex)) # Does not always behave as expected


# In[47]:


(5.0).is_integer() # Special test for floating-point values


# **`None`/`NoneType`.** The "non-value" value, `None`, and its and type:

# In[48]:


x = None
print(x, ':', type(x))
print(x == None, x != None)
print(x is None, x is not None) # Preferred idiom


# **"Big" integers by default.** Integers do not "overflow" as they do in most other languages.

# In[49]:


print('(a)', 3**4)
print('(b)', 3**40)
print('(c)', 3**400)


# **Math functions.** For numerical values, many of the functions you might want are available in the `math` module.

# In[50]:


import math

math


# ## Strings

# In[51]:


bottle = "message"
print(bottle, ':', type(bottle))


# In[52]:


bottle + bottle


# In[53]:


'sag' in bottle


# In[54]:


sixpack = bottle * 6
print("|'{s}'| == {n}.".format(s=sixpack, n=len(sixpack))) # print string and its length


# In[55]:


bottle.upper()


# _Aside:_ In the cell below, type `bottle.` and then press tab.

# **Indexing strings.**

# In[56]:


print(f'|{bottle}| == {len(bottle)}')
print('0:', bottle[0])
print('1:', bottle[1])
print('2:', bottle[2])
print('3:', bottle[3])
print('4:', bottle[4])
print('5:', bottle[5])
print('6:', bottle[6])
# print('7:', bottle[7]) <-- out-of-range


# **Slices.** Let $0 \leq a < b$. Then $a:b$ is a _slice_ that specifies the right-open interval, $[a, b)$.

# In[57]:


print("'{}'".format(bottle[3:5]))
print("'{}'".format(bottle[5:3])) # Empty, since a >= b


# If $a$ (or $b$) is negative, then it is replaced with $n+a$ (or $n+b$), where $n$ is the length of the string. In other words, negative positions are interpreted as "counting from the end."

# In[58]:


print("'{}'".format(bottle[-3:-5])) # n-3:n-5, which is empty since n-3 > n-5
print("'{}'".format(bottle[-5:-3]))


# A slice may have a third parameter, $a:b:s$, where $s$ denotes the _step size_. Again, assume $0 \leq a < b$ and suppose $s > 0$. Then the 3-parameter slice $a:b:s$ expands into the sequence, $a, a+s, a+2s, \ldots, a+(k-1)s$, where $k = \left\lfloor\frac{b-a}{s}\right\rfloor$.
# 
# _Defaults._ Omitting `a`, `b`, or `s` yields the defaults of `a=0`, `b=len(x)`, and `s=1`.

# In[59]:


print(bottle[2:7:2])
print(bottle[2::2])


# Right-open intervals allow simple splittings of intervals:

# In[60]:


print(bottle[:4],  # Up to (but excluding) 4, since intervals are right-open
      bottle[4:])  # Resume at 4

k = len(bottle) - 3 # Note: len(bottle) == 7, so this is same as above
print(bottle[:k], bottle[k:])

k = -3 # Recall: a negative endpoint x becomes n+x
print(bottle[:-3], bottle[-3:]) # Shorthand


# **Negative steps** ($s < 0$) reverse the direction. That is, if $s < 0$ then one cannot have a non-empty sequence unless $a > b$. The defaults for omitted $a$ and $b$ values change to $a=n-1$ and $b=-1$ (recall that the intervals are right-open).

# In[61]:


print(bottle[::-1])
print(bottle[::-2], bottle[6::-2])
print("'{}'".format(bottle[0::1]))
print("'{}'".format(bottle[0::-1]))


# _Aside:_ **Slices are objects!**

# In[62]:


ind = slice(6, None, -2)
print(bottle[6::-2], bottle[ind])


# ## Booleans and bit manipulation
# 
# Boolean variables can take on the values of `True` or `False`. The built-in boolean operations are `and`, `or`, and `not`.

# In[63]:


print(True and False)
print(True or True)
print(not True)


# In addition to booleans, you can also perform bit-level manipulations on integers. The following operators perform logical operations bitwise between the corresponding bits of the operands.
# 
# 
# | Operator     | Name            | Description                                 |
# |--------------|-----------------|---------------------------------------------|
# | ``a & b``    | Bitwise AND     | Bits defined in both ``a`` and ``b``        |
# | <code>a &#124; b</code>| Bitwise OR      | Bits defined in ``a`` or ``b`` or both      |
# | ``a ^ b``    | Bitwise XOR     | Bits defined in ``a`` or ``b`` but not both |
# | ``a << b``   | Bit shift left  | Shift bits of ``a`` left by ``b`` units     |
# | ``a >> b``   | Bit shift right | Shift bits of ``a`` right by ``b`` units    |
# | ``~a``       | Bitwise NOT     | Bitwise negation of ``a``                          |

# First, some different ways to inspect the binary representations of integer values as strings (see also [`bin()`](https://docs.python.org/3/library/functions.html#bin) in the Python docs):

# In[64]:


print(bin(5))
print(format(5, '#b'))
print(format(5, 'b')) # no prefix
print(f'{5:#b}', f'{5:b}')
print('{:07b}'.format(5)) # 7 bits with leading zeros


# In[65]:


print('{:06b}'.format(5)) # or use '{:#06b}' for a 6-bit string prefixed by '0b'
print('{:06b}'.format(13))
print('{:06b}'.format(5 & 13))


# ## Syntactic sugar: update operations

# In[66]:


x = 5
y = x
y += 3  # Same as y = y + 3
print(x, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




