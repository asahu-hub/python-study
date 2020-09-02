
# coding: utf-8

# | &nbsp; | &nbsp; | &nbsp; |
# |--------|--------|--------|
# | [Return to Index Notebook](./index.ipynb) | [View on GitHub](https://github.com/hpcgarage/prymer) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hpcgarage/prymer/blob/master/003-control-flow.ipynb) |

# # Control flow
# 
# In earlier parts of this bootcamp, "code" consists of a linear sequence of statements, and the program executes each line one after the other. But the "flow" of execution from one statement to the next need not obey a strict linear path. The term _control flow_ (or control structures, as opposed to data structures) refers to code constructs—like "if" statements, "for" loops, and "function calls"—that cause statements to execute in a nonlinear order.
# 
# Key concepts:
# * Functions
# * Conditionals (`if`-`elif`-`else`; inline `if`-`else`)
# * For-loops and iterators
# * While-loops and other loop constructs (`break`, `continue`)
# 
# For more detailed information, see Vanderplas or the [Python tutorial on control structures](https://docs.python.org/3/tutorial/controlflow.html).

# ## Aside: Assertions
# 
# With more complex forms of program execution, writing correct programs and debugging incorrect ones becomes much harder. Therefore, we may want to check that variables have certain values or that objects meet certain conditions, or to impose simplifying assumptions on our code so that we can start building working pieces. For example, if you are writing a program to compute $n!$ (the factorial of $n$), perhaps you want to restrict the input values of $n$ that you wish to consider to be non-negative. A useful code construct in that case is an `assert` statement, which has the form,
# 
# ```python
#     assert boolean_condition, error_string
# ```
# 
# which does nothing if `boolean_condition` is `True` or, if it is `False`, aborts the execution and prints the message `error_string`.

# In[1]:


n = 0  # Try changing `0` to `-3`
assert n >= 0, f"`n` must be non-negative but instead has the value {n}."


# ## Functions

# **Defining functions.** Consider this example and note the following:
# - `def` statement, which names the function, declares its arguments, and ends in a `:`
# - An optional document string, which can be retrieved by "help" queries
# - Indentation: In Python, indentation is significant and strictly enforced. Blocks of code must be indented consistently (see [Lexical analysis](https://docs.python.org/3/reference/lexical_analysis.html) in the Python documentation)
# - The `return` statement, which specifies the value that the function returns to the caller

# In[2]:


def evens(a, b):
    """
    Returns a list of even integers in `[a, b)`,
    given integers `a` and `b`.
    """
    assert isinstance(a, int) and isinstance(b, int),            f"One of `a:{type(a)}` or `b:{type(b)}` is not an integer."
    return [x for x in range(a, b) if (x % 2) == 0]


# In[3]:


get_ipython().magic('pinfo evens')


# In[4]:


evens(-7, 10)


# In[5]:


def foo():
    return 'a', 1.23, 'b'

foo()


# **`None` returns.** A function that lacks a return statement still produces a value, namely, `None` (of type `NoneType`).

# In[6]:


def no_op():
    pass

print(no_op(), type(no_op()))


# **Functions are objects.**

# In[7]:


print(evens, type(evens))


# In[8]:


def run_two_param_function(fun, x, y):
    assert callable(fun),            f"First arg is not callable; it's a {type(fun)})."
    return fun(x, y)


# In[9]:


print(run_two_param_function(evens, 3, 12))
print(run_two_param_function(divmod, 13, 3))  # `divmod` is a built-in function


# **Anonymous, or _lambda_, functions.** Occasionally, you need to define a short, single-use function that returns a value. In such cases, the `lambda` construct gives you a succint way to do so.
# 
# In this first example, the `lambda` defines a two-parameter function that, given arguments named $x$ and $y$, returns $2x - y$.

# In[10]:


def somefunction(x, y):
    return 2*x - y


# In[11]:


run_two_param_function(lambda x, y: 2*x - y,
                       7, 3)


# Here is another example. First, consider Python's built-in [`sorted` function](https://docs.python.org/3/howto/sorting.html), which returns a sorted instance of an input collection.

# In[12]:


sorted([8, 3, 2, -7, 10, 11, 4, 1 -4])


# However, suppose our list has more complex values, like this list of dictionaries:

# In[13]:


data = [{'first':'Guido', 'last':'Van Rossum', 'YOB':1956},
        {'first':'Grace', 'last':'Hopper',     'YOB':1906},
        {'first':'Alan',  'last':'Turing',     'YOB':1912}]


# Suppose we wish to sort this list by year-of-birth. The `sorted` function takes an optional parameter that allows us to define the sort key. The key is given by a function that, given a list element, returns the value to use for sorting:

# In[14]:


sorted(data, key=lambda x: x['YOB'])


# Conceptually, the following are equivalent:

# In[15]:


def get_year_of_birth__v1(x):
    return x['YOB']

get_year_of_birth__v2 = lambda x: x['YOB']

print(get_year_of_birth__v1(data[0]),
      get_year_of_birth__v2(data[0]))


# > Under the hood, these are not equivalent. In particular, the lambda function has no _name_. When debugging and tracing through a program, it may be harder to know where you are.

# In[16]:


def a_girl_has_a_name():
    return None
    
print(a_girl_has_a_name.__name__, ':', type(a_girl_has_a_name))


# In[17]:


a_girl_has_no_name = lambda: None

print(a_girl_has_no_name.__name__, ':', type(a_girl_has_no_name))


# **Default arguments.** The `sorted` example was a function that took an _optional_ parameter, which allowed you to customize the key. Your functions can have a similar feature if you supply _default_ arguments.

# In[18]:


# (base*factor + offset)**power


# In[19]:


def mul_add_pow(base, factor=1, offset=0, power=1):
    """`mul_add_pow(b, f, o, p)`: returns `(b*f + o)**p`."""
    return (base*factor + offset)**power

print(mul_add_pow(5))              # (5*1 + 0)**1
print(mul_add_pow(5, 2, 3))        # (5*2 + 3)**1
print(mul_add_pow(5, 2, power=2))  # (5*2 + 0)**2
print(mul_add_pow(5, 2, 4, -1))    # (5*2 + 4)**(-1) ~ 0.07
print(mul_add_pow(5, power=-1, factor=2, offset=4))  # Same as above; named args in any order


# > Order of arguments: Optional arguments must appear _after_ required arguments. At the call-site, unnamed optional arguments are assigned in the order of the definition; named arguments must appear after unnamed arguments, but may appear in any order.

# **Flexible arguments.** Per the note above, you can think of the arguments to a function as being two groups: _required arguments_, which do not have names, and _optional arguments_, which are named. Indeed, you can write functions where you do not know these in advance but leave placeholders for them:
# 
# ```python
#     def func(*req, **opt):
#         ...
# ```
# 
# These are available in the body of the function as a tuple (`req`) and dictionary (`opt`), as the next example illustrates.

# In[20]:


def catch_all(*req, **opt):
    print("req =", req)
    print("opt = ", opt)
    return list(req) + list(opt.values())


# In[21]:


catch_all(1, 2, 3, a=4, b=5)


# In[22]:


catch_all('a', keyword=2)


# In[23]:


inputs = (1, 2, 3)
keywords = {'pi': 3.14}
catch_all(*inputs, **keywords)


# **Variable scoping.** Like most other languages, there is a notion of the _scope_ of a variable, which refers to what parts of the program is a variable name visible.

# In[24]:


def foo(y):
    print(y + z_outside)  # `z` must refer to a global variable


# In[25]:


z_outside = 2
foo(-3)


# In[26]:


def bar(x):  # "Hides" any global `x`
    x = x**3
    print(x)

x = 5
bar(2)
print(x)


# In[28]:


def baz(x):  # "Hides" any global `x`
    global x_global
    x_global = x**3
    print(x)
    
x_global = 5
baz(2)
print(x_global)


# **Modifying arguments.** If an argument is a mutable type, the function can change it!

# In[29]:


s0 = 3**40
s1 = s0
s0 /= 3**40
print(s0, s1)


# In[30]:


def add_nothing(s):
    assert isinstance(s, int) # recall: `int` is immutable
    s += 3
    print('add_nothing:', s)
    
s0 = 5
add_nothing(s0)
print(s0)


# In[31]:


def add_abcs(s):
    assert isinstance(s, list) # recall: `list` is mutable
    s += ['a', 'b', 'c']
    print('add_abcs:', s)    
    
s1 = [1, 2, 3]
add_abcs(s1)
print(s1)


# **Nesting functions.** You can also enclose a function definition within a function.

# In[32]:


def sort_dict_by_key(d, k):
    def get_key(x):
        return x


# ## Conditionals
# 
# Your basic `if-then-else` statement. `:D`

# In[33]:


x = -15   # Try `float('nan')`, `float('inf')`

if x == 0:
    print(x, "is zero")
elif x > 0:
    print(x, "is positive")
elif x < 0:
    print(x, "is negative")
else:
    print(x, "is unlike anything I've ever seen...")


# For short conditionals, there is also an "inline" version (`if`-`else` only, though these can be nested):

# In[34]:


a, b = 5, 10
msg = 'lt' if a < b else 'geq'
print(msg)


# In[35]:


a, b = 10, 10
msg = 'lt' if a < b else ('gt' if a > b else 'eq')
print(msg)


# ## Loops
# 
# There are two main types of loops: `for` loops and `while` loops.

# **`for` loops.**

# In[36]:


for N in [2, 3, 5, 7]:
    print(N, end=' ') # print all on same line


# **Other common iterators.** The subexpression to the right of the `in`, above, should be an _iterator_, which is a special type of object that produces a sequence of values. Indeed, what you see above is actually a shortcut for the following:

# In[37]:


I = iter([2, 3, 5, 7])
print(I, "\n")

for N in I:
    print(N, end=' ') # print all on same line


# In[38]:


print(range(10), "\n")

for i in range(10):
    print(i, end=' ')


# **Aside:** An iterator can be converted into an argument tuple using the `*i` syntax you saw above under the section on functions.

# In[39]:


print(*range(5))  # same as print(0, 1, 2, 3, 4)


# **`enumerate(x)`**: an iterator that produces the elements of `x` along with a "running count" that starts at 0.

# In[40]:


print(*enumerate('alsdfkj'))


# In[41]:


for i, s in enumerate('alsdfkj'):
    print(f'{i}: {s}')


# **`zip`**: an iterator that produces tuples of elements taken from its input iterators.

# In[42]:


for a, b in zip(range(3), ['a', 'b', 'c']):
    print(a, "=>", b)


# > This example suggests that a possible implementation of `enumerate(x)` is
# >
# > ```python
# >     def enumerate(x):
# >         return zip(range(len(x)), x)
# > ```

# **`map`**: an iterator that first applies a given function to each value.

# In[43]:


# find the first 10 square numbers
square = lambda x: x ** 2
for val in map(square, range(10)):
    print(val, end=' ')


# **`filter`**: an iterator that only yields values for which a given predicate function evalutes to `True`.

# In[44]:


# find values up to 10 for which x % 2 is zero
is_even = lambda x: x % 2 == 0
print(is_even(4), is_even(7), "\n")

for val in filter(is_even, range(10)):
    print(val, end=' ')


# **`itertools` and generators**. There are many other interesting iterators; see the [`itertools`](https://docs.python.org/3/library/itertools.html) module for a bunch more, as well as the [Python Functional Programming How-To](https://docs.python.org/3/howto/functional.html) for how to create your own iterators.
# 
# Here is one example from `itertools`: producing combinations of a set.

# In[45]:


zoo = {'cat', 'dog', 'emu', 'zebra'}  # a set of animals

from itertools import combinations    # Try also: `permutations`
for x in combinations(zoo, 3):      
    print(x)


# **Set iterators.**

# In[46]:


for s in {1, 2, 3}:
    print(s, end=' ')


# **Iterating over dictionaries.** By default, using a dictionary as the iterator will yield keys. To get values or key-value pairs, use `.values()` and `.items()`, respectively.

# In[47]:


D = {k: v for v, k in enumerate('abcdef')}
print(D)


# In[48]:


for k in D:             # or, `for k in D.keys(): ...`
    print(k, end=' ')


# In[49]:


for v in D.values():
    print(v, end=' ')


# In[50]:


print("==> Version 1:")
for p in D.items(): # (key, value) pairs
    print(p, end=' ')
    
print("\n==> Version 2:")
for k, v in D.items(): # Unpack the pairs into `k` and `v` variables
    print(f'{k}:{v}', end=' ')


# **`while` loops.**

# In[51]:


i = 0
while i < 10:
    print(i, end=' ')
    i += 1


# **`break` and `continue`**.

# In[52]:


# Print odd integers in [0, 20)
for n in range(20):
    if n % 2 == 0:
        continue
    print(n, end=' ')


# In[53]:


# Print Fibonacci sequence for values <= 100
a, b = 0, 1
amax = 100
L = []

while True:
    (a, b) = (b, a + b)
    if a > amax:
        break
    L.append(a)

print(L)


# In[ ]:





# In[ ]:




