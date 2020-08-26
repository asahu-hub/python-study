
# coding: utf-8

# | &nbsp; | &nbsp; | &nbsp; |
# |--------|--------|--------|
# | [Return to Index Notebook](./index.ipynb) | [View on GitHub](https://github.com/hpcgarage/prymer) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hpcgarage/prymer/blob/master/002-collections.ipynb) |

# # Basic collections
# 
# Key concepts:
# * Tuples
# * Lists
# * Variables are references!
# * Sets
# * Dictionaries
# * Nesting data structures
# 
# For more detailed information, see Vanderplas or the [Python documentation on basic data structures](https://docs.python.org/3/tutorial/datastructures.html).

# Although you can do a lot with operations on scalar values, programs are more compact and efficient when you can operate on collections of values.

# ## Tuples ##
# 
# A tuple is a read-only ("immutable") fixed-length sequence of values.

# In[1]:


z = ("a", "pair")
print(z)
print("element 0:", z[0], "\nelement 1:", z[1]) # Random access of values


# In[2]:


("4-tuple", 4, 1.2, False) # Mix values


# "Immutability" means you cannot modify the length of the tuple nor change its values.

# In[4]:


# z[1] = "triple?"   # Uncommenting and running this statement will produce an error


# **Aside: mutability.** Mutability is an important concept in Python, as certain critical operations may only be applied to mutable objects. A good discussion of mutability appears [here](https://medium.com/@meghamohan/mutable-and-immutable-side-of-python-c2145cf72747). You can think of the primitive built-in types––`int`, `float`, `bool`, `str`––as **immutable**. Tuples are, as noted above, also immutable, although it may hold mutable objects. Let's keep going and come back to this point later, after we meet a mutable object: the Python list (`list`).

# ## Lists ##
# 
# A list of values is a sequence, and is similar to arrays in other languages. It provides fast random access (`x[i]`, using zero-based indices) and fast appends, but slow arbitrary insertions and search. Individual values (elements) of a list need not have the same type. 

# In[5]:


x = [1, 2, 3]
y = [4, "xo", 6.7, True]
z = x + y
print("Concat lists: {} + {} = {} (total length is {})".format(x, y, z, len(z)))


# In[6]:


print(z[2], z[4], z[6])
print(z[2::2]) # slice (and later, dice)
print(z[::-1])


# **List constructor.** The list type, denoted by `list` in Python, can also be used to construct an empty list or a list object from another collection type. A pair of empty square brackets is synonymous.

# In[7]:


empty_list = []
print(empty_list)

another_empty_list = list()
print(another_empty_list)

list_from_string = list('abcdefg')
print(list_from_string)


# **Modifying lists.**

# In[8]:


# Make a change
print(z)
z[2] = -(z[2] + z[3])
print(z)

# Undo the change
z[2] *= -1
z[2] -= z[3]
print(z)


# In[9]:


z.append("one more value...")  # Methods, some of which involve in-place modifications
print(z)


# In[10]:


print(z)
z.reverse()
print(z)
z.reverse() # Undo
print(z)


# **List comprehensions.** Use this compact construct to build up lists of values.

# In[11]:


c = [2*xi for xi in x+y] # "double" every element
print(c)


# Additionally, there is a conditional clause for filtering comprehension values.

# In[12]:


g = [i for i in range(10)] # 0 .. 9

from random import shuffle
shuffle(g) # permute randomly

print('g =', g)

# Select only even values
h = [i for i in g if (i % 2) == 0]
print('h = even(g) =', h)


# **Zipper iterations.** Use `zip(a, b, ...)` to "merge" two or more collections. Conceptually, `zip()` produces tuples of corresponding elements from each input collection.

# In[13]:


# "Merge" forward and reverse lists
[(f, b) for f, b in zip(c, c[::-1])]


# **Performance: append vs. insert.** Appending values to a list is fast compared to arbitrary insertions.

# In[14]:


# Insert 100,000 values at the front of the list (index=0)
c0 = []
t_insert = get_ipython().magic("timeit -o -n1000 -r100 c0.insert(0, 'a')")


# In[15]:


# Append 100,000 values at the end of the list
c1 = []
t_append = get_ipython().magic("timeit -o -n1000 -r100 c1.append('a')")


# In[16]:


# Verify that the outputs of the above are the same
assert all([a == b for a, b in zip(c0, c1)]), "Answers differed?"

# Report the ratio of execution times
print("==> (insert time) / (append time) for 100,000 ops: ~ {:.1f}x".format(t_insert.average / t_append.average))


# In[17]:


# Demonstrate scaling: Same experiment as above, but triple the ops
c0 = []
t_insert = get_ipython().magic("timeit -o -n3000 -r100 c0.insert(0, 'a')")

# Append the same 100,000 values at the end of the list
c1 = []
t_append = get_ipython().magic("timeit -o -n3000 -r100 c1.append('a')")

# Verify that the outputs of the above are the same
assert all([a == b for a, b in zip(c0, c1)]), "Answers differed?"

# Report the ratio of execution times
print("\n==> (insert time) / (append time) for 300,000 ops: ~ {:.1f}x".format(t_insert.average / t_append.average))


# **Performance, part 2: search.** Simple searches, which can be performed using the membership-test operator, `in`, can be slow.

# In[18]:


long_list = list(range(100000))
shuffle(long_list)

# Inspect first and last five elements:
print('long_list =', long_list[:5], '...', long_list[-5:], '  (all values are unique)')
first_elem = long_list[0]
last_elem = long_list[-1]
print('\n{} in long_list == {}'.format(first_elem, first_elem in long_list))
print('{} in long_list == {}'.format(last_elem, last_elem in long_list))


# In[19]:


print('\nTimes to search a list of {} values (all unique):'.format(len(long_list)))
t_first = get_ipython().magic('timeit -o first_elem in long_list')
t_last = get_ipython().magic('timeit -o last_elem in long_list')

print('\n==> Ratio: ~ {:.1f}x'.format(t_last.average / t_first.average))


# ## Aside: Variables are references
# 
# One subtlety about variables is that you should always think of them as references or "pointers" to the underlying object or value. Example:

# In[20]:


x = [1, 2, 3]
y = x
x.append(4)
print(x)
print(y)


# Observe that `y` and `x` "point" to the same object. Here is a nice visualization of this code and concept: [Python Tutor](http://pythontutor.com/visualize.html#code=x%20%3D%20%5B1,%202,%203%5D%0Ay%20%3D%20x%0Ax.append%284%29%0Aprint%28x%29%0Aprint%28y%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=py3anaconda&rawInputLstJSON=%5B%5D&textReferences=false).

# The Python `is` operator can be used to test whether two references are identical, and `.copy()` can be used to clone a list to be distinct references.

# In[21]:


print(x is y)
print(x is not y) # By the way...


# In[22]:


z = y.copy()
print(x is z)

x.append(len(x)+1)
print(x, z)


# Try out the `.copy()` code in [Python tutor](http://pythontutor.com/visualize.html#code=x%20%3D%20%5B1,%202,%203%5D%0Ay%20%3D%20x.copy%28%29%0Ax.append%284%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=py3anaconda&rawInputLstJSON=%5B%5D&textReferences=false).

# ## Sets
# 
# A Python set is like a mathematical set: all values are unique and immutable (more later). Therefore, the underlying implementation is free to organize values in clever ways so that search or membership-test becomes much faster than with lists.
# 
# > A set is itself mutable, even though its values must be immutable. If you require an immutable set, use [`frozenset`](https://docs.python.org/3/library/stdtypes.html#frozenset).

# In[23]:


A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}
print(A, '&', B, '==', A & B) # intersection
print(A, '|', B, '==', A | B) # union
print(A, '-', B, '==', A - B) # difference
print(A, '<', B, '==', A < B) # proper subset (use `<=` for subset, `>` or `>=` for superset, or '==' for equality)


# In[24]:


A.add(8)
print(A)

A.update({-1, -2, -3})
print(A)


# Set values are not necessarily unordered in the event you iterate over them:

# In[25]:


[a for a in A]


# Values are not limited to integers and, like lists, elements may mix types. **However, the elements must be immutable**. Recall that the primitive built-in types (e.g., `int`, `float`, `str`, `bool`) are immutable, as is `tuple`. However, a `list` is mutable, and therefore cannot be a set element.

# In[26]:


E = {'cat', 1.61803398875, ('a', 'b', 'c')}
# E.add([1, 2, 3]) # Error!
print(E)


# > There is a subtlety -- although `tuple` is immutable, you can construct a `tuple` that contains mutable values, e.g., the triple `(1, 2, [3, 4, 5])` where the third element is a `list`, which is mutable. This tuple would, therefore, be an invalid set element.

# In[27]:


C = {1, 2, 3.14159, bin(1387)}
print(C)


# **Performance of membership tests.**

# In[28]:


big_set = set(long_list) # Convert list to set

print('{} in big_set == {}'.format(first_elem, first_elem in big_set))
print('{} in big_set == {}'.format(last_elem, last_elem in big_set))


# In[29]:


print('\nTimes to search a set of {} values (all unique):'.format(len(big_set)))
t_first = get_ipython().magic('timeit -o first_elem in big_set')
t_last = get_ipython().magic('timeit -o last_elem in big_set')

print('\n==> Ratio: ~ {:.1f}x'.format(t_last.average / t_first.average))


# **Another example.** Let's use sets, whose values will be unique, to count the number of unique values in a list.

# In[30]:


# http://socialgoodipsum.com/#/
social_good_ipsum = "Agile bandwidth; greenwashing citizen-centered; scale and impact shared value theory of change mass incarceration. Entrepreneur entrepreneur benefit corporation think tank effective her body her rights her body her rights strengthening infrastructure. Collective impact, her body her rights innovation thought provoking social enterprise boots on the ground. Radical black lives matter academic, our work energize granular inclusion expose the truth. Collective impact collective impact LGBTQ+ effective we must stand up. Collaborative cities; inspire, social intrapreneurship outcomes society impact bandwidth. Mass incarceration ecosystem problem-solvers, communities best practices social return on investment and synergy synergy. Shared value, equal opportunity; social innovation segmentation vibrant. Agile problem-solvers progress paradigm B-corp. State of play compelling, compelling; social innovation, effective systems thinking parse cultivate preliminary thinking. Global, triple bottom line; replicable, low-hanging fruit society collective impact society cultivate boots on the ground. Movements entrepreneur targeted, segmentation, game-changer. Empower communities, save the world thought leadership mobilize social entrepreneurship systems thinking impact investing. Correlation policymaker movements inspirational, empower communities, but, B-corp emerging. Social entrepreneurship, ideate scalable, philanthropy then thought leader. Save the world technology unprecedented challenge milestones challenges and opportunities sustainable a. Rubric collaborative consumption targeted, empathetic emerging; collaborative cities leverage. Corporate social responsibility inspire social return on investment cultivate shared vocabulary efficient commitment. State of play benefit corporation, theory of change strategy gender rights catalyze. Effective, change-makers engaging co-creation; circular think tank synergy we must stand up improve the world. Natural resources milestones the scale and impact silo compassion. Then, resilient, shared unit of analysis inspiring the inspire. Social return on investment change-makers strategize, co-create change-makers scale and impact issue outcomes overcome injustice. Correlation, social entrepreneurship shared value, social enterprise blended value LGBTQ+ strategy. Indicators, catalyze shared value inclusion; initiative unprecedented challenge and. Collaborative consumption, to, revolutionary ecosystem thought leader benefit corporation replicable engaging. Initiative gender rights collective."
alpha_only = ''.join([c for c in social_good_ipsum if c.isalpha() or c.isspace()])
tokens = alpha_only.split()
print(f"The original string has {len(tokens)} tokens.")

unique_tokens = set(tokens)
print(f"Of these, {len(unique_tokens)} are unique.")


# ## Dictionaries
# 
# A dictionary is a map of unique keys to values. Like set elements, keys must be unique.

# In[31]:


d = {'pi_approx': 3.14159265359,
     'e_approx': 2.71828182846,
     'c_approx': '299,792,458 m/s',
     'h_bar_approx': '6.62607004e-34 m^2 kg / s'}
print('d =', d)


# The keys in the above example all have the same type, but need not. For example, let's add some integer keys.

# In[32]:


d[0] = 0.0
d[1] = 1.0
d[-1] = -1.0
d['minus_one'] = d[-1]
print(len(d), 'key-value pairs:', d)


# Deletion of keys:

# In[33]:


d.pop('minus_one')
print(len(d), 'key-value pairs:', d)


# In[ ]:


print(d['c_approx'])
print(d.keys())
print(d.values())
print(d.items()) # (key, value) pairs


# Key-membership testing:

# In[34]:


print('h_bar_approx' in d)
print('planck constant' in d)


# Referencing missing keys will produce an error:

# In[36]:


# d['planck constant']   # Uncomment to see an error


# Example: Build an English-to-French dictionary.

# In[37]:


english = "one two three four five six seven eight nine ten".split()
print(english)

french = "un deux trois quatre cinq six sept huit neuf dix".split()
print(french)


# In[38]:


e2f = {k: v for k, v in zip(english, french)} # Dictionary comprehension
print(e2f['three'])


# **Performance of key-membership testing.** The time to check if a key is present should be comparable to set-membership. That fact would imply that looking up a value in a dictionary will take about the same amount of time.

# In[39]:


big_dict = {v: True for v in long_list} # Convert list to dictionary of (value, True) pairs

print('{} in big_dict == {}'.format(first_elem, first_elem in big_dict))
print('{} in big_dict == {}'.format(last_elem, last_elem in big_dict))

print('\nTimes to search a dictionary of {} keys:'.format(len(big_dict)))
t_first = get_ipython().magic('timeit -o first_elem in big_dict')
t_last = get_ipython().magic('timeit -o last_elem in big_dict')

print('\n==> Ratio: ~ {:.1f}x'.format(t_last.average / t_first.average))


# ## Aside: Nesting
# 
# Since lists and dictionaries can contain values (but not keys, in the case of dictionaries) of arbitrary types, you can nest data structures.

# **Example: Dictionary of dictionaries.**

# In[40]:


numbers = {i+1: {'english': english[i],
                 'french': french[i]
                } for i in range(10)
          }

print('5 is "{english}" in English and "{french}" in French.'.format(english=numbers[5]['english'],
                                                                     french=numbers[5]['french']))


# **Aside within an aside: argument unpacking with dictionaries.** One trick that a dictionary enables is called [argument unpacking](https://docs.python.org/3/tutorial/controlflow.html#tut-unpacking-arguments). That is the substituion of keys for named arguments.
# 
# For example, look at the print statement above: the `.format(...)` subexpression requires named arguments. If you have any dictionary with the same keys, you can substitute those using the `**` operator:

# In[41]:


numbers


# In[42]:


six = numbers[6]
print(f"Dictionary `six` is {six}.")

print('\n==> 6 is "{english}" in English and "{french}" in French.'.format(**six))


# In[ ]:





# In[ ]:




