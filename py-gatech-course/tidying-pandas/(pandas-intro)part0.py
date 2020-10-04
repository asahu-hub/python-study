
# coding: utf-8

# # Supplemental notes on Pandas
# 
# The [**pandas** library](https://pandas.pydata.org/) is a Python module for representing what we call "tibbles" in Topic 7. Beyond what you see there, this notebook has additional notes to help you understand how to manipulate objects in Pandas. These notes adapt those found in the recommended text, [Python for Data Analysis (2nd ed.)](http://shop.oreilly.com/product/0636920050896.do), which is written by the createor of pandas, [Wes McKinney](http://wesmckinney.com/).

# **Versions.** The state of pandas is a bit in-flux, so it's important to be flexible and accommodate differences in functionality that might vary by version. The following code shows you how to check what version of pandas you have.

# In[ ]:


import pandas as pd  # Standard idiom for loading pandas

print("=== pandas version: {} ===\n".format(pd.__version__))

import sys
print("=== Python version ===\n{}".format(sys.version))


# The main object that pandas implements is the `DataFrame`, which is essentially a 2-D table. It's an ideal target for holding the tibbles of Topic+Notebook 7, and its design derives in part from [data frame objects in the R language](https://www.rdocumentation.org/packages/base/versions/3.5.1/topics/data.frame).
# 
# In addition to `DataFrame`, another important component of pandas is the `Series`, which is essentially one column of a `DataFrame` object (and, therefore, corresponds to variables and responses in a tibble).

# In[ ]:


from pandas import DataFrame, Series


# # `Series` objects
# 
# A pandas [`Series`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html) object is a column-oriented object that we will use to store a variable of a tibble.

# In[ ]:


obj = Series([-1, 2, -3, 4, -5])
print(f"`obj` has type `{type(obj)}`:\n\n{obj}")


# Observe the common **base type** (`dtype: int64`) and **index** (element numbers).

# Regarding the base type, a `Series` differs from a Python `list` in that the types of its elements are assumed to be the same. Doing so allows many operations on a `Series` to be faster than their counterparts for `list` objects, as in this search example.

# In[ ]:


from random import randint
n_ints = 10000000
max_value = 5*n_ints

print(f"""
Creating random `list` and `Series` objects:
- Length: {n_ints} elements
- Range: [{-max_value}, {max_value}]
""")
a_list = [randint(-max_value, max_value) for _ in range(n_ints)]
a_series = Series(a_list)

print("==> Estimating time to search the `list`:")
t_list_search = get_ipython().magic('timeit -o randint(-max_value, max_value) in a_list')

print("\n==> Estimating time to search the `Series`:")
t_series_search = get_ipython().magic('timeit -o a_series.isin([randint(-max_value, max_value)])')

print(f"\n==> (`list` time) divided by `Series` time is roughly {t_list_search.average / t_series_search.average:.1f}x")


# If you create a `Series` with "mixed types," the `dtype` will become the most generic Python type, `object`. (A deeper understanding of what this fact means requires some knowledge of object-oriented programming, but that won't be necessary for our course.)

# In[ ]:


obj2 = Series([-1, '2', -3, '4', -5])
obj2


# If you want to query the base type, use:

# In[ ]:


print(obj.dtype)
print(obj2.dtype)


# Regarding the index, it provides a convenient way to reference individual elements of the `Series`.

# By default, a `Series` has an index that is akin to `range()` in standard Python, and effectively numbers the entries from 0 to `n-1`, where `n` is the length of the `Series`. A `Series` object also becomes list-like in how you reference its elements.

# In[ ]:


print("obj.index: {}".format(obj.index))
print("range(0, 5): {}".format(range(0, 5)))


# In[ ]:


print("==> obj[2]:\n{}\n".format(obj[2]))
print("==> obj[3]:\n{}\n".format(obj[3]))
print("==> obj[1:3]:\n{}\n".format(obj[1:4]))


# You can also use more complex index objects, like lists of integers and conditional masks.

# In[ ]:


I = [0, 2, 3]
obj[I] # Also: obj[[0, 2, 3]]


# In[ ]:


I_pos = obj > 0
print(type(I_pos), I_pos)


# In[ ]:


print(obj[I_pos])


# However, the index can be a more general structure, which effectively turns a `Series` object into something that is "dictionary-like."

# In[ ]:


obj3 = Series([      1,    -2,       3,     -4,        5,      -6],
              ['alice', 'bob', 'carol', 'dave', 'esther', 'frank'])
obj3

# In[ ]:


print("* obj3['bob']: {}\n".format(obj3['bob']))
print("* obj3['carol']: {}\n".format(obj3['carol']))


# In fact, you can construct a `Series` from a dictionary directly:

# In[ ]:


peeps = {'alice': 1, 'carol': 3, 'esther': 5, 'bob': -2, 'dave': -4, 'frank': -6}
obj4 = Series(peeps)
print(obj4)


# In[ ]:


mujeres = [0, 2, 4] # list of integer offsets
print("* las mujeres of `obj3` at offsets {}:\n{}\n".format(mujeres, obj3[mujeres]))


# In[ ]:


hombres = ['bob', 'dave', 'frank'] # list of index values
print("* hombres, by their names, {}:\n{}".format(hombres, obj3[hombres]))


# In[ ]:


I_neg = obj3 < 0
print(I_neg)


# In[ ]:


print(obj3[I_neg])


# Because of the dictionary-like naming of `Series` elements, you can use the Python `in` operator in the same way you would a dictionary.
# 
# > Note: In the timing experiment comparing `list` search and `Series` search, you may have noticed that the benchmark does not use `in`, but rather, `Series.isin`. Why is that?

# In[ ]:


print('carol' in peeps)
print('carol' in obj3)


# Basic arithmetic works on `Series` as vector-like operations.

# In[ ]:


print(obj3, "\n")
print(obj3 + 5, "\n")
print(obj3 + 5 > 0, "\n")
print((-2.5 * obj3) + (obj3 + 5))


# A `Series` object also supports vector-style operations with automatic alignment based on index values.

# In[ ]:


print(obj3)


# In[ ]:


obj_l = obj3[mujeres]
obj_l


# In[ ]:


obj3 + obj_l


# Observe what happened with undefined elements. If you are familiar with relational databases, this behavior is akin to an _outer-join_. 

# Another useful transformation is the `.apply(fun)` method. It returns a copy of the `Series` where the function `fun` has been applied to each element. For example:

# In[ ]:


abs(-5) # Python built-in function


# In[ ]:


obj3 # Recall


# In[ ]:


obj3.apply(abs)


# In[ ]:


obj3 # Note: `.apply()` returned a copy, so the original is untouched


# A `Series` may be _named_, too.

# In[ ]:


print(obj3.name)


# In[ ]:


obj3.name = 'peep'
obj3


# When we move on to `DataFrame` objects, you'll see why names matter.

# # `DataFrame` objects
# 
# A pandas [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) object is a table whose columns are `Series` objects, all keyed on the same index. It's the perfect container for what we have been referring to as a tibble.

# In[ ]:


cafes = DataFrame({'name': ['east pole', 'chrome yellow', 'brash', 'bar crema', '3heart', 'spiller park pcm'],
                   'zip': [30324, 30312, 30318, 30030, 30306, 30308],
                   'poc': ['jared', 'kelly', 'matt', 'julian', 'nhan', 'dale']})
print("type:", type(cafes))
print(cafes)


# In[ ]:


display(cafes) # Or just `cafes` as the last line of a cell


# The `DataFrame` has named columns, which are stored as an `Index` (more later!):

# In[ ]:


cafes.columns


# Each column is a named `Series`:

# In[ ]:


type(cafes['zip']) # Aha!


# As you might expect, these `Series` objects should all have the same index.

# In[ ]:


cafes.index


# In[ ]:


cafes.index == cafes['zip'].index


# In[ ]:


cafes['zip'].index == cafes['poc'].index


# You may use complex indexing of columns.

# In[ ]:


target_fields = ['zip', 'poc']
cafes[target_fields]


# But slices apply to rows.

# In[ ]:


cafes[1::2]


# The index above is, by default, an integer range.

# In[ ]:


cafes.index


# In[ ]:


cafes2 = cafes[['poc', 'zip']]
cafes2.index = cafes['name']
cafes2.index.name = None
cafes2


# You can access subsets of rows using the `.loc` field and index values:

# In[ ]:


cafes2.loc[['chrome yellow', 'bar crema']]


# Alternatively, you can use integer offsets via the `.iloc` field, which is 0-based.

# In[ ]:


cafes2.iloc[[1, 3]]


# Adding columns is easy. Suppose every cafe has a 4-star rating on Yelp! and a two-dollar-sign cost:

# In[ ]:


cafes2['rating'] = 4.0
cafes2['price'] = '$$'
cafes2


# And vector arithmetic should work on columns as expected.

# In[ ]:


prices_as_ints = cafes2['price'].apply(lambda s: len(s))
prices_as_ints


# In[ ]:


cafes2['value'] = cafes2['rating'] / prices_as_ints
cafes2


# Because the columns are `Series` objects, there is an implicit matching that is happening on the indexes. In the preceding example, it works because all the `Series` objects involved have identical indexes.

# However, the following will not work as intended because referencing rows yields copies.
# 
# For instance, suppose there is a price hike of one more `'$'` for being in the 30306 and 30308 zip codes. (If you are in Atlanta, you may know that these are the zip codes that place you close to, or in, [Ponce City Market](http://poncecitymarket.com/) and the [Eastside Beltline Trail](https://beltline.org/explore-atlanta-beltline-trails/eastside-trail/)!) Let's increase the price there, on a copy of the dataframe, `cafes3`.

# In[ ]:


cafes3 = cafes2.copy()
cafes3


# In[ ]:


is_fancy = cafes3['zip'].isin({30306, 30308})
# Alternative:
#is_fancy = cafes3['zip'].apply(lambda z: z in {30306, 30308})
is_fancy


# In[ ]:


cafes3[is_fancy]


# In[ ]:


# Recall: Legal Python for string concatenation
s = '$$'
s += '$'
print(s)


# In[ ]:


cafes3[is_fancy]['price'] += '$'


# What does that error message mean? Let's see if anything changed.

# In[ ]:


cafes3


# Nope! When you slice horizontally, you get copies of the original data, not references to subsets of the original data. Therefore, we'll need different strategy.
# 
# Observe that the error message suggests a way!

# In[ ]:


cafes3.loc[is_fancy, 'price'] += '$'
cafes3


# **A different approach.** For pedagogical purposes, let's see if we can go about solving this problem in other ways to see what might or might not work.

# In[ ]:


cafes4 = cafes2.copy() # Start over
cafes4


# Based on the earlier discussion, a well-educated first attempt might be to construct a `Series` with a named index, where the index values for fancy neighborhoods have an additional `'$'`, and then use string concatentation.

# In[ ]:


fancy_shops = cafes4.index[is_fancy]
fancy_shops


# In[ ]:


fancy_markup = Series(['$'] * len(fancy_shops), index=fancy_shops)
fancy_markup


# In[ ]:


cafes4['price'] + fancy_markup


# Close! Remember that missing values are treated as `NaN` objects.
# 
# **Exercise**. Develop an alternative scheme.

# In[ ]:


# Preliminary observation:
print("False * '$' == '{}'".format(False * '$'))
print("True * '$' == '{}'".format(True * '$'))


# In[ ]:


cafes4 = cafes2.copy()
cafes4['price'] += Series([x * '$' for x in is_fancy.tolist()], index=is_fancy.index)
cafes4


# **More on `apply()` for `DataFrame` objects.** As with a `Series`, there is a `DataFrame.apply()` procedure. However, it's meaning is a bit more nuanced because a `DataFrame` is generally 2-D rather than 1-D.

# In[ ]:


cafes4.apply(lambda x: type(x)) # What does this do? What does the output tell you?


# A useful parameter is `axis`:

# In[ ]:


cafes4.apply(lambda x: type(x), axis=1) # What does this do? What does the output tell you?


# And just to quickly verify what you get when `axis=1`:

# In[ ]:


cafes4.apply(lambda x: print(x) if x.name == 'east pole' else None, axis=1)


# **Exercise.** Use `DataFrame.apply()` to update the `'value'` column in `cafes4`, which is out of date given the update of the prices.

# In[ ]:


cafes4 # Verify visually that `'value'` is out of date


# In[ ]:


def calc_value(row):
    return row['rating'] / len(row['price'])

cafes4['value'] = cafes4.apply(calc_value, axis=1)
cafes4


# Another useful operation is gluing `DataFrame` objects together. There are several helpful operations covered in Notebook 7; one not mentioned there, but useful in one of its exercises, is `.concat()`.

# In[ ]:


# Split based on price
is_cheap = cafes4['price'] <= '$$'
cafes_cheap = cafes4[is_cheap]
cafes_pricey = cafes4[~is_cheap]

display(cafes_cheap)
display(cafes_pricey)


# In[ ]:


# Never mind; recombine
pd.concat([cafes_cheap, cafes_pricey])


# ## More on index objects
# 
# A pandas [`Index`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html), used by `Series` and `DataFrame`, is "list-like." It has a number of useful operations, including set-like operations (e.g., testing for membership, intersection, union, difference):

# In[ ]:


from pandas import Index


# In[ ]:


cafes4.index


# In[ ]:


cafes4.index.isin(['brash', '3heart'])


# In[ ]:


cafes4.index.union(['chattahoochee'])


# In[ ]:


cafes4.index.difference(['chattahoochee', 'starbucks', 'bar crema'])


# If you need to change the index of a `DataFrame`, here is one way to do it.

# In[ ]:


cafes5 = cafes4.reindex(Index(['3heart', 'east pole', 'brash', 'starbucks']))

display(cafes4)
display(cafes5)


# Observe that this reindexing operation matches the supplied index values against the existing ones. (What happens to index values you leave out? What happens with new index values?)

# Another useful operation is dropping the index (and replacing it with the default, integers).

# In[ ]:


cafes6 = cafes4.reset_index(drop=True)
cafes6['name'] = cafes4.index
cafes6


# **Fin!** That's the end of these notes. With this information as background, you should be able to get through Notebook 7.
