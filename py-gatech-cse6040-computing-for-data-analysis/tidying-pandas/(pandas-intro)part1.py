
# coding: utf-8

# # Pandas walk-through: Federal Election Commission dataset
# 
# This walk-through is adapted from Chapter 14.5 of Wes McKinney's book, [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) (3rd edition).

# In[ ]:


get_ipython().magic('matplotlib inline')
import pandas as pd
print(pd.__version__)


# # Setup
# 
# The following code cell imports a local Python module (stored in `cse6040utils.py`) and uses one of its utility functions to open the sample dataset.
# 
# > Note: Due to the size of the data file, we are not making it available for download. You will need to run this notebook on Vocareum.

# In[ ]:


from cse6040utils import download_dataset
local_data = download_dataset({'P00000001-ALL.csv': '31df639d0b5dbd3b6d755f91d6bf6fb4'}, vocareum_only=True)


# # Initial exploration and clean-up

# In[ ]:


# Load CSV file
fecdata = pd.read_csv(local_data['P00000001-ALL.csv'])
fecdata.head()


# In[ ]:


fecdata.head()


# In[ ]:


fecdata.info()


# Get a random sample:

# In[ ]:


fecdata.sample(5)


# Summarize numerical data (`.describe()`):

# In[ ]:


fecdata.describe()


# Get a list of the unique candidates (`unique_candidates`):

# In[ ]:


unique_candidates = fecdata['cand_nm'].unique()
unique_candidates


# Assign party affiliations (they are all Republicans except for Barack Obama):

# In[ ]:


party_affiliations = {name: 'D' if name == 'Obama, Barack' else 'R' for name in unique_candidates}
party_affiliations


# In[ ]:


aff = {name: "D" if name == "Obama, Barack" else "R" for name in unique_candidates}
aff


# In[ ]:


candidate_sample = fecdata['cand_nm'].sample(5)
candidate_sample


# In[ ]:


candidate_sample.map(party_affiliations)


# In[ ]:


fecdata['party'] = fecdata['cand_nm'].map(party_affiliations)


# In[ ]:


fecdata.sample(5)


# # Total contributions by party and candidate

# What was the total amount of contributions (in millions of dollars)?

# In[ ]:


fecdata['contb_receipt_amt'].sum()*1e-6 # millions of dollars


# Which **party** got more individual donations (transactions, not total dollars)?

# In[ ]:


fecdata['party'].value_counts()


# Which party got more total dollars?

# In[ ]:


fecdata.groupby('party')['contb_receipt_amt'].sum()*1e-6


# Filter all the data to include only the two main candidates, Romney and Obama.

# In[ ]:


keep_candidates = {'Obama, Barack', 'Romney, Mitt'}


# In[ ]:


matches = fecdata['cand_nm'].apply(lambda x: x in keep_candidates)
fecdata[matches].shape


# In[ ]:


fecmain = fecdata[fecdata['cand_nm'].isin(keep_candidates)].copy()
print(fecmain['cand_nm'].unique())
display(fecmain.sample(5))
display(fecmain.groupby('cand_nm')['contb_receipt_amt'].sum()*1e-6)


# # Who contributes?

# Get a list of top occupations:

# In[ ]:


len(fecmain['contbr_occupation'].unique())


# In[ ]:


fecmain['contbr_occupation'].value_counts()


# Replace synonyms: (also: `dict.get()`)

# In[ ]:


occ_mapping = {'INFORMATION REQUESTED': 'NOT PROVIDED',
               'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
               'INFORMATION REQUESTED (BEST EFFORTS)': 'NOT PROVIDED',
               'C.E.O.': 'CEO'}


# In[ ]:


fecmain['contbr_occupation'].map(occ_mapping)


# In[ ]:


# .get()!
print(occ_mapping.get('PROFESSOR'))
print(occ_mapping.get('PROFESSOR', 'PROFESSOR'))


# In[ ]:


fecmain['contbr_occupation'] = fecmain['contbr_occupation'].map(lambda x: occ_mapping.get(x, x))


# In[ ]:


fecmain['contbr_occupation']


# Synonymous employer mappings:

# In[ ]:


emp_mapping = occ_mapping.copy()
emp_mapping['SELF'] = 'SELF-EMPLOYED'
emp_mapping['SELF EMPLOYED'] = 'SELF-EMPLOYED'
emp_mapping


# In[ ]:


fecmain['contbr_employer'] = fecmain['contbr_employer'].map(lambda x: emp_mapping.get(x, x))


# In[ ]:


emp_mapping.get('prof','pro')


# Create a "pivot table" that shows occupations as rows and party affiliation as columns, summing the individual contributions.

# In[ ]:


by_occ = fecmain.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
by_occ


# Determine which occupations account for $1 million or more in contributions. Compare the amounts between the two party affiliations. (Bonus: Make a plot to compare these visually.)

# In[ ]:


over_1mil = by_occ[by_occ.sum(axis=1) > 1e6]*1e-6
len(over_1mil)


# In[ ]:


over_1mil


# In[ ]:


sorted_occ = over_1mil.sum(axis=1).sort_values()


# In[ ]:


over_1mil.sum(axis=1).sort_values()


# In[ ]:


over_1mil_sorted = over_1mil.loc[sorted_occ.index]
over_1mil_sorted.plot(kind='barh', stacked=True, figsize=(10, 6));


# # Simple ranking

# Determine largest donors:

# In[ ]:


largest_donors = fecmain['contb_receipt_amt'].nlargest(7)
largest_donors


# In[ ]:


fecmain.loc[largest_donors.index]


# Display largest donors, grouped by candidate:

# In[ ]:


grouped = fecmain.groupby('cand_nm')
grouped['contb_receipt_amt'].nlargest(3)


# In[ ]:


type(grouped)


# `.apply()` for groups:

# In[ ]:


grouped.apply(lambda x: type(x))


# Use `.apply()` to get `DataFrame` objects showing the largest donors, grouped by candidate _and_ occupation:

# In[ ]:


def top_amounts_by_occupation(df, n=5):
    # Fill me in!
    totals = df.groupby('contbr_occupation')['contb_receipt_amt'].sum()
    return totals.nlargest(n)

top_amounts_by_occupation(fecmain)


# In[ ]:


grouped.apply(top_amounts_by_occupation, n=10)


# # Big vs. small donations

# For each of the leading two candidates, did most of their money come from large or small donations?

# In[ ]:


bins = [0] + [10**k for k in range(0, 8)]
bins


# In[ ]:


labels = pd.cut(fecmain['contb_receipt_amt'], bins, right=False)
labels[:5]


# In[ ]:


grouped = fecmain.groupby(['cand_nm', labels])
grouped.size()


# **Fin!**
