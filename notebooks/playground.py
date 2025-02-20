# Load in Packages
from survey_tools import tabs, rake_weight, recode, get_names
import pandas as pd
import numpy as np
import os

link = 'https://csed.byu.edu/00000183-a4c5-d2da-abe3-feed7be30001/2021data'
data = pd.read_stata(link)

tabs(data, 'newsint', dropna=False)
data['newsint'] = data.newsint.cat.codes
tabs(data, 'newsint', dropna=False)
data['newsint_rc'] = recode(data, 'newsint', "0='interested';1:5='not interested'")

tabs(data, 'religpew', dropna=True)
tabs(data, 'religpew', 'newsint_rc',dropna=False)



def make_interaction(data, var1, var2):
    # make both variables category types
    if data[var1].dtype.name != 'category':
        data[var1] = data[var1].astype('category')
    if data[var2].dtype.name != 'category':
        data[var2] = data[var2].astype('category')
    new_cat = _make_interactions_categories(data[var1].cat.categories, data[var2].cat.categories)
    # make a new variable that is the interaction of the two
    data[var1 + 'X' + var2] = data[var1].astype(str) + ' ' + data[var2].astype(str)
    data[var1 + 'X' + var2] = data[var1 + 'X' + var2].astype('category')
    data[var1 + 'X' + var2] = data[var1 + 'X' + var2].cat.set_categories(new_cat)
    #make sure NaNs are still in the data
    if data[var1].isna().any() or data[var2].isna().any():
        data.loc[((data.var1.isna()) | (data.var2.isna())), f'{var1}X{var2}'] = np.nan
    print(f'successfully created {var1}X{var2}')
    return data

def _make_interactions_categories(A, B):
    res = []
    for a in A:
        for b in B:
            res.append(f'{a} {b}')
    return res

get_names(data,'age')

data = make_interaction(data, 'gender', 'newsint_rc')

tabs(data, 'religpew', 'genderXnewsint_rc',dropna=False)