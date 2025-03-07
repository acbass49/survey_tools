# Load in Packages
from survey_tools import tabs, rake_weight, recode, get_names, make_interaction
import pandas as pd
import numpy as np
import os

# pip install -e .

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

tabs(data, 'religpew', 'genderXnewsint_rc', display="column")

def make_tabs(data:pd.DataFrame, vars:list[str], demos:list[str], display:str = "column", wts:str = None, to_excel: str = None):
    '''
    Create crosstabs of variables in vars by demographics in demos
    
    Required Arguments:
        data: `pd.DataFrame` - the data to be used
        vars: `list[str]` - the variables to be crosstabbed
        demos: `list[str]` - the demographics to be crosstabbed by
        display: `str` - how to display the crosstabs, either `column` or `count`. Count will give Nsize in each group. Default is column.
    
    Keyword Arguments:
        wts: `str` - the name of the weights column in the data
        to_excel: `str` - the name of the excel file to save the output to
        
    Returns:
        New `pandas.DataFrame` of tabulation given parameter specifications
    '''
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(vars, list), "vars must be a list of strings"
    assert isinstance(demos, list), "demos must be a list of strings"
    assert isinstance(display, str), "display must be a string"
    assert display in ['column', 'count'], "display must be either 'column' or 'count'"
    assert wts is None or isinstance(wts, str), "wts must be a string"
    assert to_excel is None or isinstance(to_excel, str), "to_excel must be a string"
    #check if variables are in data
    for var in vars:
        assert var in data.columns, f"{var} not in data"
    #check if demographics are in data
    for demo in demos:
        assert demo in data.columns, f"{demo} not in data"
    #check if weights are in data
    if wts:
        assert wts in data.columns, f"{wts} not in data"
    
    #create a list of crosstabs
    tab_s = []
    tabs_to_excel = []
    for var in vars:
        var_level_list = []
        for demo in demos:
            if wts:
                tab = tabs(data, var, demo, wts = wts, display = display)
            else:
                tab = tabs(data, var, demo, display = display)
            #rename columns with demo name appended
            tab.columns = [f'{demo}: ' + str(col) for col in tab.columns]
            #rename index with var name appended
            tab.index = [f'{var}: ' + str(ind) for ind in tab.index]
            var_level_list.append(tab)
        #concatenate column-wise
        var_tabs = pd.concat(var_level_list, axis = 1)
        if to_excel:
            tabs_to_excel.append(var_tabs)
            if len(vars) > 1:
                blank = pd.DataFrame(index = ['']*4, columns = var_tabs.columns)
                blank = blank.fillna('')
                blank.iloc[3,:] = var_tabs.columns
                tabs_to_excel.append(blank)
        tab_s.append(var_tabs)
    #concatenate row-wise
    final_tab = pd.concat(tab_s, axis = 0)
    #save to excel if specified
    if to_excel:
            if len(tabs_to_excel) == 1:
                final_tab_to_excel = tabs_to_excel[0]
            else:
                final_tab_to_excel = pd.concat(tabs_to_excel, axis = 0)
            final_tab_to_excel.to_excel(to_excel)
    return final_tab


make_tabs(data=data, vars=['religpew'], demos=['genderXnewsint_rc'], to_excel='~/Desktop/test.xlsx')
