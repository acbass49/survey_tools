# survey_tools

This is a simple python package for survey research analysis containing convenient functions for weighted crosstabs, recoding, and weighting.

# Install

You can install survey_tools with `pip`:

```
python -m pip install survey_tools
```

# Assorted Python Functions for Survey Research

`tabs` - 1, 2, and 3 way tabs. Make them weighted or unweighted. Have them include NAs or not. Have them be counts or normalized by row, column, or cell.

```python
tabs(data, 'var_name', display='column', wts='weight_var', dropna=False)
```

`rake_weight` - weight a survey to specified targets using the raking method.

```python
rake_weight(data, df_of_proportions_to_weight_to)
```

`recode` (Similar to car::recode in R) - Select a variable and create a simple recoding string to easily recode variables.

```python
recode(data, 'var_name', 'lo:5=1;6:10=2;11:hi=3;NaN=NaN')
```

`get_names` function - Use regex to easily select names in a given pandas.DataFrame

```python
get_names(data, r'^[Yy]ear.+')
```

# Future Add-ons
1. Add CI function
2. Stacking survey
3. Add other weighting functions like matching, propensity weighting, or multiple combinations of these and raking.

link I was working on: https://packaging.python.org/en/latest/tutorials/packaging-projects/  
link to publish releases: https://medium.com/@blackary/publishing-a-python-package-from-github-to-pypi-in-2024-a6fb8635d45d
