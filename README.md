# survey_tools

This is a simple python package for survey research analysis containing convenient functions for weighted crosstabs, recoding, and weighting.

# Assorted Python Functions for Survey Research

1. `tabs` - 1, 2, and 3 way tabs. Make them weighted or unweighted. Have them include NAs or not. Have them be counts or normalized by row, column, or cell.
2. `rake_weight` - weight a survey to specified targets using the raking method.
3. `recode` (Similar to car::recode in R) - Select a variable and create a simple recoding string to easily recode variables.
4. `get_names` function - Use regex to easily select names in a given pandas.DataFrame

# ToDo
1. Build a vignette on personal website and jupyter notebook in the repository. Or perhaps create my own website for the package?

# Future Add-ons
1. Add CI function
2. Stacking survey
3. Add other weighting functions like matching, propensity weighting, or multiple combinations of these and raking.


link I was working on: https://packaging.python.org/en/latest/tutorials/packaging-projects/  
link to publish releases: https://medium.com/@blackary/publishing-a-python-package-from-github-to-pypi-in-2024-a6fb8635d45d
