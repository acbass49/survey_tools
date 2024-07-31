# Title : Tests for recode function
# Author: Alex Bass
# Date : 22 August 2023

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '..')

import survey_tools

@pytest.fixture
def test_data_all():
    exp = pd.DataFrame({
        "a" : [1,2,3,4,5,1,2,3,4,5],
        "b" : [1,2,3,4,9,9,9,3,2,1],
        "c" : [9,8,7,6,5,4,3,2,1,1],
        'wts' : [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,2]
    })
    return exp

@pytest.fixture
def test_data_NA():
    exp = pd.DataFrame({
        "a" : [1,np.nan,3,4,5,1,np.nan,3,4,5],
        "b" : [1,2,3,4,9,9,9,np.nan,2,1],
        "c" : [9,8,7,6,5,4,3,2,1,1],
        'wts' : [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,2]
    })
    return exp

def _series_is_equal(s1, s2):
    return all(s1 == s2.to_list())

def test_1_way_unweighted_count(test_data_all):
    assert _series_is_equal(survey_tools.tabs(test_data_all, "a"), pd.Series([2,2,2,2,2]))

def test_1_way_unweighted_column(test_data_all):
    assert _series_is_equal(
        survey_tools.tabs(test_data_all, "a", display="column"), 
        pd.Series([20,20,20,20,20])
    )

def test_1_way_unweighted_cell(test_data_all):
    assert _series_is_equal(
        survey_tools.tabs(test_data_all, "a", display="cell"), 
        pd.Series([20,20,20,20,20])
    )
