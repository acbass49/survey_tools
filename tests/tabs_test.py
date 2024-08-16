# Title : Tests for recode function
# Author: Alex Bass
# Date : 22 August 2023

import pytest
import pandas as pd
import numpy as np

from survey_tools import tabs

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

@pytest.fixture
def test_data_short_na():
    exp = pd.DataFrame({
        "a" : [1,np.nan,1,2],
        "b" : [1,2,3,4],
        "c" : ['a','a','b','b'],
        'wts' : [0.1,0.1,0.1,2]
    })
    return exp

def _series_is_equal(s1, s2):
    return all(s1 == s2.to_list())

def test_1_way_unweighted_count(test_data_all):
    assert _series_is_equal(tabs(test_data_all, "a"), pd.Series([2,2,2,2,2]))

def test_1_way_weighted_count(test_data_all):
    assert _series_is_equal(tabs(test_data_all, "a", wts = "wts"), pd.Series([0.2,0.2,0.2,0.2,2.1]))

def test_1_way_unweighted_column(test_data_all):
    assert _series_is_equal(
        tabs(test_data_all, "a", display="column"), 
        pd.Series([20,20,20,20,20])
    )

def test_1_way_weighted_column(test_data_all):
    assert \
        tabs(test_data_all, "a", display="column", wts="wts").round(1).to_list() == \
        [6.9,6.9,6.9,6.9,72.4]

def test_1_way_unweighted_cell(test_data_all):
    assert _series_is_equal(
        tabs(test_data_all, "a", display="cell"), 
        pd.Series([20,20,20,20,20])
    )

def test_1_way_weighted_cell(test_data_all):
    assert \
        tabs(test_data_all, "a", display="cell", wts="wts").round(1).to_list() == \
        [6.9,6.9,6.9,6.9,72.4]

def test_1_way_weighted_cell_na(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="cell", wts="wts").round(1).to_list() == \
        [7.4,7.4,7.4,77.8]

def test_1_way_weighted_cell_w_na(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="cell", wts="wts", dropna=False).round(1).to_list() == \
        [6.9,6.9,6.9,72.4,6.9]

def test_1_way_weighted_column_w_na(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="column", wts="wts",dropna=False).round(1).to_list() == \
        [6.9,6.9,6.9,72.4,6.9]

def test_1_way_weighted_column_na_true(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="column", wts="wts").round(1).to_list() == \
        [7.4,7.4,7.4,77.8]

def test_1_way_weighted_cell_na_true(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="cell", wts="wts").round(1).to_list() ==\
        [7.4,7.4,7.4,77.8]

def test_1_way_unweighted_cell_na_true(test_data_NA):
    assert \
        tabs(test_data_NA, "a", display="cell", dropna=False).round(1).to_list() == \
        [20.0,20.0,20.0,20.0,20.0]

def test_2_way_unweighted_col(test_data_short_na):
    res = tabs(test_data_short_na, "a", "b", display="column")
    truth = pd.DataFrame({
        1:[50.0,0.0],
        2:[0.0,0.0],
        3:[50.0,0.0],
        4:[0.0,100.0],
    })
    truth.index = [1.0,2.0]
    assert all(res == truth)

def test_2_way_unweighted_cell(test_data_short_na):
    res = tabs(test_data_short_na, "a", "b", display="cell")
    truth = pd.DataFrame({
        1:[33.3,0.0],
        2:[0.0,0.0],
        3:[33.3,0.0],
        4:[0.0,33.3],
    })
    truth.index = [1.0,2.0]
    assert all(res == truth)

def test_2_way_unweighted_row(test_data_short_na):
    res = tabs(test_data_short_na, "a", "b", display="row")
    truth = pd.DataFrame({
        1:[100.0,0.0],
        2:[0.0,0.0],
        3:[100.0,0.0],
        4:[0.0,100.0],
    })
    truth.index = [1.0,2.0]
    assert all(res == truth)
