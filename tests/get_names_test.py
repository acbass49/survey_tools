# Title : Tests for recode function
# Author: Alex Bass
# Date : 22 August 2023

import pytest
import pandas as pd
import numpy as np

from survey_tools import get_names

@pytest.fixture
def test_data_names():
    exp = pd.DataFrame({
        "a" : [1,np.nan,1,2],
        "b" : [1,2,3,4],
        "c" : ['a','a','b','b'],
        'wts' : [0.1,0.1,0.1,2],
        "ab" : [1,np.nan,1,2],
        "bas2df" : [1,2,3,4],
        "c " : ['a','a','b','b'],
        'wts12' : [0.1,0.1,0.1,2],
        "aasdf" : [1,np.nan,1,2],
        "bvc sdr" : [1,2,3,4],
        "ccdr" : ['a','a','b','b'],
        'wtsacrne;irg' : [0.1,0.1,0.1,2],
    })
    return exp

def test_no_matches(test_data_names):
    assert isinstance(get_names(test_data_names, "z"),list)
    assert len(get_names(test_data_names, "z")) == 0

def test_simple_match(test_data_names):
    assert get_names(test_data_names, "a") == \
        ['a', 'ab', 'bas2df', 'aasdf', 'wtsacrne;irg']

def test_special_char_match1(test_data_names):
    assert get_names(test_data_names, r"\s") == \
        ['c ', 'bvc sdr']

def test_special_char_match2(test_data_names):
    assert get_names(test_data_names, r"\d") == \
        ['bas2df', 'wts12']
