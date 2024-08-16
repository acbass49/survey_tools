import pytest
import pandas as pd
import numpy as np

from survey_tools import recode

@pytest.fixture
def data():
    exp = pd.DataFrame({
        'a':[1,1,2,2,2,2,2,2,2,2],
        'b':[np.nan,2,2,np.nan,2,2,2,2,2,2],
        'c':['a','a','c','c','a','c','a','c','a','a']
    })
    return exp


def test_simple_recode(data):
    assert all(recode(data, 'a', '1=2').eq(2))
    
def test_recoding_NaN(data):
    assert all(recode(data, 'b', 'NaN=2').eq(2))
    
def test_recoding_strings(data):
    assert all(recode(data, 'c', '"a"="c"').eq('c'))

def test_recoding_all_NaN(data):
    assert all(recode(data, 'b', '2=NaN').isna())
