import pytest
import pandas as pd
import numpy as np

from survey_tools import rake_weight, tabs

@pytest.fixture
def test_survey_data():
    exp = pd.DataFrame({
        'a':[1,1,2,2,2,2,2,2,2,2],
        'b':[1,1,1,1,1,2,2,2,2,2],
        'c':[1,1,1,1,2,2,2,2,2,2],
        'd':[1,1,1,1,1,1,2,2,2,2],
    })
    return exp

@pytest.fixture
def test_weighting_props():
    exp = pd.DataFrame({
        'Names':['c', 'c', 'd', 'd'],
        'Levels':[1,2,1,2],
        'Proportions':[0.5,0.5,0.5,0.5],
    })
    return exp

def test_weights_work(test_survey_data,test_weighting_props):
    data = rake_weight(data=test_survey_data, weighting_df=test_weighting_props)
    assert tabs(data, 'c', wts='weight', display='column').to_list() == \
        tabs(data, 'd', wts='weight', display='column').to_list()