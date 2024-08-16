import pandas as pd
import numpy as np
import functools
import operator
import warnings

def tabs(data:pd.DataFrame, var1:str, var2:str = None, var3:str=None, wts:str = None, display:str = "count", dropna:bool = True):
    '''
    Tabulate your data with 1, 2, and 3 variables. Use weighted counts if desired. 
    Get counts or proportions by cell, row, or column. Drop NaNs or include them in your tabulations.
    
    Required Arguments:
        data: `pandas.DataFrame` object which contains var1, var2, var3, and wts depending on the request
        var1: `str` of 1st variable name in `data` to use for tabulation
    
    Keyword Arguments:
        var2: `str` of 2nd variable name in `data`
        var3: `str` of 3rd variable name in `data`
        wts: `str` of weighting variable name in `data`
        display: takes one of four possible values depending how you want the data summarized: 
            `count`, `cell`, `row`, and `column`. The default is `count`
        dropna: `bool` indicating whether you want to include missing values in the final tabulation
    
    Returns:
        New `pandas.DataFrame` of tabulation given parameter specifications
    '''
    import pandas as pd
    import numpy as np
    assert isinstance(data, pd.DataFrame), \
        f"data should be an instance of `pandas.DataFrame` not {type(data)}"
    assert isinstance(var1, str), f"`var1` should be a `str` not {type(var1)}"
    assert var1 in data.columns
    assert display in ['cell', 'row', 'column', 'count']
    
    if var2 is not None:
        assert isinstance(var2, str), f"`var2` should be a `str` not {type(var2)}"
        assert var2 in data.columns, f"Please check the data, {var2} is not found"
        var2 = data[var2]
    if wts is not None:
        assert isinstance(wts, str), f"`wts` should be a `str` not {type(wts)}"
        assert wts in data.columns, f"Please check the data, {wts} is not found"
        wts = data[wts]
    if var3 is not None:
        assert isinstance(var3, str), f"`var3` should be a `str` not {type(var3)}"
        assert var3 in data.columns, f"Please check the data, {var3} is not found"
        var3 = data[var3]
    
    assert isinstance(dropna, bool), f"`var3` should be a `bool` not {type(dropna)}"
    
    var1 = data[var1]
    
    if var2 is not None and wts is not None and var3 is None:
        type_q = "2way weighted"
    elif wts is not None and var3 is None:
        type_q = "1way weighted"
    elif var2 is not None and var3 is None:
        type_q = "2way unweighted"
    elif wts is None and var2 is None and var3 is None:
        type_q = "1way unweighted"
    elif wts is None and var2 is not None and var3 is not None:
        type_q = "3way unweighted"
    elif var2 is not None and var3 is not None:
        type_q = "3way weighted"
    else:
        raise Exception("Variable was not correctly classified")
        
    
    if type_q == "1way weighted" or type_q == "1way unweighted":
        categories = list(var1.value_counts().index)
        if not dropna:
            categories.append("NA_python")
        response_dict = {}
        for category in categories:
            if category == "NA_python":
                idx = var1.isna()
                if type_q == "1way unweighted":
                    response_dict["NaN"] = idx.sum()
                elif type_q == "1way weighted":
                    response_dict["NaN"] = wts[idx].sum()
            else:
                idx = var1[var1.eq(category)].index
                if type_q == "1way unweighted":
                    response_dict[category] = len(idx)
                elif type_q == "1way weighted":
                    response_dict[category] = wts[idx].sum()
        response = pd.Series(response_dict.values())
        response.index = response_dict.keys()
    elif type_q == "2way weighted" or type_q == "2way unweighted":
        categories1 = list(var1.value_counts().index)
        categories2 = list(var2.value_counts().index)
        if not dropna:
            categories1.append("NaN")
            categories2.append("NaN")
        
        #instantiate matrix
        final_matrix = np.zeros((len(categories1), len(categories2)))
        
        for category1 in categories1:
            row_index = categories1.index(category1)
            for category2 in categories2:
                col_index = categories2.index(category2)
                if category1 == "NaN" and category2 == "NaN":
                    fltr = var1.isna() & var2.isna()
                elif category1 != "NaN" and category2 == "NaN":
                    fltr = var1.eq(category1) & var2.isna()
                elif category1 == "NaN" and category2 != "NaN":
                    fltr = var1.isna() & var2.eq(category2)
                elif category1 != "NaN" and category2 != "NaN":
                    fltr = var1.eq(category1) & var2.eq(category2)
                if any(fltr):
                    if type_q == "2way weighted":
                        val = wts[fltr].sum()
                    elif type_q == "2way unweighted":
                        val = sum(fltr)
                    final_matrix[row_index, col_index] = np.where(pd.isna(val), 0.0, val)
        response = pd.DataFrame(final_matrix)
        response.index = categories1
        response.columns = categories2
    elif type_q == "3way weighted" or type_q == "3way unweighted":
        categories1 = list(var1.value_counts().index)
        categories2 = list(var2.value_counts().index)
        categories3 = list(var3.value_counts().index)
        
        if str(var3.dtype) == "cat":
            if var3.cat.ordered:
                categories3 = list(var3.cat.categories)
        
        if not dropna:
            categories1.append("NaN")
            categories2.append("NaN")
            categories3.append("NaN")
        
        #instantiate matrix
        final_matrix = np.zeros((len(categories1), len(categories2), len(categories3)))
        
        for category1 in categories1:
            row_index = categories1.index(category1)
            for category2 in categories2:
                col_index = categories2.index(category2)
                for category3 in categories3:
                    z_index = categories3.index(category3)
                    func_to_pass = \
                        [np.where(x == "NaN", pd.Series.isna, pd.Series.eq) \
                            for x in [category1,category2,category3]]
                    vars_to_eq = {}
                    for i,func in enumerate(func_to_pass):
                        if func == pd.Series.eq:
                            vars_to_eq[i] = eval(f'var{i+1}').eq(eval(f'category{i+1}'))
                        elif func == pd.Series.isna:
                            vars_to_eq[i] = eval(f'var{i+1}').isna()
                    fltr = vars_to_eq[0] & vars_to_eq[1] & vars_to_eq[2]
                    if any(fltr):
                        if type_q == "3way weighted":
                            val = wts[fltr].sum()
                        elif type_q == "3way unweighted":
                            val = sum(fltr)
                        final_matrix[row_index, col_index, z_index] = np.where(pd.isna(val), 0.0, val)
        
        #take z axis and and append it to X
        final_matrix = final_matrix.reshape((len(categories1), len(categories2) * len(categories3)))
        response = pd.DataFrame(final_matrix)
        response.index = categories1
        response.columns = _get_interactions(categories2, categories3)
    
    else:
        raise Exception("The function did not know how to classify\
            this question as '1way summary', etc. Please read documentation and retry.")
        
    if _is_ordered_category_var(var1):
        categories = list(var1.cat.categories)
        if not dropna:
            categories = categories + ["NaN"]
        response = response.reindex(categories)
    
    if type_q == "2way unweighted" or type_q == "2way weighted":
        if _is_ordered_category_var(var2):
            categories = list(var2.cat.categories)
            if not dropna:
                categories = categories + ["NaN"]
            assert set(categories) == set(response.columns.to_list()), \
                f"there was a problem with the categories of '{var2}'."
            response = response[categories]
    
    if var3 is not None:
        response.columns = \
            pd.MultiIndex.from_tuples([(c.split("|")[0], c.split("|")[1]) for c in response.columns])
    
    if display == "count":
        return response
    if display == "row":
        if isinstance(response, pd.Series):
            raise Exception("There was only one column, so summarizing them row-wise doesn't make sense")
        if var3 is not None: #is a 3way crosstab
            return response.groupby(level=[0], axis = 1) \
                .transform(lambda s: round(s/sum(s),3)*100).fillna(0.0)
        else:
            return response.apply(lambda a: round(a/sum(a),3)*100,axis=1)
    if display == "column":
        if isinstance(response, pd.Series):
            return round(response/response.sum(), 3) * 100
        else:
            return response.apply(lambda a: round(a/sum(a),3)*100).fillna(0)
    if display == "cell":
        if isinstance(response, pd.Series):
            return round(response/sum(response),3) * 100
        else: 
            base = response.sum().sum() #summing rows and columns
            def quick_cell(val):
                return round(val/base, 3)*100
            return response.apply(quick_cell)

def recode(data, var, recode_str):
    '''
    Easily recode a variable inside a `pandas.DataFrame` given the instuctions inside the `recode_str`
    
    Required Arguments:
        data: `pandas.DataFrame` object which contains var
        var: `str` of variable name in `data` to recode
        recode_str: `str` of recoding instructions. For each value to be recoded, first write 
            the original value, then separate by an equal sign, then write the new value 
            (e.g. "5=3"). If there are multiple recodes, separate them by `;` (e.g. "5=3;4=3"). 
            Include quotations in your `recode_str` when recoding string values. You can include 
            intervals separated by a `:` (e.g. "1:5=1;6:10=2"). If you are unsure of the highest or
            lowest number in the interval you may use special key words `hi` and `lo` 
            (e.g. "lo:5=1;6:hi=2"). The other special keyword is `NaN` notifying a missing value 
            (e.g. "NaN=1" or "1:5=NaN"). 
    
    Returns:
        a `pandas.Series` of the newly recoded `var`
    '''
    assert isinstance(data, pd.DataFrame), f"`data` should be a pandas DataFrame not {type(data)}"
    assert isinstance(var, str), f"`var` should be a string not {type(var)}"
    assert var in data.columns, f'`var` is not a column in the DataFrame'
    assert isinstance(recode_str, str), f"`recode_str` should be a string not {type(recode_str)}"
    
    var_to_edit = data[var]
    original_dt = var_to_edit.dtype
    
    #parse the recode string
    recode_dict = _recode_string_to_instructions(recode_str, var_to_edit)
    
    #filter dict to make only relevant changes
    recode_dict = {k:v for k,v in recode_dict.items() if k != v}
    
    to_delete = {}
    for k,v in recode_dict.items():
        if k == "NaN":
            to_delete[k] = v
        if v == "NaN":
            recode_dict[k] = np.nan
    
    for k,v in to_delete.items():
        del recode_dict[k]
        recode_dict[np.nan] = v
    
    exp = data[var].replace(recode_dict)
    
    exp_dt = exp.dtype
    exp_str = [isinstance(x, str) for x in exp.to_list()]
    exp_number = [isinstance(x, float) or isinstance(x, int) for x in exp.to_list()]
    
    if any(exp_str) and any(exp_number):
        warnings.warn('Recoded column contains both string and float/int/NaN values!')
    
    if original_dt != exp_dt:
        warnings.warn(f'Column dtype changed from {str(original_dt).upper()} to {str(exp_dt).upper()}.')
        
    return exp

def _recode_string_to_instructions(recode_string, var):
    '''take in a string and return a dictionary to replace'''
    assert isinstance(recode_string, str), f"`recode_str` should be a string not {type(recode_string)}"
    assert isinstance(var, pd.Series), f"`var` should be a pandas Series not {type(var)}"
    
    keys = var.drop_duplicates().to_list()
    mapping = dict(zip(keys, keys))
    mapping['NaN'] = "NaN"
    
    #extract mapping from str
    pieces = [x.strip() for x in recode_string.split(";")]
    
    all_recodes = []
    #apply recode function
    for piece in pieces:
        if piece: #if the user ends with a colon will create a blank space
            piece_edited = _evaluate_piece(piece, mapping)
            all_recodes.append(piece_edited)
            for k,v in piece_edited.items():
                mapping[k] = v
    
    check_for_duplicates = list(map(_grab_keys, all_recodes))
    check_for_duplicates = functools.reduce(operator.add, check_for_duplicates)
    assert len(list(set(check_for_duplicates))) == len(check_for_duplicates), \
        f"Error! You have made more than one assignment to a number or string."
    
    return mapping

def _evaluate_piece(piece_str, mapping):
    '''take in a single recode key-value pair and spit out a dictionary of recodes'''
    assert "=" in piece_str, f"There is an `=` in the wrong place. Check your recode string!"
    assert ";" not in piece_str, f"There is a `;` in the wrong place. Check your recode string!"
    mapping_is_digit = isinstance(list(mapping.keys())[0], int) or \
        isinstance(list(mapping.keys())[0], float)
    parts = piece_str.split("=")
    parts = [x.strip() for x in parts]
    map_value = parts[1]
    map_key = parts[0]
    if map_value.isdigit():
        map_value = int(map_value)
    else:
        if map_value == "NaN":
            map_value = np.nan
        else:
            map_value = map_value.replace("'", "").replace('"', "")
    
    #interval
    if ":" in map_key:
        assert map_key.count(":") == 1
        if "lo:" in map_key:
            map_key = map_key.replace("lo:", "")
            map_key = int(map_key)
            list_to_iterate = \
                [x for x in list(mapping.keys()) if isinstance(x, int) or isinstance(x, float)]
            map_key = [x for x in list_to_iterate if x<=map_key]
        elif ":hi" in map_key:
            map_key = map_key.replace(":hi", "")
            map_key = int(map_key)
            list_to_iterate = \
                [x for x in list(mapping.keys()) if isinstance(x, int) or isinstance(x, float)]
            map_key = [x for x in list_to_iterate if x>=map_key]
        else: 
            map_key = map_key.split(":")
            map_key = [x for x in range(int(map_key[0]), int(map_key[1])+1)]
        map_value = [map_value for x in range(len(map_key))]
        map_dict = dict(zip(map_key, map_value))
        assert len(map_dict)<100, \
            "Please recode this a different/more efficient way. There are 100 or more recodes\
                in one category!"
    
    #list
    elif "[" in map_key and "]" in map_key:
        map_key = map_key.replace("[", "").replace("]", "")
        if mapping_is_digit:
            map_key = [int(x) for x in map_key]
        else:
            map_key = map_key.replace("'", "").replace('"', "")
        map_key = map_key.split(",")
        map_value = [map_value for x in range(len(map_key))]
        map_dict = dict(zip(map_key, map_value))
    
    #single value
    elif map_key.isdigit():
        map_dict = {int(map_key):map_value}
    else:
        map_key = map_key.replace("'", "").replace('"', "")
        map_dict = {map_key:map_value}

    return map_dict

def _get_interactions(list1, list2):
    '''return list of interactions'''
    exp = []
    for el1 in list1:
        for el2 in list2:
            exp.append(f'{el1}|{el2}')
    return exp

def _grab_keys(dictionary):
    '''return dictionary's keys as a list'''
    return list(dictionary.keys())

def _is_ordered_category_var(var):
    '''check if is an ordered category variable given it is a `pd.Series` object'''
    assert isinstance(var, pd.Series), "The variable input is not a series. \
        There is something wrong with the variable in this dataframe."
    if str(var.dtype) == "category":
        return var.cat.ordered
    else:
        return False

def get_names(data, match_str):
    '''returns list of column names given regex matching. 
    If no matches are found returns empty list'''
    assert isinstance(data, pd.DataFrame), 'data must be an instance of `pandas.DataFrame`'
    assert isinstance(match_str, str), '`match_str` must be a `str`'
    return list(data.filter(regex=match_str).columns)

def _check_eq(data, weighting_df, weight_nm):
    col_bools = []
    for wt_col in weighting_df.Names.drop_duplicates().to_list():
        are_equal = tabs(data, wt_col, wts=weight_nm, display='column').sort_index().round(1).to_list() == \
            weighting_df[weighting_df.Names == wt_col].set_index('Levels')['Proportions'].mul(100).sort_index().round(1).to_list()
        col_bools.append(are_equal)
    return all(col_bools)

def rake_weight(
    data:pd.DataFrame,
    weighting_df:pd.DataFrame,
    cap:int = 10,
    weight_nm:str = 'weight',
    qa:bool = True
):
    '''Weight data using raking given a weighting `pd.DataFrame`. Returns a new `pd.DataFrame` with an additional column
    with the new weights.
    
    Required Arguments:
        data: `pandas.DataFrame` of wide survey data
        weighting_df: `pandas.DataFrame` of variables to weight on. All levels of weighting variables must be in data.
            Weigting variables cannot be missing data. If there exists a category less than 5%, will throw error. 
            First column is weighting variables names. Second column is weighting variable levels. Third column
            is target proportion. Columns should be named: `Names`, `Levels`, and `Proportions`.
    
    Keyword Arguments:
        cap: `int` default value 10. Weights will not be above cap.
        weight_nm: `str` default value is `weight`. Specify different name if desired.
        qa: `bool` default is True. Prints weighting diagnostics.
    
    Returns:
        New `pandas.DataFrame` of original survey data with additional weighting column
    '''
    #Checking `weighting_df` is set up correctly
    assert weighting_df.shape[1] == 3, 'only three columns allowed in `weighting_df`'
    assert weighting_df.columns.to_list() == ['Names', 'Levels', 'Proportions'], 'names of `weighting_df` are incorrect'
    missing_cols = [col not in data.columns.to_list() for col in weighting_df.Names.to_list()]
    assert not all(missing_cols), f'these are not in data columns: \
        {list(set(weighting_df.Names[missing_cols].to_list()))}'
    assert all(weighting_df.groupby('Names')['Proportions'].sum().eq(1)), \
        f'All proportions of weighting variables must sum to 1 in `weighting_df` \
            {weighting_df.groupby("Names")["Proportions"].sum()}'
    assert weighting_df.isna().sum().sum() == 0, 'No NAs allowed in `weighting_df`'
    assert weight_nm not in data.columns.to_list(), f'`weight_nm`: {weight_nm} should not be in data already'
    reset = False
    
    #Check no NA in weighting variables
    weighting_cols = weighting_df.Names.drop_duplicates().to_list()
    base = data[weighting_cols].isna().sum()
    cols_w_NAs = base.index[base.ne(0)].to_list()
    assert len(cols_w_NAs) == 0, f'These cols have NAs:{cols_w_NAs}'
    
    #Check all levels are in data
    for col in weighting_cols:
        lvls_in_data = set(data[col].value_counts().index.to_list())
        lvls_in_wts = set(weighting_df.query('Names == @col').Levels.to_list())
        assert len(lvls_in_data) == len(lvls_in_wts), \
            f'for {col}, {lvls_in_data} in data and {lvls_in_wts} in weights'
    
    #Check all levels are >5% of data
    for col in weighting_cols:
        lvls_in_data = data[col].value_counts(normalize = True).gt(0.05)
        assert all(lvls_in_data), f'some levels are not >5% of data. E.G. \
            {data[col].value_counts(normalize = True)}'
            
    
    # Begin Algorithm... 
    # Step 1: Generate new column for weights
    data[weight_nm] = 1.0
    
    # Step 2: Iterate through each variable resetting proportions
    N = data.shape[0]
    Not_Converged_Flag = True
    iterations = 0
    safety_cap = np.subtract(cap,0.1)
    
    while Not_Converged_Flag:
        
        iterations += 1
    
        for wt_col in weighting_cols:
            
            wt_truth = weighting_df[weighting_df.Names == wt_col]
            data_state = tabs(data, wt_col, wts=weight_nm, display='column')
            wt_truth_it = wt_truth.Levels.to_list()
            
            for lvl in wt_truth_it:
                true_count = np.multiply(wt_truth[wt_truth.Levels == lvl]['Proportions'].iloc[0],N)
                data_count = np.multiply(data_state.div(100).loc[lvl],N)
                scaling_factor = np.divide(true_count,data_count)
                data.loc[data[wt_col]==lvl, weight_nm] = \
                    np.multiply(data.loc[data[wt_col]==lvl, weight_nm],scaling_factor)
            
            #create minimum and cap
            # if any(data[weight_nm].lt(0.1)):
            #     data.loc[data[weight_nm].lt(0.1), weight_nm] = 0.1
            #     reset = True
            if any(data[weight_nm].gt(safety_cap)):
                data.loc[data[weight_nm].gt(safety_cap), weight_nm] = safety_cap
                reset = True
            
            if reset:
                scaling_factor = np.divide(N,data[weight_nm].sum())
                data[weight_nm] = np.multiply(data[weight_nm],scaling_factor)
                reset = False
        
        # Step 3: Check all proportion alignment if not repeat Step 2
        if _check_eq(data, weighting_df, weight_nm):
            Not_Converged_Flag = False
        
        #reset weights to sum to N
        scaling_factor = np.divide(N,data[weight_nm].sum())
        data[weight_nm] = np.multiply(data[weight_nm],scaling_factor)
        
        if iterations >= 5_000:
            for var in weighting_cols:
                print("Variable: ",var)
                print(tabs(data, var, wts=weight_nm, display='column'))
            print(f'''
                Iterations: {iterations}
                Max Weight: {data[weight_nm].max()}
                Min Weight: {data[weight_nm].min()}
                '''
            )
            print(data)
            raise Exception("Iterations exceeded 5_000 without converging. Please review data or adjust variables.")
        
    if qa:
        for var in weighting_cols:
            print("Variable: ",var)
            print(tabs(data, var, wts=weight_nm, display='column'))
        print(f'''
            Iterations: {iterations}
            Max Weight: {data[weight_nm].max()}
            Min Weight: {data[weight_nm].min()}
            '''
        )
    
    return data
