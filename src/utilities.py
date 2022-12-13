import math
import json
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
from params import choose_params, include_sci, feature_selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn import tree
from sklearn.inspection import permutation_importance

''' ################################################
            Functions for dataset preparation 
    ################################################ '''
def remove_prefix(text, prefix):
    '''
    Remove everything before a character in a string
    :param text: string
    :param prefix: character before which text must be removed
    :return: text without characters before the 'character'
    '''
    return text[text.find(prefix):]

def bool_vars(mimi):
    ''' This function transforms categorical variables in the dataset into boolean ones'''
    print('Deriving boolean variables from categorical features ...')

    '''Neighbors (T/F)'''
    mimi['origin_neighbors'] = mimi['origin_neighbors'].fillna('[]')
    mimi['destination_neighbors'] = mimi['destination_neighbors'].fillna('[]')
    mimi['neighbors'] = mimi.apply(lambda x: 1 if x['origin_name'] in x['destination_neighbors'] or
                                                  x['destination_name'] in x['origin_neighbors'] else 0, axis=1)

    '''Same contintent (T/F)'''
    mimi['share_cont'] = mimi.apply(lambda x: 1 if x['origin_cont_code'] == x['destination_cont_code'] else 0, axis=1)

    '''Same religion (T/F)'''
    mimi['share_rel'] = mimi.apply(lambda x: 1 if x['origin_religion'] == x['destination_religion'] else 0, axis=1)

    '''If share at least one language (T/F)'''
    mimi['origin_languages'] = mimi['origin_languages'].str.split(',').fillna('')
    mimi['destination_languages'] = mimi['destination_languages'].str.split(',').fillna('')
    w = []
    for a, b in zip(mimi['origin_languages'], mimi['destination_languages']):
        if ('' in a) | ('' in b):
            w.append(math.nan)
        else:
            w.append(any(i in b for i in a))
            # print(any(i in b for i in a))
    mimi['share_lang'] = w
    mimi['share_lang'] = mimi['share_lang'].fillna(0)  # TODO: VERIFICARE CORRETTEZZA COCNETTUALE
    mimi['share_lang'] = mimi['share_lang'].astype(int)  # Boolean True/False into int 1/0

    return mimi

def compute_indices(mimi):
    ''' This function computes unidirectional and bidirectional migration indicators '''
    print('Computing migration indices ...')
    print('Country coupling check:\n', mimi[mimi.from_to=='IT-RO'], mimi[mimi.from_to=='RO-IT'])
    print('Assigning unique ID to combination of two columns ...')
    mimi["id"] = mimi.groupby(mimi[["origin_country", "destination_country"]].apply(frozenset, axis=1)).ngroup() + 1
    print('The dataset has', mimi.id.nunique(), 'unique pairs of countries.')

    mimi_orig = mimi.copy()

    '''
    Each migration index is computed yearly, in the following range: 2010-2020 for UN flows, 2010-2019 for EUROSTAT flows. 
    Moreover, each migration indicator is computed separately for UN and EUROSTAT, and separately also by citizenship/birthplace and by residence.
    
    ** Methodology for computing indices: **
    '''
    # TODO: ricontrollare e terminare parte testuale sopra

    print('Computing indices for UN flows ...')

    ''' UN by cit'''
    for y in range(2010, 2021):

        ''' Numerator'''
        mimi[f'numerator_UN_{str(y)}_T_T_cit_count'] = mimi.groupby(["id"])[f'UN_{str(y)}_T_T_cit'].transform('count') > 1
        mimi[f'numerator_UN_{str(y)}_T_T_cit'] = math.nan
        mimi[f'numerator_UN_{str(y)}_T_T_cit'] = \
        mimi[mimi[f'numerator_UN_{str(y)}_T_T_cit_count'].eq(True)].groupby(["id"])[f'UN_{str(y)}_T_T_cit'].transform(lambda x: np.sum(x.values))

        ''' BMI '''  # (flow i->j + flow j->i) / (pop_i * pop_j)
        mimi[f'UN_flow_BMI_{str(y)}_T_T_cit'] = mimi[f'numerator_UN_{str(y)}_T_T_cit'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

        ''' BMP '''  # (flow i->j + flow j->i) / (pop_i + pop_j)
        mimi[f'UN_flow_BMP_{str(y)}_T_T_cit'] = mimi[f'numerator_UN_{str(y)}_T_T_cit'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

        ''' Unidirectional indices '''
        # UMI: (flow i->j) / (pop_i * pop_j)
        mimi[f'UN_flow_UMI_{str(y)}_T_T_cit'] = mimi[f'UN_{str(y)}_T_T_cit'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}']) * 1000000000
        # UMP: (flow i->j) / (pop_i + pop_j)
        mimi[f'UN_flow_UMP_{str(y)}_T_T_cit'] = mimi[f'UN_{str(y)}_T_T_cit'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

    ''' UN by res'''
    for y in range(2010, 2021):

        ''' Numerator'''
        mimi[f'numerator_UN_{str(y)}_T_T_res_count'] = mimi.groupby(["id"])[f'UN_{str(y)}_T_T_res'].transform('count') > 1
        mimi[f'numerator_UN_{str(y)}_T_T_res'] = math.nan
        mimi[f'numerator_UN_{str(y)}_T_T_res'] = \
        mimi[mimi[f'numerator_UN_{str(y)}_T_T_res_count'].eq(True)].groupby(["id"])[f'UN_{str(y)}_T_T_res'].transform(lambda x: np.sum(x.values))

        ''' BMI '''  # (flow i->j + flow j->i) / (pop_i * pop_j)
        mimi[f'UN_flow_BMI_{str(y)}_T_T_res'] = mimi[f'numerator_UN_{str(y)}_T_T_res'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

        ''' BMP '''  # (flow i->j + flow j->i) / (pop_i + pop_j)
        mimi[f'UN_flow_BMP_{str(y)}_T_T_res'] = mimi[f'numerator_UN_{str(y)}_T_T_res'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

        ''' Unidirectional indices'''
        # UMI: (flow i->j) / (pop_i * pop_j)
        mimi[f'UN_flow_UMI_{str(y)}_T_T_res'] = mimi[f'UN_{str(y)}_T_T_res'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}']) * 1000000000
        # UMP: (flow i->j) / (pop_i + pop_j)
        mimi[f'UN_flow_UMP_{str(y)}_T_T_res'] = mimi[f'UN_{str(y)}_T_T_res'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}']) * 1000000000

    print('Computing indices for EUROSTAT flows ...')

    ''' ESTAT by cit'''
    for y in range(2010, 2020):
        mimi[f'numerator_ESTAT_{str(y)}_T_T_cit_count'] = mimi.groupby(["id"])[f'ESTAT_{str(y)}_T_T_cit'].transform('count') > 1
        mimi[f'numerator_ESTAT_{str(y)}_T_T_cit'] = math.nan
        mimi[f'numerator_ESTAT_{str(y)}_T_T_cit'] = \
        mimi[mimi[f'numerator_ESTAT_{str(y)}_T_T_cit_count'].eq(True)].groupby(["id"])[
            f'ESTAT_{str(y)}_T_T_cit'].transform(lambda x: np.sum(x.values))
        mimi[f'ESTAT_flow_BMI_{str(y)}_T_T_cit'] = mimi[f'numerator_ESTAT_{str(y)}_T_T_cit'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] * mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_BMP_{str(y)}_T_T_cit'] = mimi[f'numerator_ESTAT_{str(y)}_T_T_cit'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] + mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_UMI_{str(y)}_T_T_cit'] = mimi[f'ESTAT_{str(y)}_T_T_cit'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] * mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_UMP_{str(y)}_T_T_cit'] = mimi[f'ESTAT_{str(y)}_T_T_cit'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] + mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000

    ''' ESTAT by res'''
    for y in range(2010, 2020):
        mimi[f'numerator_ESTAT_{str(y)}_T_T_res_count'] = mimi.groupby(["id"])[f'ESTAT_{str(y)}_T_T_res'].transform(
            'count') > 1
        mimi[f'numerator_ESTAT_{str(y)}_T_T_res'] = math.nan
        mimi[f'numerator_ESTAT_{str(y)}_T_T_res'] = \
        mimi[mimi[f'numerator_ESTAT_{str(y)}_T_T_res_count'].eq(True)].groupby(["id"])[
            f'ESTAT_{str(y)}_T_T_res'].transform(lambda x: np.sum(x.values))
        mimi[f'ESTAT_flow_BMI_{str(y)}_T_T_res'] = mimi[f'numerator_ESTAT_{str(y)}_T_T_res'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] * mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_BMP_{str(y)}_T_T_res'] = mimi[f'numerator_ESTAT_{str(y)}_T_T_res'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] + mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_UMI_{str(y)}_T_T_res'] = mimi[f'ESTAT_{str(y)}_T_T_res'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] * mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000
        mimi[f'ESTAT_flow_UMP_{str(y)}_T_T_res'] = mimi[f'ESTAT_{str(y)}_T_T_res'].divide(
            mimi[f'ESTAT_origin_pop_{str(y)}'] + mimi[f'ESTAT_destination_pop_{str(y)}']) * 1000000000

    ''' MERGE by cit '''
    for y in range(2010, 2021):
        mimi_orig[f'TOT_flow_BMP_{str(y)}_T_T_cit'] = mimi[f'UN_flow_BMP_{str(y)}_T_T_cit']
        mimi_orig[f'TOT_flow_BMI_{str(y)}_T_T_cit'] = mimi[f'UN_flow_BMI_{str(y)}_T_T_cit']
        mimi_orig[f'TOT_flow_UMP_{str(y)}_T_T_cit'] = mimi[f'UN_flow_UMP_{str(y)}_T_T_cit']
        mimi_orig[f'TOT_flow_UMI_{str(y)}_T_T_cit'] = mimi[f'UN_flow_UMI_{str(y)}_T_T_cit']
    for y in range(2010, 2020):
        mimi_orig.loc[mimi_orig[f'TOT_flow_BMP_{str(y)}_T_T_cit'] == 0,
                      f'TOT_flow_BMP_{str(y)}_T_T_cit'] = mimi[f'ESTAT_flow_BMP_{str(y)}_T_T_cit']
        mimi_orig.loc[mimi_orig[f'TOT_flow_BMI_{str(y)}_T_T_cit'] == 0,
                      f'TOT_flow_BMI_{str(y)}_T_T_cit'] = mimi[f'ESTAT_flow_BMI_{str(y)}_T_T_cit']
        mimi_orig.loc[mimi_orig[f'TOT_flow_UMP_{str(y)}_T_T_cit'] == 0,
                      f'TOT_flow_UMP_{str(y)}_T_T_cit'] = mimi[f'ESTAT_flow_UMP_{str(y)}_T_T_cit']
        mimi_orig.loc[mimi_orig[f'TOT_flow_UMI_{str(y)}_T_T_cit'] == 0,
                      f'TOT_flow_UMI_{str(y)}_T_T_cit'] = mimi[f'ESTAT_flow_UMI_{str(y)}_T_T_cit']

    ''' MERGE by res '''
    for y in range(2010, 2021):
        mimi_orig[f'TOT_flow_BMP_{str(y)}_T_T_res'] = mimi[f'UN_flow_BMP_{str(y)}_T_T_res']
        mimi_orig[f'TOT_flow_BMI_{str(y)}_T_T_res'] = mimi[f'UN_flow_BMI_{str(y)}_T_T_res']
        mimi_orig[f'TOT_flow_UMP_{str(y)}_T_T_res'] = mimi[f'UN_flow_UMP_{str(y)}_T_T_res']
        mimi_orig[f'TOT_flow_UMI_{str(y)}_T_T_res'] = mimi[f'UN_flow_UMI_{str(y)}_T_T_res']
    for y in range(2010, 2020):
        mimi_orig.loc[mimi_orig[f'TOT_flow_BMP_{str(y)}_T_T_res'] == 0,
                      f'TOT_flow_BMP_{str(y)}_T_T_res'] = mimi[f'ESTAT_flow_BMP_{str(y)}_T_T_res']
        mimi_orig.loc[mimi_orig[f'TOT_flow_BMI_{str(y)}_T_T_res'] == 0,
                      f'TOT_flow_BMI_{str(y)}_T_T_res'] = mimi[f'ESTAT_flow_BMI_{str(y)}_T_T_res']
        mimi_orig.loc[mimi_orig[f'TOT_flow_UMP_{str(y)}_T_T_res'] == 0,
                      f'TOT_flow_UMP_{str(y)}_T_T_res'] = mimi[f'ESTAT_flow_UMP_{str(y)}_T_T_res']
        mimi_orig.loc[mimi_orig[f'TOT_flow_UMI_{str(y)}_T_T_res'] == 0,
                      f'TOT_flow_UMI_{str(y)}_T_T_res'] = mimi[f'ESTAT_flow_UMI_{str(y)}_T_T_res']

    print('Computing indices for migration stocks ...')

    ''' STOCKS '''
    for y in range(2000, 2025, 5):
        mimi[f'numerator_stocks_{str(y)}_T_T'] = mimi.groupby(["id"])[f'UN_migr_stocks_{str(y)}_T_T'].transform(sum)
        mimi[f'stocks_BMI_{str(y)}_T_T'] = mimi[f'numerator_stocks_{str(y)}_T_T'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}'])
        mimi[f'stocks_BMP_{str(y)}_T_T'] = mimi[f'numerator_stocks_{str(y)}_T_T'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}'])
        mimi[f'stocks_UMI_{str(y)}_T_T'] = mimi[f'UN_migr_stocks_{str(y)}_T_T'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] * mimi[f'UN_destination_pop_{str(y)}'])
        mimi[f'stocks_UMP_{str(y)}_T_T'] = mimi[f'UN_migr_stocks_{str(y)}_T_T'].divide(
            mimi[f'UN_origin_pop_{str(y)}'] + mimi[f'UN_destination_pop_{str(y)}'])

    return mimi, mimi_orig

def num_vars(mimi):
    ''' This function computes new numerical features derived from existing ones'''
    # GDP difference + .fillna GDP 2018 with .ffill method
    mimi['gdp_diff_2018'] = mimi.origin_gdp_2018 - mimi.destination_gdp_2018
    x = [col for col in mimi if col.startswith('origin_gdp')]
    y = [col for col in mimi if col.startswith('destination_gdp')]
    gdp_orig_df = mimi[x].sort_index(axis=1)
    gdp_dest_df = mimi[y].sort_index(axis=1)
    gdp_orig_df = gdp_orig_df.fillna(axis=1, method='ffill')
    gdp_dest_df = gdp_dest_df.fillna(axis=1, method='ffill')
    #print('\nOrig GDP 2018 null values: \n Prima: ', len(mimi[mimi[x].origin_gdp_2018.isna()]), '\n Dopo: ', len(gdp_orig_df[gdp_orig_df.origin_gdp_2018.isna()]))
    #print('\nDest GDP 2018 null values: \n Prima: ', len(mimi[mimi[y].destination_gdp_2018.isna()]), '\n Dopo: ', len(gdp_dest_df[gdp_dest_df.destination_gdp_2018.isna()]))
    # substitute cols
    mimi['origin_gdp_2018'] = gdp_orig_df['origin_gdp_2018']
    mimi['destination_gdp_2018'] = gdp_dest_df['destination_gdp_2018']

    return mimi

''' ###################################################
        Stepwise elimination for linear regression 
    ################################################### '''

def forwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst", sl=0.05):
    """
    Forward Selection is a function, based on regression dump, that returns significant features and selection iterations.\n
    Required Libraries: pandas, numpy, statmodels

    Parameters
    ----------
    X : Independent variables (Pandas Dataframe)\n
    y : Dependent variable (Pandas Series, Pandas Dataframe)\n
    model_type : 'linear' or 'logistic'\n
    elimination_criteria : 'aic', 'bic', 'r2', 'adjr2' or None\n
        'aic' refers Akaike information criterion\n
        'bic' refers Bayesian information criterion\n
        'r2' refers R-squared (Only works on linear model type)\n
        'r2' refers Adjusted R-squared (Only works on linear model type)\n
    varchar_process : 'drop', 'dummy' or 'dummy_dropfirst'\n
        'drop' drops varchar features\n
        'dummy' creates dummies for all levels of all varchars\n
        'dummy_dropfirst' creates dummies for all levels of all varchars, and drops first levels\n
    sl : Significance Level (default: 0.05)\n

    Returns
    -------
    columns(list), iteration_logs(str)\n\n
    Not Returns a Model

    Tested On
    ---------
    Python v3.6.7, Pandas v0.23.4, Numpy v1.15.04, StatModels v0.9.0

    See Also
    --------
    https://en.wikipedia.org/wiki/Stepwise_regression
    """
    X = __varcharProcessing__(X, varchar_process=varchar_process)
    return __forwardSelectionRaw__(X, y, model_type=model_type, elimination_criteria=elimination_criteria, sl=sl)

def backwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst", sl=0.05):
    """
    Backward Selection is a function, based on regression dump, that returns significant features and selection iterations.\n
    Required Libraries: pandas, numpy, statmodels

    Parameters
    ----------
    X : Independent variables (Pandas Dataframe)\n
    y : Dependent variable (Pandas Series, Pandas Dataframe)\n
    model_type : 'linear' or 'logistic'\n
    elimination_criteria : 'aic', 'bic', 'r2', 'adjr2' or None\n
        'aic' refers Akaike information criterion\n
        'bic' refers Bayesian information criterion\n
        'r2' refers R-squared (Only works on linear model type)\n
        'r2' refers Adjusted R-squared (Only works on linear model type)\n
    varchar_process : 'drop', 'dummy' or 'dummy_dropfirst'\n
        'drop' drops varchar features\n
        'dummy' creates dummies for all levels of all varchars\n
        'dummy_dropfirst' creates dummies for all levels of all varchars, and drops first levels\n
    sl : Significance Level (default: 0.05)\n

    Returns
    -------
    columns(list), iteration_logs(str)\n\n
    Not Returns a Model

    Tested On
    ---------
    Python v3.6.7, Pandas v0.23.4, Numpy v1.15.04, StatModels v0.9.0

    See Also
    --------
    https://en.wikipedia.org/wiki/Stepwise_regression
    """
    X = __varcharProcessing__(X, varchar_process=varchar_process)
    return __backwardSelectionRaw__(X, y, model_type=model_type, elimination_criteria=elimination_criteria, sl=sl)

def __varcharProcessing__(X, varchar_process="dummy_dropfirst"):
    dtypes = X.dtypes
    if varchar_process == "drop":
        X = X.drop(columns=dtypes[dtypes == np.object].index.tolist())
        print("Character Variables (Dropped):", dtypes[dtypes == np.object].index.tolist())
    elif varchar_process == "dummy":
        X = pd.get_dummies(X, drop_first=False)
        print("Character Variables (Dummies Generated):", dtypes[dtypes == np.object].index.tolist())
    elif varchar_process == "dummy_dropfirst":
        X = pd.get_dummies(X, drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):",
              dtypes[dtypes == np.object].index.tolist())
    else:
        X = pd.get_dummies(X, drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):",
              dtypes[dtypes == np.object].index.tolist())

    X["intercept"] = 1
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]

    return X

def __forwardSelectionRaw__(X, y, model_type="linear", elimination_criteria="aic", sl=0.05):
    iterations_log = ""
    cols = X.columns.tolist()

    def regressor(y, X, model_type=model_type):
        if model_type == "linear":
            regressor = sm.OLS(y, X).fit()
        elif model_type == "logistic":
            regressor = sm.Logit(y, X).fit()
        else:
            print("\nWrong Model Type : " + model_type + "\nLinear model type is seleted.")
            model_type = "linear"
            regressor = sm.OLS(y, X).fit()
        return regressor

    selected_cols = ["intercept"]
    other_cols = cols.copy()
    other_cols.remove("intercept")

    model = regressor(y, X[selected_cols])

    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type == "linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type == "linear":
        criteria = model.rsquared_adj

    for i in range(X.shape[1]):
        pvals = pd.DataFrame(columns=["Cols", "Pval"])
        for j in other_cols:
            model = regressor(y, X[selected_cols + [j]])
            pvals = pvals.append(pd.DataFrame([[j, model.pvalues[j]]], columns=["Cols", "Pval"]), ignore_index=True)
        pvals = pvals.sort_values(by=["Pval"]).reset_index(drop=True)
        pvals = pvals[pvals.Pval <= sl]
        if pvals.shape[0] > 0:

            model = regressor(y, X[selected_cols + [pvals["Cols"][0]]])
            iterations_log += str("\nEntered : " + pvals["Cols"][0] + "\n")
            iterations_log += "\n\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n\n"

            if elimination_criteria == "aic":
                new_criteria = model.aic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tAIC :", model.aic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "bic":
                new_criteria = model.bic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tBIC :", model.bic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "r2" and model_type == "linear":
                new_criteria = model.rsquared
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tR2 :", model.rsquared)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "adjr2" and model_type == "linear":
                new_criteria = model.rsquared_adj
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tAdjR2 :", model.rsquared_adj)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("Break : Criteria")
                    break
            else:
                print("Entered :", pvals["Cols"][0])
                selected_cols.append(pvals["Cols"][0])
                other_cols.remove(pvals["Cols"][0])

        else:
            print("Break : Significance Level")
            break

    model = regressor(y, X[selected_cols])
    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type == "linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type == "linear":
        criteria = model.rsquared_adj

    print(model.summary())
    print("AIC: " + str(model.aic))
    print("BIC: " + str(model.bic))
    print("Final Variables:", selected_cols)

    return selected_cols, iterations_log

def __backwardSelectionRaw__(X, y, model_type="linear", elimination_criteria="aic", sl=0.05):
    iterations_log = ""
    last_eleminated = ""
    cols = X.columns.tolist()

    def regressor(y, X, model_type=model_type):
        if model_type == "linear":
            regressor = sm.OLS(y, X).fit()
        elif model_type == "logistic":
            regressor = sm.Logit(y, X).fit()
        else:
            print("\nWrong Model Type : " + model_type + "\nLinear model type is seleted.")
            model_type = "linear"
            regressor = sm.OLS(y, X).fit()
        return regressor

    for i in range(X.shape[1]):
        if i != 0:
            if elimination_criteria == "aic":
                criteria = model.aic
                new_model = regressor(y, X)
                new_criteria = new_model.aic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "bic":
                criteria = model.bic
                new_model = regressor(y, X)
                new_criteria = new_model.bic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "adjr2" and model_type == "linear":
                criteria = model.rsquared_adj
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared_adj
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "r2" and model_type == "linear":
                criteria = model.rsquared
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            else:
                new_model = regressor(y, X)
            model = new_model
            iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n"
        else:
            model = regressor(y, X)
            iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n"
        maxPval = max(model.pvalues)
        cols = X.columns.tolist()
        if maxPval > sl:
            for j in cols:
                if (model.pvalues[j] == maxPval):
                    print("Eliminated :", j)
                    iterations_log += str("\n\nEliminated : " + j + "\n\n")

                    del X[j]
                    last_eleminated = j
        else:
            break
    print(str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(model.bic))
    print("Final Variables:", cols)
    iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(model.bic) + "\n"
    return cols, iterations_log

def linear_regression_ols(df, X, Y, y):
    ''' ... '''

    ''' Rescaling:
    The most common adopted normalizations are: Z-Score and Min-Max.
    - The Min-Max normalization approach allows to have data that are always interpretable, so that we can avoid to apply inverse transformation.
    - The Z-Score normalization approach exploits the mean and standard deviation of data, and tries to center data with respect to these two statistical properties.
    - The StandardScaler ... # todo finire parte testuale
    In this analysis we choose to use the Min-Max normalization in order to have a leaner approach and avoid an unnecessary inverse transformation.
    '''
    # drop feature y to predict form X matrix
    X.drop(y, axis=1, inplace=True)

    # scaler = StandardScaler()
    # X_norm = scaler.fit_transform(X.values)

    scaler = StandardScaler() # attenzione: prima era MinMaxScaler(feature_range=(-1, 1))
    X_norm = scaler.fit_transform(X.values)

    # rescale
    scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(scaled)

    print('OLS Backward Elimination Stepwise Linear Regression STARTED')
    print('We are considering', df.origin_name.nunique(), 'countries of origin, and', df.destination_name.nunique(),
          'countries of destination.')

    X_scaled.columns = X.columns

    # print(X_scaled)

    # X_norm = sm.add_constant(X_scaled)
    X_scaled = sm.add_constant(X_scaled)

    '''
    model = sm.OLS(Y,X_scaled) #,missing='drop' to solve error 'exog contains inf or nans'
    results = model.fit()
    print(results.params)
    print(results.summary())'''

    ''' BACKWARD ELIMINATION'''
    final_vars, iterations_logs = backwardSelection(X_scaled, Y, model_type="linear")
    # Write Logs To .txt
    # with open('iterations_logs.txt', 'a') as iterations_file:
    # iterations_file.write(iterations_logs)
    iterations_file = open(f"../txt_log/back_iterations_logs_{str(y)}.txt", "a")
    iterations_file.write(iterations_logs)
    iterations_file.close()

    return final_vars

''' ######################################
        Functions for learning dump 
    ###################################### '''

def dataset_split(X, Y, valid=False):
    '''
    ...
    [sklearn.model_selection.train_test_split]
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)'''
    if valid==True:

        # In the first step we split the data in training (80%) and remaining dataset (20%)
        X_train, X_rem, y_train, y_rem = train_test_split(
            X, Y, train_size=0.8, random_state=None, shuffle=True, stratify=None)

        # Now since we want the valid and test size to be equal
        # we have to define valid_size=0.5 (that is 50% of remaining data)
        test_size = 0.5
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=test_size)

        print('Obtained dataset split: ')
        print('------ TR ------')
        print(X_train.shape, y_train.shape)
        print('------ VAL ------')
        print(X_val.shape, y_val.shape)
        print(' ------ TS ------')
        print(X_test.shape, y_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test

    else:

        X_dev, X_test, y_dev, y_test = train_test_split(
            X, Y, train_size=0.8, random_state=None, shuffle=True, stratify=None)

        print('\n Obtained dataset split: ')
        print('------ TR ------')
        print(X_dev.shape, y_dev.shape)
        print(' ------ TS ------')
        print(X_test.shape, y_test.shape)

        return X_dev, y_dev, X_test, y_test

def run_gs(train_set, train_label, mod, param_dist, n_iter_search, scoring, randomized=False, **kwargs):
    '''

    This function ... #todo finire parte testuale
    The scikit-learn library provides cross-validation random search and grid search hyperparameter optimization via the RandomizedSearchCV and GridSearchCV classes respectively.

    'n_jobs' parameter is the number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

    :param train_set:
    :param train_label:
    :param mod:
    :param param_dist:
    :param n_iter_search:
    :param scoring:
    :param randomized:
    :param kwargs:
    :return:
    '''

    # define search
    if randomized==True:
        grid_search = RandomizedSearchCV(mod, param_distributions=param_dist,
                                                n_iter=n_iter_search,
                                                n_jobs=-1,
                                                scoring=scoring,
                                                verbose=0)
    else:
        grid_search = GridSearchCV(mod, param_grid=param_dist,
                                         n_jobs=-1,
                                         scoring=scoring,
                                         verbose=1)
    # execute search
    grid_search.fit(train_set, train_label)

    return grid_search

def write_gs_results(grid_search, search_type, model_name):
    ''' ... '''
    y = choose_params(returndf=False)
    results = '\n' + '-'*100 + f'\nRegression model to predict: {str(y)}' + \
               f'\n\nModel: {str(model_name)} \n\nType of search: {str(search_type)} with Nested K-Fold Cross Validation'
    results += '\n\nBest setting parameters:\n' + str(grid_search.best_params_) + \
               '\n\nEstimator chosen (which gave the highest score):\n' + str(grid_search.best_estimator_) + \
               '\n\nMean cross-validated score of the best_estimator: ' + str(grid_search.best_score_) + \
               '\n\nScorer function used: ' + str(grid_search.scorer_) + \
               '\n\nNumber of cross-validation splits (folds/iterations): ' + str(grid_search.n_splits_) + \
               '\n\nTime (in sec) for refitting the best model on the whole dataset: ' + str(grid_search.refit_time_) + \
               '\n\n[' + str(datetime.datetime.now()) + ']\n' + '-'*100
    output_file = open(f"../txt_log/grid_search_results.txt", "a")
    output_file.write(results)
    output_file.close()
    print(f'Results/Logs have been succesfully stored in "grid_search_results" txt file')

def write_output_file(text, file_name):
    tot_text = '\n' + '-'*100 + '\n'
    tot_text += text
    tot_text += '\n\n[' + str(datetime.datetime.now()) + ']\n' + '-' * 100
    output_file = open(f"../txt_log/{str(file_name)}.txt", "a")
    output_file.write(tot_text)
    output_file.close()
    print(f'Results/Logs have been succesfully stored in "{str(file_name)}" txt file')

''' #################################
           Functions for plots
    ################################# '''

def impurity_feature_importance(model, X, mod_name): # todo considerare di eliminare questa funzione
    #feature_names = [f"feature {i}" for i in range(X.shape[1])]
    feature_names = X.columns.values
    start_time = time.time()
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    plt.savefig(f'../img/{str(mod_name)}_impurity_based_feature_importance')

    print(f"Elapsed time to compute impurity-based importances: {elapsed_time:.3f} seconds")

def permutation_feature_importance(model, X_test, y_test, mod_name):
    '''
    This function computes the feature importance by using the permutation method, which randomly shuffle each feature
    and compute the change in the model’s performance. The features which impact the performance the most are the most important.

    The function exploits the permutation_importance method from sklearn.inspection module.
    https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance

    The Mean Decrease Accuracy expresses how much accuracy the model losses by excluding each variable.
    The more the accuracy suffers, the more important the variable is for the successful regression.

    :param model:
    :param X_test:
    :param y_test:
    :param mod_name:
    :return:
    '''

    suffix = '' if include_sci else '_noSCI'
    if feature_selection == False:
        suffix += '_no_feat_sel'

    fig, ax = plt.subplots(figsize=(18, 10))

    feature_names = X_test.columns.values
    start_time = time.time()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute permutation importances: {elapsed_time:.3f} seconds")

    importances = pd.Series(result.importances_mean, index=feature_names)

    importances.plot.barh(xerr=result.importances_std)
    plt.title(f"{str(mod_name)}{str(suffix)}")
    plt.suptitle("Feature importances using permutation on full model")
    plt.ylabel("Mean accuracy decrease")

    plt.savefig(f'../img/{str(mod_name)}_permutation_feature_importance{str(suffix)}')
