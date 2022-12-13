'''
    OLS linear regression
'''
import json

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, Normalizer, normalize

from dataset_prep import prepare_dataset
from IPython.display import display
from utilities import forwardSelection, __forwardSelectionRaw__, backwardSelection, __backwardSelectionRaw__, __varcharProcessing__, linear_regression_ols
from params import choose_params, online_path, offline_path, run_ols, include_sci

''' Display options for prints '''
#pd.option_context('display.max_rows', 50, 'display.height', 10)
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

''' Importing data'''
print('Importing MIMI dataset (v2.0 of Zenodo)...')
path = offline_path
mimi = pd.read_csv(path, sep=",", low_memory=False)
#print(mimi.info(verbose=True, null_counts=True))
#print(mimi.head(5))
print('Dataset loaded.')

''' Preparing data'''
mimi, mimi_tot_indices = prepare_dataset(mimi)

''' Migr indices columns grouping '''

# TOTAL FLOWS

BMP_cols = [col for col in mimi_tot_indices if col.startswith('TOT_flow_BMP')]
BMI_cols = [col for col in mimi_tot_indices if col.startswith('TOT_flow_BMI')]
UMP_cols = [col for col in mimi_tot_indices if col.startswith('TOT_flow_UMP')]
UMI_cols = [col for col in mimi_tot_indices if col.startswith('TOT_flow_UMI')]

# Separate columns (indices) for UN and ESTAT flows

ESTAT_BMP_cols = [col for col in mimi if col.startswith('ESTAT_flow_BMP')]
ESTAT_BMI_cols = [col for col in mimi if col.startswith('ESTAT_flow_BMI')]
ESTAT_UMP_cols = [col for col in mimi if col.startswith('ESTAT_flow_UMP')]
ESTAT_UMI_cols = [col for col in mimi if col.startswith('ESTAT_flow_UMI')]

UN_BMP_cols = [col for col in mimi if col.startswith('UN_flow_BMP')]
UN_BMI_cols = [col for col in mimi if col.startswith('UN_flow_BMI')]
UN_UMP_cols = [col for col in mimi if col.startswith('UN_flow_UMP')]
UN_UMI_cols = [col for col in mimi if col.startswith('UN_flow_UMI')]

# Stocks

stocks_BMI_cols = [col for col in mimi if col.startswith('stocks_BMI_')]
stocks_BMP_cols = [col for col in mimi if col.startswith('stocks_BMP_')]
stocks_UMI_cols = [col for col in mimi if col.startswith('stocks_UMI_')]
stocks_UMP_cols = [col for col in mimi if col.startswith('stocks_UMP_')]

''' Remove infs and nan '''

print('Drop records having nan values from dataframe... ')

# Total Flows

for col in BMP_cols:
    mimi_not_null_BMP = mimi_tot_indices.loc[(~mimi_tot_indices[col].isna())]
for col in BMI_cols:
    mimi_not_null_BMI = mimi_tot_indices.loc[(~mimi_tot_indices[col].isna())]
for col in UMP_cols:
    mimi_not_null_UMP = mimi_tot_indices.loc[(~mimi_tot_indices[col].isna())]
for col in UMI_cols:
    mimi_not_null_UMI = mimi_tot_indices[(~mimi_tot_indices[col].isna())]

ESTAT_BMP_cols_cit = [col for col in ESTAT_BMP_cols if col.endswith('cit')]
ESTAT_BMP_cols_res = [col for col in ESTAT_BMP_cols if col.endswith('res')]
ESTAT_BMI_cols_cit = [col for col in ESTAT_BMI_cols if col.endswith('cit')]
ESTAT_BMI_cols_res = [col for col in ESTAT_BMI_cols if col.endswith('res')]

ESTAT_UMP_cols_cit = [col for col in ESTAT_UMP_cols if col.endswith('cit')]
ESTAT_UMP_cols_res = [col for col in ESTAT_UMP_cols if col.endswith('res')]
ESTAT_UMI_cols_cit = [col for col in ESTAT_UMI_cols if col.endswith('cit')]
ESTAT_UMI_cols_res = [col for col in ESTAT_UMI_cols if col.endswith('res')]

UN_BMP_cols_cit = [col for col in UN_BMP_cols if col.endswith('cit')]
UN_BMP_cols_res = [col for col in UN_BMP_cols if col.endswith('res')]
UN_BMI_cols_cit = [col for col in UN_BMI_cols if col.endswith('cit')]
UN_BMI_cols_res = [col for col in UN_BMI_cols if col.endswith('res')]

UN_UMP_cols_cit = [col for col in UN_UMP_cols if col.endswith('cit')]
UN_UMP_cols_res = [col for col in UN_UMP_cols if col.endswith('res')]
UN_UMI_cols_cit = [col for col in UN_UMI_cols if col.endswith('cit')]
UN_UMI_cols_res = [col for col in UN_UMI_cols if col.endswith('res')]

# ESTAT flows

for col in ESTAT_BMP_cols_cit:
    mimi_not_null_ESTAT_BMP_cit = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_BMI_cols_cit:
    mimi_not_null_ESTAT_BMI_cit = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_UMP_cols_cit:
    mimi_not_null_ESTAT_UMP_cit = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_UMI_cols_cit:
    mimi_not_null_ESTAT_UMI_cit = mimi[(~mimi[col].isna())]

for col in ESTAT_BMP_cols_res:
    mimi_not_null_ESTAT_BMP_res = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_BMI_cols_res:
    mimi_not_null_ESTAT_BMI_res = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_UMP_cols_res:
    mimi_not_null_ESTAT_UMP_res = mimi.loc[(~mimi[col].isna())]
for col in ESTAT_UMI_cols_res:
    mimi_not_null_ESTAT_UMI_res = mimi[(~mimi[col].isna())]

# UN flows

for col in UN_BMP_cols_cit:
    mimi_not_null_UN_BMP_cit = mimi.loc[(~mimi[col].isna())]
for col in UN_BMI_cols_cit:
    mimi_not_null_UN_BMI_cit = mimi.loc[(~mimi[col].isna())]
for col in UN_UMP_cols_cit:
    mimi_not_null_UN_UMP_cit = mimi.loc[(~mimi[col].isna())]
for col in UN_UMI_cols_cit:
    mimi_not_null_UN_UMI_cit = mimi[(~mimi[col].isna())]

for col in UN_BMP_cols_res:
    mimi_not_null_UN_BMP_res = mimi.loc[(~mimi[col].isna())]
for col in UN_BMI_cols_res:
    mimi_not_null_UN_BMI_res = mimi.loc[(~mimi[col].isna())]
for col in UN_UMP_cols_res:
    mimi_not_null_UN_UMP_res = mimi.loc[(~mimi[col].isna())]
for col in UN_UMI_cols_res:
    mimi_not_null_UN_UMI_res = mimi[(~mimi[col].isna())]


''' Select sub-datasets with non null Y '''

# ESTAT flows

mimi_not_null_ESTAT_BMP_res = mimi_not_null_ESTAT_BMP_res.loc[(~mimi['ESTAT_flow_BMP_2019_T_T_res'].isna())]
mimi_not_null_ESTAT_BMP_cit = mimi_not_null_ESTAT_BMP_cit.loc[(~mimi['ESTAT_flow_BMP_2019_T_T_cit'].isna())]
mimi_not_null_ESTAT_BMI_res = mimi_not_null_ESTAT_BMI_res.loc[(~mimi['ESTAT_flow_BMI_2019_T_T_res'].isna())]
mimi_not_null_ESTAT_BMI_cit = mimi_not_null_ESTAT_BMI_cit.loc[(~mimi['ESTAT_flow_BMI_2019_T_T_cit'].isna())]

mimi_not_null_ESTAT_UMP_res = mimi_not_null_ESTAT_UMP_res.loc[(~mimi['ESTAT_flow_UMP_2019_T_T_res'].isna())]
mimi_not_null_ESTAT_UMP_cit = mimi_not_null_ESTAT_UMP_cit.loc[(~mimi['ESTAT_flow_UMP_2019_T_T_cit'].isna())]
mimi_not_null_ESTAT_UMI_res = mimi_not_null_ESTAT_UMI_res.loc[(~mimi['ESTAT_flow_UMI_2019_T_T_res'].isna())]
mimi_not_null_ESTAT_UMI_cit = mimi_not_null_ESTAT_UMI_cit.loc[(~mimi['ESTAT_flow_UMI_2019_T_T_cit'].isna())]

# UN flows

mimi_not_null_UN_BMP_res = mimi_not_null_UN_BMP_res.loc[(~mimi['UN_flow_BMP_2019_T_T_res'].isna())]
mimi_not_null_UN_BMP_cit = mimi_not_null_UN_BMP_cit.loc[(~mimi['UN_flow_BMP_2019_T_T_cit'].isna())]
mimi_not_null_UN_BMI_res = mimi_not_null_UN_BMI_res.loc[(~mimi['UN_flow_BMI_2019_T_T_res'].isna())]
mimi_not_null_UN_BMI_cit = mimi_not_null_UN_BMI_cit.loc[(~mimi['UN_flow_BMI_2019_T_T_cit'].isna())]

mimi_not_null_UN_UMP_res = mimi_not_null_UN_UMP_res.loc[(~mimi['UN_flow_UMP_2019_T_T_res'].isna())]
mimi_not_null_UN_UMP_cit = mimi_not_null_UN_UMP_cit.loc[(~mimi['UN_flow_UMP_2019_T_T_cit'].isna())]
mimi_not_null_UN_UMI_res = mimi_not_null_UN_UMI_res.loc[(~mimi['UN_flow_UMI_2019_T_T_res'].isna())]
mimi_not_null_UN_UMI_cit = mimi_not_null_UN_UMI_cit.loc[(~mimi['UN_flow_UMI_2019_T_T_cit'].isna())]

''' Select dependent variables (Y to predict) '''

Y_UMP_res = mimi_not_null_ESTAT_UMP_res['ESTAT_flow_UMP_2019_T_T_res'].values.reshape(-1, 1)
Y_UMP_cit = mimi_not_null_ESTAT_UMP_cit['ESTAT_flow_UMP_2019_T_T_cit'].values.reshape(-1, 1)
Y_UMI_res = mimi_not_null_ESTAT_UMI_res['ESTAT_flow_UMI_2019_T_T_res'].values.reshape(-1, 1)
Y_UMI_cit = mimi_not_null_ESTAT_UMI_cit['ESTAT_flow_UMI_2019_T_T_cit'].values.reshape(-1, 1)
Y_BMP_res = mimi_not_null_ESTAT_BMP_res['ESTAT_flow_BMP_2019_T_T_res'].values.reshape(-1, 1)
Y_BMP_cit = mimi_not_null_ESTAT_BMP_cit['ESTAT_flow_BMP_2019_T_T_cit'].values.reshape(-1, 1)
Y_BMI_res = mimi_not_null_ESTAT_BMI_res['ESTAT_flow_BMI_2019_T_T_res'].values.reshape(-1, 1)
Y_BMI_cit = mimi_not_null_ESTAT_BMI_cit['ESTAT_flow_BMI_2019_T_T_cit'].values.reshape(-1, 1)

''' Select independent features (X matrix) '''
#gdp_orig_cols = [col for col in mimi if col.startswith('origin_gdp')]
#gdp_dest_cols = [col for col in mimi if col.startswith('destination_gdp')]
pdi_cols = [col for col in mimi if col.endswith('PDI')]
idv_cols = [col for col in mimi if col.endswith('IDV')]
uai_cols = [col for col in mimi if col.endswith('UAI')]
mas_cols = [col for col in mimi if col.endswith('MAS')]
lto_cols = [col for col in mimi if col.endswith('LTO')]
area_cols = [col for col in mimi if col.endswith('area')]
users_cols = [col for col in mimi if col.endswith('users')]
perc_users_cols = [col for col in mimi if col.endswith('perc')]

X_cols = pdi_cols+idv_cols+uai_cols+mas_cols+area_cols+users_cols+perc_users_cols

X_cols.append('geodesic_distance_km')
X_cols.append('origin_gdp_2018')
X_cols.append('destination_gdp_2018')
X_cols.append('gdp_diff_2018')
X_cols.append('neighbors')
X_cols.append('share_cont')
X_cols.append('share_rel')
X_cols.append('share_lang')
X_cols.append('sci_2020')
X_cols.append('from_to')

print('Importing chosen parameters and selecting respective data ...')
df_str, y = choose_params()
df = eval(df_str)

X_cols.append(y)

# select X matrix
X = df[X_cols]
X = X.set_index('from_to') # matrix index

''' MODIFICHE FEATURES PER INDICI BIDIREZIONALI '''

X_new = X[['geodesic_distance_km','gdp_diff_2018']]

if include_sci:
    X_new['sci_2020'] = X['sci_2020']

X_new = X_new.join(X[y])

X_new['gdp_diff_2018'] = X_new['gdp_diff_2018'].abs()
X_new['gdp_mean_2018'] = X[["origin_gdp_2018", "destination_gdp_2018"]].mean(axis=1)

X_new['neighbors'] = X['neighbors']
X_new['share_rel'] = X['share_rel']
X_new['share_lang'] = X['share_lang']

X_new['PDI_diff'] = X.origin_PDI - X.destination_PDI
X_new['PDI_diff'] = X_new['PDI_diff'].abs()

X_new['IDV_diff'] = X.origin_IDV - X.destination_IDV
X_new['IDV_diff'] = X_new['IDV_diff'].abs()

X_new['UAI_diff'] = X.origin_UAI - X.destination_UAI
X_new['UAI_diff'] = X_new['UAI_diff'].abs()

X_new['MAS_diff'] = X.origin_MAS - X.destination_MAS
X_new['MAS_diff'] = X_new['MAS_diff'].abs()

#X_new['LTO'] = X.origin_LTO - X.destination_LTO
#X_new['LTO'] = X_new['LTO'].abs()

X_new['fb_users_diff'] = X.origin_fb_users - X.destination_fb_users
X_new['fb_users_diff'] = X_new['fb_users_diff'].abs()

X_new['fb_users_perc_diff'] = X.origin_fb_users_perc - X.destination_fb_users_perc
X_new['fb_users_perc_diff'] = X_new['fb_users_perc_diff'].abs()

X_new['fb_users_perc_mean'] = X[["origin_fb_users_perc", "destination_fb_users_perc"]].mean(axis=1)
X_new['fb_users_mean'] = X[["origin_fb_users", "destination_fb_users"]].mean(axis=1)

X_new['area_diff'] = X.origin_area - X.destination_area
X_new['area_diff'] = X_new['area_diff'].abs()

X_new['area_mean'] = X[["origin_area", "destination_area"]].mean(axis=1)

X_new['share_cont'] = X['share_cont']

X=X_new


# eventually drop some columns
#X.drop(['origin_IDV','destination_IDV', 'gdp_diff_2018'], axis=1, inplace=True)

# drop nan observations for OLS model
X = X.dropna()

# extract and reshape vector Y to predict
Y_orig = X[y] # for stepwise
Y = X[y].values.reshape(-1, 1) # rescale indicator to perform linear regression # needed because of: ValueError: The indices for endog and exog are not aligned

X_orig = X.copy()


''' Linear regression model '''
vars_storage = "../dump/OLS_selected_variables.json"
if run_ols:
    selected_variables = linear_regression_ols(df, X, Y, y) # run OLS
    json.dump(selected_variables, open(vars_storage, "w")) # save OLS result
else:
    selected_variables = json.load(open(vars_storage, "r")) # load OLS results previously computed stored in json file